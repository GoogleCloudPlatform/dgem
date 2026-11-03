#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""dgem decisions in a Dataflow (Apache Beam) batch pipeline with RunInference.

Beam's built-in VLLMCompletionsModelHandler starts stock vLLM and sends free-form completion prompts, which cannot
serve dgem (it needs the pinned vLLM build and structured_server.py's readout). DgemModelHandler instead starts the
dgem image's own entrypoint (vLLM + structured_server.py) once per worker in load_model() and sends each batch to it
concurrently, the same way batch_worker.py does on the other platforms.

  JSONL (GCS) -> parse -> Reshuffle -> RunInference(DgemModelHandler) -> JSONL (GCS)

Run one SDK process per worker (--experiments=no_use_multiple_sdk_containers) so the 26B model loads once per GPU.
The schema is rendered once with `dgem template render` before launch and read by every worker.

Launch: see deploy/batch/README.md.
"""
import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.metrics import Metrics
from apache_beam.ml.inference.base import ModelHandler, PredictionResult, RunInference
from apache_beam.options.pipeline_options import PipelineOptions

import batch_worker as bw

NS = "dgem"


class DgemModel:
    """What load_model() returns: a warm local dgem server plus a request pool."""

    def __init__(self, decider, concurrency):
        self.decider = decider
        self.pool = ThreadPoolExecutor(concurrency)


class DgemModelHandler(ModelHandler[dict, PredictionResult, DgemModel]):
    def __init__(self, schema_uri, state_key="comment", concurrency=32, min_batch=32, max_batch=256, server_env=None):
        super().__init__()
        self.schema_uri, self.state_key, self.concurrency = schema_uri, state_key, concurrency
        self.min_batch, self.max_batch = min_batch, max_batch
        self.server_env = server_env or {}
        self._saw_first = False

    def load_model(self) -> DgemModel:
        t0 = time.time()
        logging.info("dgem: starting local server, gpu=%s", bw.gpu_info())
        srv = bw.LocalServer(env=self.server_env).start()
        srv.wait_ready()
        Metrics.gauge(NS, "server_ready_s").set(int(time.time() - t0))
        with FileSystems.open(self.schema_uri) as f:
            schema = f.read().decode()
        return DgemModel(bw.Decider(srv.url, schema, self.state_key), self.concurrency)

    def run_inference(self, batch, model: DgemModel, inference_args=None):
        out = model.decider.decide_many(batch, model.pool)
        if not self._saw_first and out:
            Metrics.gauge(NS, "first_decision_unix").set(int(time.time()))
            self._saw_first = True
        errs = sum(1 for r in out if "error" in r)
        Metrics.counter(NS, "decisions_ok").inc(len(out) - errs)
        Metrics.counter(NS, "decisions_error").inc(errs)
        return [PredictionResult(x, y) for x, y in zip(batch, out)]

    def batch_elements_kwargs(self):
        return {"min_batch_size": self.min_batch, "max_batch_size": self.max_batch}

    def share_model_across_processes(self):
        return False


def run(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="gs://.../part-*.jsonl")
    ap.add_argument("--output", required=True, help="gs://.../out/part")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--state_key", default="comment")
    ap.add_argument("--concurrency", type=int, default=32, help="requests in flight per batch")
    ap.add_argument("--max_batch", type=int, default=256)
    ap.add_argument("--server_env", default="{}", help="JSON env for the dgem server (KV_CACHE_GB, MAX_INFLIGHT, ...)")
    a, rest = ap.parse_known_args(argv)
    handler = DgemModelHandler(a.schema, a.state_key, a.concurrency, max_batch=a.max_batch, server_env=json.loads(a.server_env))
    with beam.Pipeline(options=PipelineOptions(rest)) as p:
        (p
         | "Read" >> beam.io.ReadFromText(a.input)
         | "Parse" >> beam.Map(json.loads)
         | "Spread" >> beam.Reshuffle()
         | "Decide" >> RunInference(handler)
         | "Format" >> beam.Map(lambda r: json.dumps(r.inference, ensure_ascii=False))
         | "Write" >> beam.io.WriteToText(a.output, file_name_suffix=".jsonl"))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
