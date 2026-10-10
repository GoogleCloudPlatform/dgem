---
title: "Regression Matrix"
description: "Run a fixed set of benchmarks against dgem serving endpoints and compare a candidate with your own baseline in the same session: tiers by change type, how to run and read it, and pass rules based on the noise each run measures."
---

One command runs a fixed set of benchmarks against one or more dgem serving endpoints and says whether anything
changed: `scripts/bench_matrix.py`. Use it for every serving-image change, release, configuration change or prompt
change that could move answers. You compare a **candidate** (a new image, configuration or deployment) with a
**baseline** you trust (usually your current deployment) in the same session; the runner measures the noise floor
itself and judges the difference against it.

The tiers and suites are defined in a matrix file. The repository ships
[`benchmarks/matrix/matrix_example.json`](https://github.com/GoogleCloudPlatform/dgem/blob/main/benchmarks/matrix/matrix_example.json), the default for
`scripts/bench_matrix.py`. It has no reference ranges. Copy and edit it to add your own suites or change thresholds,
and pass it with `--matrix <file>`. Keep tier contents stable once you rely on them: if you change what a tier
contains, save it as a new file so older runs stay comparable.

## Which tier to run

| Tier | When | Suites in the example matrix |
| :--- | :--- | :--- |
| **T0 smoke** | Every deploy or configuration change | `/health`, API contract (including multilingual cases with known answers), calibration suite ×1, multilingual spot check (MASSIVE **validation**, ru/th/hi/ja/es × 20). About 200 requests |
| **T1 gate** | Serving-image, prompt-path or server changes; release validation | T0 + calibration ×3, JevBench ×3 through `bench-jev` (chat/completions) and ×3 through `/v1/systemone`, intents slices ×3 (banking77, clinc150), the [Decision Index adapter track](#decision-index-adapter-track), image decisions, latency (5 modes × 100, sweep at 16 and 32 workers) |
| **T2 full** | Release candidates, vLLM or model changes; needs `--confirm` | T1 + MASSIVE test (51 languages × 100), XNLI test (15 × 300), typed-decisions test (2,000 decisions), option order (739 cases × 4 orderings), bounding boxes, the Decision Index panel, and the Decision Index kit's compatibility pass when configured. Tens of thousands of requests per target |

`scripts/bench_matrix.py list` prints the tiers and suites of the matrix file. Every report also computes ECE and
NLL with a 5-fold **held-out** temperature from the saved receipts, without extra requests.

Some suites need files that are not in the repository; they are listed with `requires` in the matrix file. The
JevBench suites (`jev_native`, `jev_systemone`, the JevBench part of `order`) need the public items:

```bash
./bin/dgem bench-jev --sync
```

The image suites (`vision_spot`, `vision`) read a seeded synthetic sweep that the runner generates on first use
(`pip install -r scripts/requirements-vision.txt`).

The suites that read parquet datasets need `pyarrow` (`pip install -r scripts/requirements-matrix.txt`): `di_catchall`
(CLINC150) and `rag_dev` (RAGTruth) in T1, plus `xnli`, `typed` and `order` (XNLI, typed-decisions, emotion) in T2. T0
needs nothing extra. Without it, a T1 or T2 run stops before it sends any request and names those suites: install it,
or leave them out with `--skip`. `--plan` prints the same message.

## Run it

```bash
make build                                   # bin/dgem: the dgem bench-* suites shell out to it
pip install -r scripts/requirements-matrix.txt   # pyarrow, for the suites that read parquet (T1 and T2; not T0)
scripts/bench_matrix.py list                 # tiers and suites

# Compare a candidate with your current deployment in the same session
scripts/bench_matrix.py run --tier T1 \
  --target base=<CURRENT_URL> --target new=<CANDIDATE_URL> --baseline base --label image-<TAG>

# Smoke-check one target (results are INFO only without a baseline)
scripts/bench_matrix.py run --tier T0 --target gpu=http://<GPU_HOST>:8080

# Full matrix
scripts/bench_matrix.py fetch                # download and verify the pinned datasets once
scripts/bench_matrix.py run --tier T2 --confirm \
  --target base=<CURRENT_URL> --target new=<CANDIDATE_URL> --baseline base

# See what would run without sending requests
scripts/bench_matrix.py run --tier T1 --target x=<URL> --plan
```

- **Target URLs:** a Vertex dedicated endpoint invoke base
  (`https://<ENDPOINT_ID>.<REGION>-<PROJECT_NUMBER>.prediction.vertexai.goog/v1/projects/<PROJECT>/locations/<REGION>/endpoints/<ENDPOINT_ID>/invoke`),
  a Cloud Run URL (`<CLOUD_RUN_URL>`; tagged revision URLs work, so a `--no-traffic` canary can be tested), or
  `http://<GPU_HOST>:8080`.
- **Auth:** Vertex gets an access token and Cloud Run an ID token, from the metadata server, Application Default
  Credentials or gcloud. Self-hosted servers get none, unless you set `DGEM_MATRIX_TOKEN` (for example the
  server's `API_KEY`).
- **Single target:** without `--baseline` (and with no reference ranges in the matrix file) per-suite results are
  INFO: the run reports accuracy, calibration, coverage and latency, while checks that need no comparison (health,
  the contract, the multilingual minimum, the adapter probes) still pass or fail. To get verdicts on accuracy and
  latency, run a baseline next to the candidate.
- **Interleaving:** with two targets, every repeated suite alternates targets run by run, so both see the same
  conditions. Latency runs last, so it is not measured under the matrix's own load.
- **Request options (A/B on one deployment):** append `#option=value` to a target URL to add a request option to
  every `/v1/systemone` and chat body that target gets, and the matching flag to its local adapter. Today:
  `layout=document_first` or `layout=schema_first` (`--prompt-layout` on the adapter; default `document_first`
  from v0.2.0). For example `--target base=<URL> --target old=<URL>#layout=schema_first --baseline base`. Template
  suites (`dgem decide`) don't get the option.
- **Selecting suites:** `--only <suite> ...` or `--skip <suite> ...`; `--workers` sets workers per harness.
- **Resuming:** an interrupted run continues with `--resume` and the same `--label`.
- **Exit code:** 1 if any target's overall verdict is FAIL, so the command can gate a script.

### Output

Each run is a directory named `<YYYYMMDD>-<label>` under the repository's `benchmarks` folder by default (`--out-dir` to change it):

| File | Contents |
| :--- | :--- |
| `manifest.json` | matrix version, tier, targets with the `version` / `revision` / `vllm_commit` each reported at `/health`, git commit, one entry per receipt with its SHA-256 |
| `<suite>__<target>__r<N>.json` | receipts: the `dgem bench-*` receipt as written, or per-item `/v1/systemone` rows |
| `report.md` | verdicts, per-suite tables, the measured noise floor, raw vs held-out calibration, per-language tables, latency |
| `summary.json` | the machine-readable result ([schema](https://github.com/GoogleCloudPlatform/dgem/blob/main/benchmarks/matrix/summary.schema.json), `dgem.matrix.summary/v2`): verdicts, the noise floor, and per suite and target every run's accuracy, coverage, macro-F1, ECE, Brier, NLL, AUROC and latency, plus held-out calibration and reliability bins; option-order flip rates; latency modes and sweep. Automation should read this file, not `report.md` |

`scripts/bench_matrix.py report <run_dir>` rebuilds `report.md` and `summary.json` from the receipts. Receipts and
logs are redacted as they are written: endpoint URLs become `<kind:name>` and tokens `<TOKEN>`. `--upload
gs://<bucket>/<prefix>` copies the run directory to Cloud Storage when it finishes, and `--log-json` prints one JSON
event line per step, for running the matrix as a job.

### Metrics in the report

- **Case-exact:** for suites with several questions per case (typed-decisions), the share of cases with every field
  right. Field accuracy alone can hide cases that are only partly right.
- **Coverage:** answered / attempted items. Refusals (for example HTTP 422 when a request exceeds a capacity limit)
  and errors count as unanswered. Accuracy and the calibration metrics are computed over answered items, so read them
  together with coverage.
- **Macro-F1:** F1 averaged over gold labels, with labels kept separate per question in multi-question suites. Intent
  benchmarks such as BANKING77 and CLINC150 are usually reported this way, so use it when comparing with published
  numbers; accuracy alone favours frequent labels.
- **ECE, Brier, NLL, AUROC:** confidence quality: how well stated confidence matches accuracy (ECE), proper scores for
  the whole distribution (Brier, NLL), and how well confidence separates right from wrong answers (AUROC). Held-out ECE
  uses a temperature fitted on other folds.

## How verdicts are decided

The model is not deterministic: a few percent of answers can change between identical requests on the same image. A
single run's score therefore says little on its own. The matrix measures that noise in every run (agreement between
repeated runs on the same target) and judges changes against it, so don't quote a fixed noise band.

| Gate | PASS when | Otherwise |
| :--- | :--- | :--- |
| Health | `/health` is `ok` and vLLM is ready | FAIL |
| Contract | every case returns the expected status; no multilingual case answers wrong; status codes match the baseline | FAIL |
| Accuracy suites | answer agreement between candidate and baseline is at least the measured within-target agreement minus 2 points, **and** a paired McNemar test is not significant (p ≥ 0.01) | REVIEW if one fails, FAIL if both |
| Multilingual spot check | every language at least 70% | FAIL |
| Coverage | every accuracy suite answers as many items as the baseline. Unanswered items are classified as `context` (prompt longer than the served context), `capacity` (server shape limits), `na` (skipped by the matrix) or `error`, in the report and in `summary.json` (`refusals`) | REVIEW |
| Option order | net flip rate (permuted minus identical-repeat) no more than 0.05 above the baseline | REVIEW |
| Latency | every mode's p50 within ×1.10 of the baseline, throughput at least ×0.90, 0 errors | REVIEW up to ×1.25, FAIL beyond or on errors |
| Bounding boxes, Decision Index panel | reported, not gated (INFO) | — |

Thresholds live in the matrix file (`thresholds`). Suites marked `"accuracy": true` are scored per item (accuracy,
calibration, coverage and the noise floor); a new per-item suite only needs that flag, not a code change. Look at a
REVIEW before promoting; a FAIL blocks.

A matrix file may also carry a `reference` section with accuracy ranges you measured on your own deployment; single-
target runs are then judged against those ranges. The example matrix has none: build your own from several baseline
runs if you want single-target verdicts.

Accuracy is computed over answered items, so the coverage gate is what catches an item that starts being refused, for
example a prompt that a new prompt layout or a smaller `MAX_MODEL_LEN` pushes past the served context length.

Latency depends on hardware, region and network, so compare it only between targets measured in the same session
from the same client.

## Decision Index adapter track

Decision Index-format suites go through `dgem systemone serve`, the adapter that splits questions with more than 26
options into brackets, batches more than 8 questions per request, and turns capacity limits into HTTP 422 responses.
The matrix starts that adapter locally, from this checkout's `bin/dgem`, in front of each target. These suites
therefore test the adapter code you are about to ship against the deployed model:

| Suite | Checks |
| :--- | :--- |
| `di_probes` | Wide options at K = 27, 41, 61, 101, 151 and 255 (right answer; probabilities for every option, summing to 1); a 12-question request (slot batching); option descriptions given as JSON objects and arrays; a ~9k-token and a ~31k-token input that must be answered, or refused with 422 plus a capacity marker (`maximum context length`, `options per choice`, …). Any miss is a FAIL. Some informational probes are reported but not gated, including the top probability on unambiguous wide-option items: a flat ceiling across K would mean bracket fusion is capping confidence |
| `di_wide` | MASSIVE validation (English) with the full ~60-label set as options, 100 items, judged like the other accuracy suites |
| `di_catchall` | CLINC150 **validation**: 151 options including an "out of scope" catch-all, 80 in-scope + 20 out-of-scope items. Reports (INFO) how often in-scope requests are answered "out of scope", a known weakness of bracket routing. `dgem systemone serve --catch-all final` trades out-of-scope recall for in-scope accuracy |
| `rag_dev` | RAGTruth **train**, 198 items, a hallucination question with true/false criteria. Reports (INFO) F1 and recall on the hallucinated class against the always-flag baseline, which shows a yes/no "no" bias. `--noul-mode choice` on the adapter raises recall on this family |
| `di_kit_compat` (T2) | The Decision Index kit's own compatibility pass (86 requests) through the adapter. Runs only when `DGEM_DI_KIT_DIR` (a kit checkout with its `.venv`) and `DGEM_DI_COMPAT_ROWS` (compatibility rows) are set; otherwise reported as skipped. FAIL on any `error` row |

## Frozen held-out sets

MASSIVE test, XNLI test, typed-decisions test and the option-order suites run only in T2. Score them only after a
change ships, and never use them to tune prompts, wording or thresholds. Do development and wording work on
JevBench, the calibration and intents suites, and validation splits (the multilingual spot check reads MASSIVE
*validation*). This keeps the T2 numbers honest: a change that only helps because it was tuned on the test items
cannot show up there.

The runtime datasets are listed in [`benchmarks/matrix/datasets.lock.json`](https://github.com/GoogleCloudPlatform/dgem/blob/main/benchmarks/matrix/datasets.lock.json),
each file pinned to a Hugging Face dataset commit and checked against its SHA-256 before use
(`scripts/bench_matrix.py fetch`). The files are cached in `~/.cache/dgem-matrix` (`DGEM_MATRIX_CACHE`) and never
committed. Check each dataset's licence on its dataset card before use (for example MASSIVE CC BY 4.0,
typed-decisions Apache-2.0, XNLI CC BY-NC 4.0). Anonymous Hugging Face downloads can be rate-limited; set `HF_TOKEN`
if they are.

## What the matrix tells you about calibration and order

- **Calibration needs a per-domain temperature.** The held-out temperature the report fits differs between suites;
  raw confidence is too high on some domains. A single global temperature does not fit every domain; fit one on
  held-out data from your own decisions ([Calibrate your policy](/dgem/confidence/calibrate-your-policy/)).
- **Option order** (T2) reports the net flip rate (shuffled minus identical repeat) per suite and how often the
  first-shown option is picked against its gold rate. Order effects are monitored here rather than corrected per
  request.

## Related

- [Operations runbook: promote a new serving image](/dgem/operate/runbook/#promote-a-new-serving-image) (uses T1)
- [Evaluate dgem on your own GPU](/dgem/deploy/evaluate/) (uses T0 and T1)
- [Monitoring and alerts](/dgem/operate/monitoring/) (the hourly synthetic probe is separate and lighter than T0)
