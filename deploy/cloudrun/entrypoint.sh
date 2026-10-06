#!/usr/bin/env bash
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

set -euo pipefail

MODEL="${MODEL:-/mnt/gcs/dgemma}"
CANVAS="${CANVAS:-128}"
PORT="${AIP_HTTP_PORT:-${PORT:-8080}}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISABLE_MM="${DISABLE_MM:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

ROLE="${ROLE:-default}"
TEMPERATURE="${TEMPERATURE:-1.0}"
API_KEY="${API_KEY:-}"

# 1. Resolve model weights
if [ -d "/opt/dgemma/weights" ] && [ -f "/opt/dgemma/weights/config.json" ]; then
  echo "[init] Using baked NVFP4 model weights at /opt/dgemma/weights (0.0s download)..."
  MODEL="/opt/dgemma/weights"
elif [ -d "$MODEL" ] && [ -f "$MODEL/config.json" ]; then
  echo "[init] Using mounted model weights at $MODEL..."
elif [ -n "${DGEM_WEIGHTS_URI:-}" ] || [ -n "${AIP_STORAGE_URI:-}" ]; then
  export GCS_URI="${DGEM_WEIGHTS_URI:-${AIP_STORAGE_URI}}"
  echo "[init] Staging weights from GCS: $GCS_URI to /tmp/dgemma..."
  mkdir -p /tmp/dgemma
  python3 -c '
import json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
uri = os.environ["GCS_URI"].rstrip("/")
bucket, _, prefix = uri[5:].partition("/")
req = urllib.request.Request(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    headers={"Metadata-Flavor": "Google"}
)
token = json.loads(urllib.request.urlopen(req, timeout=5).read().decode())["access_token"]
list_url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={prefix}/"
items = json.loads(urllib.request.urlopen(urllib.request.Request(list_url, headers={"Authorization": f"Bearer {token}"})).read().decode()).get("items", [])
HDR_BYTES = 4 * 1024 * 1024
manifest = []
def stage_initial(item):
    name = item["name"]
    rel = name[len(prefix)+1:]
    if not rel or "/" in rel:
        return
    sz = int(item.get("size", 0))
    dst = os.path.join("/tmp/dgemma", rel)
    dl_url = f"https://storage.googleapis.com/{bucket}/{name}"
    if rel.endswith(".safetensors"):
        end = min(HDR_BYTES, sz) - 1
        rreq = urllib.request.Request(dl_url, headers={"Authorization": f"Bearer {token}", "Range": f"bytes=0-{end}"})
        with urllib.request.urlopen(rreq, timeout=30) as r, open(dst, "wb") as f:
            f.write(r.read())
            f.truncate(sz)
        manifest.append({"file": rel, "size": sz})
    else:
        rreq = urllib.request.Request(dl_url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(rreq, timeout=30) as r, open(dst, "wb") as f:
            f.write(r.read())
with ThreadPoolExecutor(max_workers=16) as ex:
    list(ex.map(stage_initial, items))
with open("/tmp/dgemma/.gcs_manifest.json", "w") as mf:
    json.dump({"bucket": bucket, "prefix": prefix, "files": manifest}, mf)
'
  cat << 'EOF' > /tmp/sitecustomize.py
import os, time
try:
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    _orig_load_weights = DefaultModelLoader.load_weights
    def _waiting_load_weights(self, model, model_config):
        if str(getattr(model_config, "model", "")).startswith("/tmp/dgemma") and not os.path.exists("/tmp/dgemma/.ready"):
            print("[init] EngineCore reached DefaultModelLoader.load_weights; waiting for /tmp/dgemma/.ready...", flush=True)
            while not os.path.exists("/tmp/dgemma/.ready"):
                time.sleep(0.2)
            print("[init] /tmp/dgemma/.ready confirmed! Loading 17.53 GiB weights via zero-copy mmap...", flush=True)
        return _orig_load_weights(self, model, model_config)
    DefaultModelLoader.load_weights = _waiting_load_weights
except Exception:
    pass
EOF
  export PYTHONPATH="/tmp:${PYTHONPATH:-}"
  export COPY_TO_SHM=1
  (
    set -e
    python3 -c '
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
req = urllib.request.Request(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    headers={"Metadata-Flavor": "Google"}
)
token = json.loads(urllib.request.urlopen(req, timeout=5).read().decode())["access_token"]
with open("/tmp/dgemma/.gcs_manifest.json") as mf:
    meta = json.load(mf)
bucket, prefix = meta["bucket"], meta["prefix"]
hdr = 4 * 1024 * 1024
chunk = 64 * 1024 * 1024
tasks = []
for fmeta in meta["files"]:
    bn, sz = fmeta["file"], fmeta["size"]
    df = os.path.join("/tmp/dgemma", bn)
    for off in range(min(hdr, sz), sz, chunk):
        tasks.append((bucket, prefix, token, bn, df, off, min(chunk, sz - off)))
def dl(t):
    bucket, prefix, token, bn, df, off, length = t
    url = f"https://storage.googleapis.com/{bucket}/{prefix}/{bn}"
    for attempt in range(4):
        try:
            rreq = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Range": f"bytes={off}-{off+length-1}"})
            with urllib.request.urlopen(rreq, timeout=45) as resp:
                data = resp.read()
            fd = os.open(df, os.O_WRONLY)
            try:
                os.pwrite(fd, data, off)
            finally:
                os.close(fd)
            return len(data)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.5 * (attempt + 1))
t0 = time.time()
with ThreadPoolExecutor(max_workers=64) as ex:
    for fut in as_completed([ex.submit(dl, t) for t in tasks]):
        fut.result()
print(f"[init] Vertex AI 64-stream GCS HTTPS staging completed in {time.time()-t0:.2f}s", flush=True)
'
    touch /tmp/dgemma/.ready
  ) &
  MODEL="/tmp/dgemma"
else
  # Default: Download public weights from Hugging Face
  MODEL_HF="${MODEL_HF:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}"
  LOCAL_WEIGHTS="/tmp/dgemma"
  if [ -d "$LOCAL_WEIGHTS" ] && [ -f "$LOCAL_WEIGHTS/config.json" ]; then
    echo "[init] Using pre-downloaded weights at $LOCAL_WEIGHTS..."
  else
    echo "[init] Downloading public model weights from Hugging Face: $MODEL_HF..."
    mkdir -p "$LOCAL_WEIGHTS"
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_HF', local_dir='$LOCAL_WEIGHTS')"
  fi
  MODEL="$LOCAL_WEIGHTS"
fi

mkdir -p /root/.cache/flashinfer /root/.triton /root/.cache/vllm

# If JIT cache archive exists in GCS, unpack it to eliminate runtime JIT compile overhead
JIT_CACHE_ARCHIVE="/mnt/gcs/jit-cache/jit-cache.tar.gz"
if [ -f "$JIT_CACHE_ARCHIVE" ]; then
  tar -xzf "$JIT_CACHE_ARCHIVE" -C /root/ || true
fi

# Record container boot and warmup stage telemetry in /tmp/dgemma/warmup_state.json
mkdir -p /tmp/dgemma
export BOOT_TS="$(date +%s.%N)"
DETECTED_PROJECT="${GCP_PROJECT:-}"
if [ -z "$DETECTED_PROJECT" ]; then
  DETECTED_PROJECT=$(python3 -c '
import urllib.request
try:
    req = urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/project/project-id", headers={"Metadata-Flavor": "Google"})
    print(urllib.request.urlopen(req, timeout=1).read().decode().strip())
except Exception:
    print("")
' 2>/dev/null || true)
fi
export GCS_BUCKET="${GCS_BUCKET:-dgem-weights-${DETECTED_PROJECT}}"
printf '{"phase":"mounting_gcs","boot_ts":%s,"bytes_staged_gb":0.0}\n' "$BOOT_TS" > /tmp/dgemma/warmup_state.json

# On large-memory instances (e.g. RTX Pro 6000 with 80GB RAM), stage weights into /tmp/dgemma via 64-stream Direct GCS HTTPS Range API overlapped with vLLM initialization
# Skip when weights are baked into the image (dgem-weights): staging would copy 17.53 GiB from Cloud Storage
# anyway and switch MODEL to /tmp/dgemma, ignoring the baked copy.
if [ "${COPY_TO_SHM:-0}" = "1" ] && [ "$MODEL" != "/opt/dgemma/weights" ] && [ -d "/mnt/gcs/dgemma" ] && [ -f "/mnt/gcs/dgemma/config.json" ]; then
  echo "[init] Overlapping 64-stream GCS HTTPS Range staging (/tmp/dgemma) with vLLM + CUDA initialization..."
  STAGE_START_TS="$(date +%s.%N)"
  export STAGE_START_TS
  printf '{"phase":"staging_tmpfs","boot_ts":%s,"stage_start_ts":%s,"bytes_staged_gb":0.0}\n' "$BOOT_TS" "$STAGE_START_TS" > /tmp/dgemma/warmup_state.json
  find /mnt/gcs/dgemma -maxdepth 1 -type f ! -name "*.safetensors" | xargs -I {} cp -f {} /tmp/dgemma/

  # Pre-copy first 4 MB (containing the complete safetensors JSON header) and truncate to exact size synchronously (<0.2s)
  # so VllmConfig header validation at t=38s succeeds immediately while tensor bodies stream in parallel
  python3 -c '
import glob, os
HDR_BYTES = 4 * 1024 * 1024
for sf in sorted(glob.glob("/mnt/gcs/dgemma/*.safetensors")):
    df = os.path.join("/tmp/dgemma", os.path.basename(sf))
    sz = os.path.getsize(sf)
    with open(sf, "rb") as f_in, open(df, "wb") as f_out:
        f_out.write(f_in.read(min(HDR_BYTES, sz)))
        f_out.truncate(sz)
'

  # Install sitecustomize.py hook on DefaultModelLoader.load_weights so vLLM runs all Python/Torch/Config/CUDA/NCCL init
  # in parallel and only waits right before copying tensor weights into GPU VRAM (t ~ 65s)
  cat << 'EOF' > /tmp/sitecustomize.py
import os, time
try:
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    _orig_load_weights = DefaultModelLoader.load_weights
    def _waiting_load_weights(self, model, model_config):
        if str(getattr(model_config, "model", "")).startswith("/tmp/dgemma") and not os.path.exists("/tmp/dgemma/.ready"):
            print("[init] EngineCore reached DefaultModelLoader.load_weights; waiting for /tmp/dgemma/.ready...", flush=True)
            while not os.path.exists("/tmp/dgemma/.ready"):
                time.sleep(0.2)
            print("[init] /tmp/dgemma/.ready confirmed! Loading 17.53 GiB weights via zero-copy mmap...", flush=True)
        return _orig_load_weights(self, model, model_config)
    DefaultModelLoader.load_weights = _waiting_load_weights
except Exception:
    pass
EOF
  export PYTHONPATH="/tmp:${PYTHONPATH:-}"

  (
    set -e
    python3 -c '
import glob, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

req = urllib.request.Request(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    headers={"Metadata-Flavor": "Google"}
)
with urllib.request.urlopen(req, timeout=5) as r:
    token = json.loads(r.read().decode())["access_token"]

bucket = os.environ.get("GCS_BUCKET", "")
src_files = sorted(glob.glob("/mnt/gcs/dgemma/*.safetensors"))
hdr = 4 * 1024 * 1024
chunk = 64 * 1024 * 1024
tasks = []
pre_staged = 0
for sf in src_files:
    bn = os.path.basename(sf)
    df = os.path.join("/tmp/dgemma", bn)
    sz = os.path.getsize(sf)
    start_off = min(hdr, sz)
    pre_staged += start_off
    for off in range(start_off, sz, chunk):
        tasks.append((bucket, token, bn, df, off, min(chunk, sz - off)))

def download_range(t):
    bucket, token, bn, df, off, length = t
    end = off + length - 1
    url = f"https://storage.googleapis.com/{bucket}/dgemma/{bn}"
    for attempt in range(4):
        try:
            rreq = urllib.request.Request(url, headers={
                "Authorization": f"Bearer {token}",
                "Range": f"bytes={off}-{end}"
            })
            with urllib.request.urlopen(rreq, timeout=45) as resp:
                data = resp.read()
            fd_out = os.open(df, os.O_WRONLY)
            try:
                os.pwrite(fd_out, data, off)
            finally:
                os.close(fd_out)
            return len(data)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.5 * (attempt + 1))

t0 = float(os.environ.get("STAGE_START_TS", time.time()))
boot_ts = float(os.environ.get("BOOT_TS", t0))
staged = pre_staged
last_write = 0.0
with ThreadPoolExecutor(max_workers=64) as ex:
    futs = [ex.submit(download_range, t) for t in tasks]
    for fut in as_completed(futs):
        staged += fut.result()
        now = time.time()
        if now - last_write >= 0.5 or staged >= 18800000000:
            last_write = now
            gb = round(staged / (1024**3), 2)
            try:
                with open("/tmp/dgemma/warmup_state.json.tmp", "w") as wf:
                    json.dump({
                        "phase": "staging_tmpfs",
                        "boot_ts": boot_ts,
                        "stage_start_ts": t0,
                        "bytes_staged_gb": gb
                    }, wf)
                os.replace("/tmp/dgemma/warmup_state.json.tmp", "/tmp/dgemma/warmup_state.json")
            except Exception:
                pass
dt = time.time() - t0
print(f"[init] 64-stream GCS HTTPS range copy of {round(staged/(1024**3),2)} GiB to /tmp/dgemma completed in {dt:.2f}s", flush=True)
'
    STAGE_END_TS="$(date +%s.%N)"
    printf '{"phase":"loading_vllm_siglip","boot_ts":%s,"stage_start_ts":%s,"stage_end_ts":%s,"bytes_staged_gb":17.53}\n' "$BOOT_TS" "$STAGE_START_TS" "$STAGE_END_TS" > /tmp/dgemma/warmup_state.json
    touch /tmp/dgemma/.ready
    echo "[init] Safetensors copy to /tmp/dgemma complete (stage_end_ts=$STAGE_END_TS)"
  ) &
  MODEL="/tmp/dgemma"
else
  printf '{"phase":"loading_vllm_siglip","boot_ts":%s,"stage_start_ts":%s,"stage_end_ts":%s,"bytes_staged_gb":17.53}\n' "$BOOT_TS" "$BOOT_TS" "$BOOT_TS" > /tmp/dgemma/warmup_state.json
fi

SS_PORT="$PORT"
if [ "${ROLE:-default}" = "decision-index" ]; then
  SS_PORT=8081
fi

echo "[init] Starting structured_server proxy on port $SS_PORT (upstream: http://127.0.0.1:8000)..."
python3 /opt/dgemma/structured_server.py \
  --upstream http://127.0.0.1:8000 \
  --model dgemma \
  --tokenizer "$MODEL" \
  --canvas "$CANVAS" \
  --host 0.0.0.0 \
  --port "$SS_PORT" &

if [ "${ROLE:-default}" = "decision-index" ]; then
  echo "[init] Starting dgem systemone adapter on port $PORT -> http://127.0.0.1:8081/v1 (T*=${TEMPERATURE:-1.0})..."
  /usr/local/bin/dgem systemone serve \
    --port "$PORT" \
    --upstream "http://127.0.0.1:8081/v1" \
    --temperature "${TEMPERATURE:-1.0}" \
    --api-key "${API_KEY:-}" &
fi

VLLM_EXTRA_ARGS=()
if [ "$ENFORCE_EAGER" = "1" ]; then
  export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
  VLLM_EXTRA_ARGS+=(--enforce-eager --kernel-config '{"enable_flashinfer_autotune":false,"enable_cutedsl_warmup":false,"enable_jit_warmup":false}')
fi

if [ "$DISABLE_MM" = "1" ]; then
  VLLM_EXTRA_ARGS+=(--language-model-only --skip-mm-profiling --limit-mm-per-prompt '{"image":0,"video":0}')
fi

if [ -n "$EXTRA_ARGS" ]; then
  VLLM_EXTRA_ARGS+=($EXTRA_ARGS)
fi

if [ "${COPY_TO_SHM:-0}" = "1" ] && [ "$MODEL" = "/tmp/dgemma" ]; then
  # Note: vLLM starts immediately and overlaps its ~65s Python/Torch/CUDA/NCCL init with /tmp/dgemma staging via /tmp/sitecustomize.py!
  # Reclaim 17.53 GiB tmpfs RAM as soon as vLLM moves weights to GPU VRAM
  (
    while ! python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)' >/dev/null 2>&1; do
      sleep 2
    done
    rm -f /tmp/dgemma/*.safetensors
    # Merge (not overwrite): the self-warmup runs concurrently and records "warmed" in the same file.
    python3 -c 'import json; p="/tmp/dgemma/warmup_state.json"
try: st = json.load(open(p))
except Exception: st = {}
st.update({"phase": "ready", "bytes_staged_gb": 17.53, "vllm_ready": True}); json.dump(st, open(p + ".tmp", "w"))
import os; os.replace(p + ".tmp", p)' || true
    echo "[init] Reclaimed 17.53 GiB /tmp/dgemma RAM after vLLM GPU initialization."
  ) &
else
  VLLM_EXTRA_ARGS+=(--safetensors-load-strategy prefetch)
fi

# Self-warmup: once vLLM answers /health, send samples=1, samples=4 and (with SigLIP enabled) one image
# decision through the local structured server so the first user request doesn't pay kernel/JIT warmup.
if [ "${SELF_WARMUP:-1}" = "1" ]; then
  (
    while ! python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)' >/dev/null 2>&1; do
      sleep 2
    done
    WARM_T0="$(date +%s.%N)"
    PORT="$PORT" DISABLE_MM="${DISABLE_MM:-0}" python3 - <<'PYWARM' || true
import base64, json, os, struct, time, urllib.request, zlib
port = os.environ["PORT"]
def png(w=64, h=64):
    raw = b"".join(b"\x00" + bytes([200, 60, 60]) * w for _ in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
q = [{"id": "urgent", "type": "boolean", "instructions": "Is this urgent?"},
     {"id": "team", "type": "choice", "instructions": "Which team?", "options": [{"name": "billing", "description": "charges"}, {"name": "technical", "description": "outages"}, {"name": "account", "description": "profile"}]}]
jobs = [(1, "Production API is down for all customers."), (4, "I was charged twice for my plan.")]
for n, text in jobs:
    body = {"model": "dgemma", "messages": [{"role": "system", "content": json.dumps({"instructions": "Triage.", "samples": n, "think": 0, "questions": q})}, {"role": "user", "content": json.dumps({"ticket": text})}]}
    if n == 1 and os.environ.get("DISABLE_MM", "0") == "0":
        img = "data:image/png;base64," + base64.b64encode(png()).decode()
        warm_img = dict(body); warm_img["messages"] = [body["messages"][0], {"role": "user", "content": [{"type": "text", "text": json.dumps({"ticket": "Describe the image."})}, {"type": "image_url", "image_url": {"url": img}}]}]
        jobs_img = [warm_img]
    else:
        jobs_img = []
    for b in [body] + jobs_img:
        t = time.time()
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(b).encode(), {"content-type": "application/json"}), timeout=300).read()
            print(f"[warmup] ok samples={n} image={isinstance(b['messages'][1]['content'], list)} {time.time()-t:.1f}s", flush=True)
        except Exception as e:
            print(f"[warmup] failed samples={n}: {e!r}", flush=True)
PYWARM
    python3 - "$WARM_T0" <<'PYSTATE' || true
import json, sys, time
p = "/tmp/dgemma/warmup_state.json"
try: st = json.load(open(p))
except Exception: st = {}
st.update({"warmed": True, "warmup_s": round(time.time() - float(sys.argv[1]), 1)})
json.dump(st, open(p + ".tmp", "w"))
import os; os.replace(p + ".tmp", p)
PYSTATE
    echo "[init] Self-warmup complete."
  ) &
fi

echo "[init] Launching vLLM engine core on port 8000 (overlapped with /tmp/dgemma staging)..."
exec vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name dgemma \
  --trust-remote-code \
  --max-num-seqs "${MAX_SEQS:-32}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --attention-backend "${ATTN:-TRITON_ATTN}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.40}" \
  --kv-cache-memory "$(( ${KV_CACHE_GB:-2} * 1073741824 ))" \
  --max-logprobs 32 \
  --enable-prefix-caching \
  --diffusion-config "{\"canvas_length\": ${CANVAS}}" \
  --override-generation-config '{"max_new_tokens": null}' \
  --async-scheduling \
  "${VLLM_EXTRA_ARGS[@]}"
