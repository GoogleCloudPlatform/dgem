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

# refresh_structured_server.sh
# Fetches upstream structured_server.py from vllm-project/vllm:main at a pinned commit,
# records provenance in UPSTREAM, and applies dgem.patch.

COMMIT="${1:-a9eafde59cbd55182dc2265cc398b7186f0a0eaa}"
TARGET_DIR="deploy/cloudrun/server"
UPSTREAM_URL="https://raw.githubusercontent.com/vllm-project/vllm/${COMMIT}/examples/features/structured_diffusion/structured_server.py"

echo "==> Fetching upstream structured_server.py @ ${COMMIT:0:10}..."
curl -fsSL "$UPSTREAM_URL" -o "${TARGET_DIR}/structured_server.upstream.py"

cat <<EOF > "${TARGET_DIR}/UPSTREAM"
repository: https://github.com/vllm-project/vllm
commit: ${COMMIT}
file: examples/features/structured_diffusion/structured_server.py
pulled_at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

echo "==> Applying dgem.patch to produce ${TARGET_DIR}/structured_server.py..."
cp "${TARGET_DIR}/structured_server.upstream.py" "${TARGET_DIR}/structured_server.py"
patch "${TARGET_DIR}/structured_server.py" "${TARGET_DIR}/dgem.patch"

python3 -m py_compile "${TARGET_DIR}/structured_server.py"
echo "==> Successfully refreshed and verified ${TARGET_DIR}/structured_server.py!"
