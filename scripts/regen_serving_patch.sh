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

# regen_serving_patch.sh
# Regenerates deploy/cloudrun/server/dgem.patch after you edit structured_server.py. The patch is written in GNU
# `diff -u` form (repository paths in the two header lines, no timestamps, no function names in hunk headers) and is
# checked to re-apply to structured_server.upstream.py exactly, as CI does. Use this instead of `diff -u` by hand:
# macOS diff splits hunks differently. Run from anywhere in the repository; it needs git and patch.

cd "$(dirname "$0")/.."
D="deploy/cloudrun/server"
T="$(mktemp -d "${TMPDIR:-/tmp}/dgem-regen.XXXXXX")"
trap 'rm -rf "$T"' EXIT

# git diff --no-index exits 1 when the files differ. The options and -c settings give GNU diff -u output whatever the
# local git configuration says.
rc=0
git -c diff.suppressBlankEmpty=false -c diff.noprefix=false -c diff.mnemonicPrefix=false \
  diff --no-index --no-color --no-ext-diff --no-textconv --diff-algorithm=myers --no-indent-heuristic \
  --inter-hunk-context=0 -U3 "$D/structured_server.upstream.py" "$D/structured_server.py" > "$T/raw" || rc=$?
if [ "$rc" -ne 1 ]; then
  echo "regen_serving_patch: git diff exited $rc (0 means structured_server.py is identical to the upstream file)" >&2
  exit 1
fi

sed -n '/^--- /,$p' "$T/raw" \
  | sed "1s|^--- .*|--- $D/structured_server.upstream.py|; 2s|^+++ .*|+++ $D/structured_server.py|" \
  | sed -E 's/^(@@ -[0-9,]+ \+[0-9,]+ @@).*/\1/' > "$T/dgem.patch"

cp "$D/structured_server.upstream.py" "$T/ss.py"
if ! patch -s "$T/ss.py" "$T/dgem.patch" || ! cmp -s "$T/ss.py" "$D/structured_server.py"; then
  echo "regen_serving_patch: the generated patch does not re-apply to structured_server.upstream.py; $D/dgem.patch is unchanged" >&2
  exit 1
fi

if cmp -s "$T/dgem.patch" "$D/dgem.patch"; then
  echo "$D/dgem.patch is up to date"
else
  cp "$T/dgem.patch" "$D/dgem.patch"
  echo "$D/dgem.patch regenerated; it re-applies to structured_server.upstream.py"
fi
