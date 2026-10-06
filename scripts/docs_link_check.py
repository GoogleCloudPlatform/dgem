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

"""Check that relative links in docs/ and README.md point at files that exist (anchors not checked)."""
import glob, os, re, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LINK = re.compile(r"\]\((?!https?://|/|#|mailto:)([^)\s#]+)(?:#[^)\s]*)?\)")
bad = 0
files = glob.glob(os.path.join(REPO, "docs", "**", "*.md"), recursive=True) + [os.path.join(REPO, "README.md")]
for f in sorted(files):
    for i, line in enumerate(open(f, encoding="utf-8"), 1):
        for tgt in LINK.findall(line):
            if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(f), tgt))):
                print(f"{os.path.relpath(f, REPO)}:{i}: broken link -> {tgt}")
                bad += 1
print(f"{'✅ PASSED' if not bad else '❌ FAILED'}: {bad} broken relative links in {len(files)} files")
sys.exit(1 if bad else 0)
