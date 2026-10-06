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

"""Compare docs/ against docs-site/src/content/docs/ by content.

Normalization before comparing (so expected site-only differences are ignored):
  * YAML front matter is stripped from both sides.
  * A leading H1 title line is dropped (Starlight renders the front-matter title).
  * Markdown link targets are dropped: [text](target) -> [text]
  * Starlight :::caution[...] / ::: asides are mapped to GitHub > [!CAUTION] form.
  * Blank lines and trailing whitespace are ignored.

Usage:
  scripts/docs_sync_check.py            # report; exit 1 if any page differs or is missing
  scripts/docs_sync_check.py --diff F   # show a unified diff for one page (path relative to docs/)
  scripts/docs_sync_check.py --warn     # report but always exit 0
"""

import argparse
import difflib
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS = os.path.join(REPO, "docs")
SITE = os.path.join(REPO, "docs-site", "src", "content", "docs")

# docs/ path -> docs-site path (relative), for files renamed on the site.
RENAMES = {
    "benchmarks-report.md": "benchmarks.md",
    "experiments/README.md": "experiments/index.md",
    "index.md": None,  # docs-site uses a bespoke index.mdx landing page
    "confidence-beyond-shannon.md": None,  # GitHub stub; the site redirects to /confidence/overview/
    "history/idc-confidence-beyond-shannon.md": None,  # docs-only historical page (not published)
    "history/confidence-journey.md": None,  # docs-only historical narrative (not published)
}

LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def site_path(rel):
    if rel in RENAMES:
        return RENAMES[rel] and os.path.join(SITE, RENAMES[rel])
    base = os.path.join(SITE, rel)
    for cand in (base, base[:-3] + ".mdx"):
        if os.path.isfile(cand):
            return cand
    return base


def normalize(text):
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            text = text[end + 4:]
    out, first = [], True
    for line in text.splitlines():
        s = line.rstrip()
        if not s:
            continue
        if s.startswith("import ") and s.endswith(";"):
            continue  # MDX component imports (site-only)
        if re.fullmatch(r"<[A-Z]\w*\s*/>", s):
            continue  # MDX components (site-only), e.g. <JevParityDashboard />
        if first:
            first = False
            if s.startswith("# "):
                continue  # the page's H1 title (site renders it from front matter)
        s = s.replace("<br />", "<br>")
        s = LINK.sub(r"[\1]", s)
        m = re.match(r"^:::(\w+)(?:\[(.*)\])?$", s)
        if m:
            s = f"> [!{m.group(1).upper()}]" + (f" **{m.group(2)}**" if m.group(2) else "")
        elif s == ":::":
            continue
        out.append(s.lstrip("> ").strip())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diff", help="Show diff for one docs/ path")
    p.add_argument("--warn", action="store_true", help="Never fail")
    args = p.parse_args()

    pages = []
    for root, _, files in os.walk(DOCS):
        for f in files:
            if f.endswith(".md"):
                pages.append(os.path.relpath(os.path.join(root, f), DOCS))
    pages.sort()

    if args.diff:
        a = normalize(open(os.path.join(DOCS, args.diff), encoding="utf-8").read())
        b = normalize(open(site_path(args.diff), encoding="utf-8").read())
        sys.stdout.writelines(l + "\n" for l in difflib.unified_diff(a, b, "docs/" + args.diff, "docs-site/" + args.diff, lineterm=""))
        return 0

    missing, differs = [], []
    for rel in pages:
        sp = site_path(rel)
        if sp is None:
            continue
        if not os.path.isfile(sp):
            missing.append(rel)
            continue
        a = normalize(open(os.path.join(DOCS, rel), encoding="utf-8").read())
        b = normalize(open(sp, encoding="utf-8").read())
        if a != b:
            changed = sum(1 for l in difflib.ndiff(a, b) if l[:1] in "+-")
            differs.append((rel, changed))

    for rel in missing:
        print(f"MISSING  {rel}")
    for rel, n in differs:
        print(f"DIFFERS  {rel}  ({n} lines; see: scripts/docs_sync_check.py --diff {rel})")
    if missing or differs:
        print(f"{len(missing)} missing, {len(differs)} differing of {len(pages)} pages.")
        return 0 if args.warn else 1
    print(f"✅ PASSED: all {len(pages)} pages match (after normalization).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
