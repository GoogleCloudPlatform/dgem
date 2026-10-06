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

"""Copy a docs/ page into docs-site/, keeping the site's front matter.

Transformations: the leading H1 is dropped (Starlight renders the front-matter
title) and relative links like (foo.md#anchor) become (/dgem/foo/#anchor).

Usage: scripts/docs_sync_page.py setup.md [user-guide.md ...]
"""

import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS = os.path.join(REPO, "docs")
SITE = os.path.join(REPO, "docs-site", "src", "content", "docs")

LINK = re.compile(r"\]\((?!https?://|/|#|mailto:)([^)\s]+?)\.md(#[^)]*)?\)")


def front_matter(text):
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[: end + 4].rstrip() + "\n", text[end + 4 :]
    return "", text


GITHUB = "https://github.com/GoogleCloudPlatform/dgem/blob/main/"


def site_link_for(rel):
    """Link rewriter for the page docs/<rel>: resolves relative targets against the page's directory."""
    page_dir = os.path.dirname(rel)

    def repl(m):
        target, anchor = m.group(1), m.group(2) or ""
        resolved = os.path.normpath(os.path.join(page_dir, target))
        if resolved.startswith(".."):
            # outside docs/ (e.g. ../benchmarks/...): link to the file on GitHub
            return f"]({GITHUB}{os.path.normpath(resolved[3:])}.md{anchor})"
        if resolved.endswith("README"):
            resolved = resolved[: -len("README")].rstrip("/")
        if resolved == "benchmarks-report":
            resolved = "benchmarks"  # renamed on the site (see docs_sync_check.RENAMES)
        if resolved in ("index", "."):
            resolved = ""
        if resolved.endswith("/index"):
            resolved = resolved[: -len("/index")]
        return f"](/dgem/{resolved}/{anchor})".replace("//", "/")

    return repl


# Relative links to any other file outside docs/ (scripts, JSON receipts, directories) -> GitHub.
OUTSIDE = re.compile(r"\]\((\.\./[^)\s#]*)(#[^)]*)?\)")


def outside_link_for(rel):
    page_dir = os.path.dirname(rel)

    def repl(m):
        resolved = os.path.normpath(os.path.join("docs", page_dir, m.group(1)))
        if resolved.startswith("docs-site/public/"):
            return f"](/dgem/{resolved[len('docs-site/public/'):]}{m.group(2) or ''})"
        if resolved.startswith("docs/") or resolved == "docs":
            return m.group(0)
        kind = "blob" if os.path.splitext(resolved)[1] else "tree"
        return f"](https://github.com/GoogleCloudPlatform/dgem/{kind}/main/{resolved}{m.group(2) or ''})"

    return repl


def sync(rel):
    src = open(os.path.join(DOCS, rel), encoding="utf-8").read()
    dst_path = os.path.join(SITE, rel)
    src_fm, body = front_matter(src)
    if os.path.exists(dst_path):
        fm, _ = front_matter(open(dst_path, encoding="utf-8").read())
    else:
        # new page: reuse the docs/ front matter (title/description), or derive it from the H1
        h1 = next((l[2:].strip() for l in body.splitlines() if l.startswith("# ")), rel)
        fm = src_fm or f'---\ntitle: "{h1}"\n---\n'
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    lines = body.lstrip("\n").splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    body = "\n".join(lines).lstrip("\n")
    body = LINK.sub(site_link_for(rel), body)
    body = OUTSIDE.sub(outside_link_for(rel), body)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(fm + "\n" + body.rstrip() + "\n")
    print(f"synced docs/{rel} -> docs-site/.../{rel}")


SITE_LINK = re.compile(r"\]\(/dgem/([^)#]*?)/?(#[^)]*)?\)")


def sync_reverse(rel, site_rel=None):
    """docs-site -> docs: keep docs/ H1 title, convert /dgem/x/ links to x.md (relative)."""
    site = open(os.path.join(SITE, site_rel or rel), encoding="utf-8").read()
    _, body = front_matter(site)
    dst_path = os.path.join(DOCS, rel)
    old = open(dst_path, encoding="utf-8").read()
    fm, old_body = front_matter(old)
    title = next((l for l in old_body.splitlines() if l.startswith("# ")), "")
    depth = rel.count("/")
    prefix = "../" * depth

    def repl(m):
        target, anchor = m.group(1), m.group(2) or ""
        if target.endswith(".html") or target.startswith("assets"):
            return f"](https://googlecloudplatform.github.io/dgem/{target}{anchor})"
        if target in ("", "experiments"):
            target = (target + "/README").lstrip("/") if target else "index"
        return f"]({prefix}{target}.md{anchor})"

    body = SITE_LINK.sub(repl, body.lstrip("\n"))
    out = (fm + "\n" if fm else "") + (title + "\n\n" if title else "") + body.rstrip() + "\n"
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"synced docs-site/.../{site_rel or rel} -> docs/{rel}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--reverse":
        for rel in args[1:]:
            src, _, site_rel = rel.partition("=")
            sync_reverse(src, site_rel or None)
    else:
        for rel in args:
            sync(rel)
