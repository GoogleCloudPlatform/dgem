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

"""Verify relative file links and Markdown heading anchors in docs and README."""

from collections.abc import Sequence
import glob
import os
import re
import sys

REPOSITORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FENCE_PATTERN = re.compile(r"^\s*(```+|~~~+)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
INLINE_CODE_PATTERN = re.compile(r"`([^`]*)`")
MARKDOWN_LINK_LABEL_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_ID_PATTERN = re.compile(r"""\bid=["']([^"']+)["']""")
RELATIVE_LINK_PATTERN = re.compile(
    r"\]\((?!https?://|/|mailto:)([^)\s#]*)(#[^)\s]+)?\)"
)


def heading_to_gfm_slug(heading_text: str) -> str:
  """Convert a Markdown heading line body into a GitHub-flavored slug.

  Args:
    heading_text: Raw heading text after the leading `#` markers.

  Returns:
    Lowercase GitHub-flavored Markdown heading slug without a leading `#`.
  """
  inline_code_spans: list[str] = []

  def _stash_code(match: re.Match[str]) -> str:
    token = f"\x00CODE{len(inline_code_spans)}\x00"
    inline_code_spans.append(match.group(1))
    return token

  staged = INLINE_CODE_PATTERN.sub(_stash_code, heading_text)
  staged = MARKDOWN_LINK_LABEL_PATTERN.sub(r"\1", staged)
  staged = HTML_TAG_PATTERN.sub("", staged)
  staged = staged.replace("*", "").replace("~", "")
  for index, code_text in enumerate(inline_code_spans):
    staged = staged.replace(f"\x00CODE{index}\x00", code_text)

  allowed_chars: list[str] = []
  for char in staged.lower():
    if char.isalnum() or char in (" ", "-", "_"):
      allowed_chars.append(char)
  return "".join(allowed_chars).strip().replace(" ", "-")


def extract_heading_slugs(markdown_path: str) -> set[str]:
  """Extract all valid GFM heading slugs and HTML anchor IDs from a file.

  Args:
    markdown_path: Path to a Markdown file on disk.

  Returns:
    Set of valid fragment identifiers (without leading `#`).
  """
  slugs: set[str] = set()
  duplicate_counts: dict[str, int] = {}
  active_fence_marker = ""

  with open(markdown_path, encoding="utf-8") as handle:
    for raw_line in handle:
      line = raw_line.rstrip("\n")
      fence_match = FENCE_PATTERN.match(line)
      if fence_match:
        marker = fence_match.group(1)
        if not active_fence_marker:
          active_fence_marker = marker[0] * len(marker)
        elif line.lstrip().startswith(active_fence_marker):
          active_fence_marker = ""
        continue
      if active_fence_marker:
        continue

      line_without_code = INLINE_CODE_PATTERN.sub("", line)
      for html_id in HTML_ID_PATTERN.findall(line_without_code):
        slugs.add(html_id)

      heading_match = HEADING_PATTERN.match(line)
      if not heading_match:
        continue
      base_slug = heading_to_gfm_slug(heading_match.group(2))
      if not base_slug:
        continue
      seen_count = duplicate_counts.get(base_slug, 0)
      duplicate_counts[base_slug] = seen_count + 1
      slugs.add(base_slug if seen_count == 0 else f"{base_slug}-{seen_count}")

  return slugs


def check_markdown_files(
    markdown_paths: Sequence[str],
    repository_root: str,
) -> list[str]:
  """Validate relative file links and Markdown heading fragments across files.

  Args:
    markdown_paths: Sequence of Markdown file paths to inspect.
    repository_root: Root directory used for formatting relative diagnostics.

  Returns:
    List of human-readable diagnostic strings for broken paths or anchors.
  """
  diagnostics: list[str] = []
  heading_slug_cache: dict[str, set[str]] = {}

  for source_path in sorted(markdown_paths):
    active_fence_marker = ""
    source_dir = os.path.dirname(source_path)
    rel_source = os.path.relpath(source_path, repository_root)

    with open(source_path, encoding="utf-8") as handle:
      for line_number, raw_line in enumerate(handle, 1):
        line = raw_line.rstrip("\n")
        fence_match = FENCE_PATTERN.match(line)
        if fence_match:
          marker = fence_match.group(1)
          if not active_fence_marker:
            active_fence_marker = marker[0] * len(marker)
          elif line.lstrip().startswith(active_fence_marker):
            active_fence_marker = ""
          continue
        if active_fence_marker:
          continue

        scannable_line = INLINE_CODE_PATTERN.sub("", line)
        for link_match in RELATIVE_LINK_PATTERN.finditer(scannable_line):
          relative_target = link_match.group(1) or ""
          fragment = link_match.group(2) or ""
          if not relative_target and not fragment:
            continue

          target_path = (
              os.path.normpath(os.path.join(source_dir, relative_target))
              if relative_target
              else source_path
          )
          if not os.path.exists(target_path):
            diagnostics.append(
                f"{rel_source}:{line_number}: broken link -> {relative_target}"
            )
            continue

          if fragment and os.path.isfile(target_path) and target_path.endswith(".md"):
            anchor_slug = fragment[1:]
            if not anchor_slug:
              continue
            if target_path not in heading_slug_cache:
              heading_slug_cache[target_path] = extract_heading_slugs(target_path)
            if anchor_slug not in heading_slug_cache[target_path]:
              formatted_target = f"{relative_target}{fragment}"
              diagnostics.append(
                  f"{rel_source}:{line_number}: broken anchor -> {formatted_target}"
              )

  return diagnostics


def main() -> int:
  """Run the repository relative-link and heading-anchor check.

  Returns:
    Process exit code (`0` when all links resolve, `1` otherwise).
  """
  markdown_paths = glob.glob(
      os.path.join(REPOSITORY_ROOT, "docs", "**", "*.md"),
      recursive=True,
  ) + [
      os.path.join(REPOSITORY_ROOT, "README.md"),
      os.path.join(REPOSITORY_ROOT, "CONTRIBUTING.md"),
      os.path.join(REPOSITORY_ROOT, "examples", "README.md"),
  ]
  existing_paths = [path for path in markdown_paths if os.path.isfile(path)]
  diagnostics = check_markdown_files(existing_paths, REPOSITORY_ROOT)
  for diagnostic in diagnostics:
    print(diagnostic)
  status_label = "PASSED" if not diagnostics else "FAILED"
  print(
      f"{status_label}: {len(diagnostics)} broken relative links or anchors"
      f" in {len(existing_paths)} files"
  )
  return 1 if diagnostics else 0


if __name__ == "__main__":
  sys.exit(main())
