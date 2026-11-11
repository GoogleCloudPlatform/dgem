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

"""Unit tests for scripts/docs_link_check.py."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import docs_link_check  # noqa: E402


class DocsLinkCheckTest(unittest.TestCase):
  """Verify relative file and GFM heading-anchor checking."""

  def test_heading_to_gfm_slug_preserves_inline_code_placeholders(self) -> None:
    slug = docs_link_check.heading_to_gfm_slug(
        "Step 3: Run Your Dataset in the Web Studio (`https://<your-dgem-gateway>`)"
    )
    self.assertEqual(
        slug,
        "step-3-run-your-dataset-in-the-web-studio-httpsyour-dgem-gateway",
    )

  def test_check_markdown_files_catches_missing_files_and_anchors(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      glossary_path = os.path.join(temp_dir, "glossary.md")
      guide_path = os.path.join(temp_dir, "guide.md")

      with open(glossary_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# Glossary\n\n"
            "### Fixed-Depth Circuits (TC0) vs. Test-Time Compute\n\n"
            "### Repeated Section\n\n"
            "### Repeated Section\n\n"
            '<a id="explicit-anchor"></a>\n'
        )

      with open(guide_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# Guide\n\n"
            "- Valid cross-file: [tc0](glossary.md#fixed-depth-circuits-tc0-vs-test-time-compute)\n"
            "- Valid duplicate: [dup](glossary.md#repeated-section-1)\n"
            "- Valid html id: [id](glossary.md#explicit-anchor)\n"
            "- Ignored inline code: `[bad](missing.md#nope)`\n"
            "```markdown\n"
            "[fenced](also_missing.md#nope)\n"
            "```\n"
            "- Broken same-file anchor: [fan-out](#fan-out-to-an-endpoint)\n"
            "- Broken cross-file anchor: [bad-frag](glossary.md#does-not-exist)\n"
            "- Broken file path: [missing](absent.md)\n"
        )

      diagnostics = docs_link_check.check_markdown_files(
          [glossary_path, guide_path],
          temp_dir,
      )
      self.assertEqual(
          diagnostics,
          [
              "guide.md:10: broken anchor -> #fan-out-to-an-endpoint",
              "guide.md:11: broken anchor -> glossary.md#does-not-exist",
              "guide.md:12: broken link -> absent.md",
          ],
      )

  def test_raw_latex_in_heading_flags_mismatched_plain_slug(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      glossary_path = os.path.join(temp_dir, "glossary.md")
      arch_path = os.path.join(temp_dir, "architecture.md")
      with open(glossary_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# Glossary\n\n"
            "### Fixed-Depth Circuits ($\\mathsf{TC}^0$) vs. Test-Time Compute\n"
        )
      with open(arch_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# Architecture\n\n"
            "[tc0](glossary.md#fixed-depth-circuits-tc0-vs-test-time-compute)\n"
        )
      diagnostics = docs_link_check.check_markdown_files(
          [glossary_path, arch_path],
          temp_dir,
      )
      self.assertEqual(
          diagnostics,
          [
              "architecture.md:3: broken anchor ->"
              " glossary.md#fixed-depth-circuits-tc0-vs-test-time-compute",
          ],
      )


if __name__ == "__main__":
  unittest.main()
