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

"""Hermetic structural, reference, and security checks for deploy/terraform."""

import json
import pathlib
import re
import unittest

TERRAFORM_DIR = pathlib.Path(__file__).resolve().parent
DASHBOARD_PATH = (
    TERRAFORM_DIR.parent / "monitoring" / "dgem-operational-dashboard.json"
)


def _strip_hcl_comments_and_strings(hcl_text: str) -> tuple[str, int, int]:
  """Strip comments/string bodies while counting brace balance and quote parity.

  Args:
    hcl_text: Raw HCL file content.

  Returns:
    Tuple of `(stripped_code, net_brace_depth, unclosed_quotes)`.
  """
  output_chars: list[str] = []
  brace_depth = 0
  in_string = False
  index = 0
  length = len(hcl_text)

  while index < length:
    char = hcl_text[index]
    if in_string:
      if char == "\\" and index + 1 < length:
        index += 2
        continue
      if char == '"':
        in_string = False
        output_chars.append('"')
      index += 1
      continue

    if char == "#":
      while index < length and hcl_text[index] != "\n":
        index += 1
      continue
    if char == "/" and index + 1 < length and hcl_text[index + 1] == "/":
      while index < length and hcl_text[index] != "\n":
        index += 1
      continue
    if char == '"':
      in_string = True
      output_chars.append('"')
      index += 1
      continue
    if char == "{":
      brace_depth += 1
    elif char == "}":
      brace_depth -= 1
      if brace_depth < 0:
        return ("".join(output_chars), brace_depth, 0)
    output_chars.append(char)
    index += 1

  return ("".join(output_chars), brace_depth, 1 if in_string else 0)


class TerraformModuleValidationTest(unittest.TestCase):
  """Validate HCL structure, variable/local references, and dashboard alignment."""

  def setUp(self) -> None:
    self.tf_files = sorted(TERRAFORM_DIR.glob("*.tf"))
    self.assertGreaterEqual(len(self.tf_files), 8)
    self.raw_by_name = {
        path.name: path.read_text(encoding="utf-8") for path in self.tf_files
    }

  def test_balanced_braces_and_quotes_in_all_tf_files(self) -> None:
    for filename, content in self.raw_by_name.items():
      _, brace_depth, unclosed_quotes = _strip_hcl_comments_and_strings(content)
      self.assertEqual(
          brace_depth, 0, f"unbalanced braces in {filename} (depth={brace_depth})"
      )
      self.assertEqual(
          unclosed_quotes, 0, f"unclosed string literal in {filename}"
      )

  def test_all_var_local_and_resource_references_resolve(self) -> None:
    declared_vars: set[str] = set()
    declared_locals: set[str] = set()
    declared_resources: set[tuple[str, str]] = set()

    var_decl_pattern = re.compile(r'variable\s+"([a-zA-Z0-9_]+)"\s*\{')
    res_decl_pattern = re.compile(
        r'resource\s+"([a-zA-Z0-9_]+)"\s+"([a-zA-Z0-9_]+)"\s*\{'
    )
    locals_block_pattern = re.compile(r"locals\s*\{")
    local_assign_pattern = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*=", re.MULTILINE)

    for content in self.raw_by_name.values():
      declared_vars.update(var_decl_pattern.findall(content))
      declared_resources.update(res_decl_pattern.findall(content))
      for match in locals_block_pattern.finditer(content):
        start = match.end()
        depth = 1
        cursor = start
        while cursor < len(content) and depth > 0:
          if content[cursor] == "{":
            depth += 1
          elif content[cursor] == "}":
            depth -= 1
          cursor += 1
        block_body = content[start : cursor - 1]
        for line in block_body.splitlines():
          # Top-level assignments inside `locals { ... }` use 2-space indent.
          if re.match(r"^  [a-zA-Z0-9_]+\s*=", line):
            assignment = local_assign_pattern.match(line)
            if assignment:
              declared_locals.add(assignment.group(1))

    combined = "\n".join(self.raw_by_name.values())
    referenced_vars = set(re.findall(r"\bvar\.([a-zA-Z0-9_]+)\b", combined))
    referenced_locals = set(re.findall(r"\blocal\.([a-zA-Z0-9_]+)\b", combined))
    referenced_resources = set(
        re.findall(r"\b(google_[a-z0-9_]+)\.([a-zA-Z0-9_]+)\b", combined)
    )
    # Exclude data.google_project.current which starts with `data.google_`.
    referenced_resources.discard(("google_project", "current"))

    self.assertEqual(referenced_vars - declared_vars, set())
    self.assertEqual(referenced_locals - declared_locals, set())
    self.assertEqual(referenced_resources - declared_resources, set())

  def test_zero_trust_iam_and_storage_guardrails(self) -> None:
    combined = "\n".join(self.raw_by_name.values())
    self.assertIn('uniform_bucket_level_access = true', combined)
    self.assertIn('public_access_prevention    = "enforced"', combined)
    self.assertIn('var.allow_member != "allUsers"', combined)
    self.assertIn('var.allow_member != "allAuthenticatedUsers"', combined)
    self.assertIn("secret_key_ref", combined)

    tfvars_example = (TERRAFORM_DIR / "terraform.tfvars.example").read_text(
        encoding="utf-8"
    )
    self.assertIn('project_id   = "<PROJECT>"', tfvars_example)
    self.assertIn("v0.4.3", tfvars_example)
    self.assertNotIn("v0.4.2", tfvars_example)
    self.assertNotIn("gcloud", tfvars_example)

  def test_cloudrun_gpu_startup_probe_uses_ready_route(self) -> None:
    gpu_hcl = self.raw_by_name["cloudrun_gpu.tf"]
    self.assertIn('path = "/ready"', gpu_hcl)
    self.assertNotIn('path = "/health"', gpu_hcl)
    self.assertIn("initial_delay_seconds = 10", gpu_hcl)
    self.assertIn("period_seconds        = 5", gpu_hcl)
    self.assertIn("timeout_seconds       = 4", gpu_hcl)
    self.assertIn("failure_threshold     = 180", gpu_hcl)

  def test_monitoring_metrics_cover_every_dashboard_user_metric(self) -> None:
    self.assertTrue(DASHBOARD_PATH.is_file())
    dashboard_raw = DASHBOARD_PATH.read_text(encoding="utf-8")
    parsed_dashboard = json.loads(dashboard_raw)
    self.assertIn("mosaicLayout", parsed_dashboard)

    dashboard_user_metrics = set(
        re.findall(r"logging\.googleapis\.com/user/([a-zA-Z0-9_]+)", dashboard_raw)
    )
    self.assertGreaterEqual(len(dashboard_user_metrics), 4)

    monitoring_hcl = self.raw_by_name["monitoring.tf"]
    provisioned_metrics = set(
        re.findall(r'name\s*=\s*"(dgem_[a-zA-Z0-9_]+)"', monitoring_hcl)
    )
    self.assertEqual(
        dashboard_user_metrics - provisioned_metrics,
        set(),
        f"dashboard references unprovisioned metrics: {dashboard_user_metrics - provisioned_metrics}",
    )


if __name__ == "__main__":
  unittest.main()
