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

echo "==> Checking prerequisites for DiffusionGemma (diffgemma)..."

# Verify OS
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Error: diffgemma requires macOS on Apple Silicon."
  exit 1
fi

# Verify architecture
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Error: Apple Silicon (arm64) is required for Metal acceleration."
  exit 1
fi

# Verify Rust / Cargo
if ! command -v cargo >/dev/null 2>&1; then
  echo "Error: cargo not found. Please install Rust via https://rustup.rs"
  exit 1
fi

echo "==> Building and installing diffgemma from upstream git repository..."
cargo install --git https://github.com/mmastrac/diffgemma diffgemma

echo "==> diffgemma installed successfully: $(command -v diffgemma)"
