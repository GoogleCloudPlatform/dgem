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

# Scheduled health check (synthetic probe) for dgem: scripts/probe.py on a small Python image.
# Built and deployed by scripts/deploy_probe.sh as Cloud Run jobs triggered by Cloud Scheduler.
FROM python:3.12-slim
WORKDIR /app
COPY scripts/probe.py scripts/contract_diff.py scripts/serving_speed.py /app/scripts/
COPY fixtures/bbox/ /app/fixtures/bbox/
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "/app/scripts/probe.py"]
