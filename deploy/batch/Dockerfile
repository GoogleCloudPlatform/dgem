# dgem batch image: the dgem serving image plus the Apache Beam SDK worker (Dataflow) and the batch runner
# (Cloud Run Jobs, Vertex AI batch prediction). vLLM, structured_server.py and entrypoint.sh are unchanged.
# Build: gcloud builds submit deploy/batch --config deploy/batch/cloudbuild.yaml \
#   --substitutions=_BASE=<REGION>-docker.pkg.dev/<PROJECT>/dgem/dgemma:<TAG>,_IMAGE=<REGION>-docker.pkg.dev/<PROJECT>/dgem/dgem-batch:<TAG>
ARG BASE
ARG BEAM_VERSION=2.69.0
FROM apache/beam_python3.12_sdk:${BEAM_VERSION} AS beam

FROM ${BASE}
ARG BEAM_VERSION=2.69.0
# vLLM keeps the system interpreter; Beam lives in its own venv so its pins can't touch vLLM's packages.
ENV DGEM_SYSTEM_PATH=${PATH}
RUN python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
 && pip install --no-cache-dir uv \
 && uv venv --seed -p "$(command -v python3)" /opt/beam-venv \
 && VIRTUAL_ENV=/opt/beam-venv uv pip install --no-cache "apache-beam[gcp]==${BEAM_VERSION}"
COPY --from=beam /opt/apache/beam /opt/apache/beam
COPY batch_worker.py dgem_beam.py /opt/dgem-batch/
RUN echo /opt/dgem-batch > /opt/beam-venv/lib/python3.12/site-packages/dgem_batch.pth \
 && /opt/beam-venv/bin/python -c 'import apache_beam, dgem_beam; print(apache_beam.__version__)' && /opt/beam-venv/bin/python -m pip show apache-beam | head -2
ENV PATH=/opt/beam-venv/bin:${PATH} \
    RUN_PYTHON_SDK_IN_DEFAULT_ENVIRONMENT=1 \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH}
# Dataflow needs the Beam boot entrypoint; Cloud Run Jobs and Vertex override the command with batch_worker.py.
ENTRYPOINT ["/opt/apache/beam/boot"]
