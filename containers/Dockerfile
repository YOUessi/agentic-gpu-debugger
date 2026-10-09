ARG CUDA_BASE
FROM ${CUDA_BASE}

# Only image construction has network access. Candidate operations never do.
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*
COPY runner.py /opt/gpu-agent/runner.py
RUN chmod 0555 /opt/gpu-agent/runner.py
ENV PATH=/usr/local/cuda/bin:/usr/bin:/bin LC_ALL=C PYTHONDONTWRITEBYTECODE=1 CUDA_CACHE_DISABLE=1
USER 65532:65532
WORKDIR /tmp
ENTRYPOINT ["/usr/bin/python3", "-I", "/opt/gpu-agent/runner.py"]
