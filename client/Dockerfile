FROM python:3.10-slim

# System deps for CARLA client + OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install deps declared in pyproject.toml at the repo root.
# We only COPY pyproject.toml at build time — the rest of the repo is
# bind-mounted at runtime (see docker-compose.yml) and made importable
# via PYTHONPATH=/app. We do NOT `pip install .` the package itself.
RUN pip install --no-cache-dir tomli
COPY pyproject.toml /tmp/pyproject.toml
RUN pip install --no-cache-dir $(python -c "\
import tomli; \
c = tomli.load(open('/tmp/pyproject.toml', 'rb')); \
deps = c['project']['dependencies'] + c['project']['optional-dependencies']['dev']; \
print(' '.join(deps))\
")

# Non-root home for bind-mounted caches (uid/gid set by docker-compose `user:`).
RUN mkdir -p /home/carla && chmod 777 /home/carla

WORKDIR /app
