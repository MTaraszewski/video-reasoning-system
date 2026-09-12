# The finder: CPU only. Decoding, windowing, merging, schema, CLI, eval.
# It never needs a GPU — that is the model server's job — so this image stays
# small and runs anywhere, including a laptop pointed at a remote GPU box.

FROM python:3.12-slim

# ffmpeg libraries for PyAV, curl for endpoint checks, and a real font.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# fonts-dejavu-core is not incidental. The timestamp burned onto each frame is
# the mechanism the whole system depends on, and it needs a real scalable font.
# PIL's built-in bitmap font does not survive downscaling to the model's input
# resolution — and Cosmos3-Edge runs at 640x360.

# uv, pinned. It is the resolver and installer both here and locally, so the
# container and a developer's venv resolve to the same versions.
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Lockfile first: a source edit must not invalidate the dependency layer.
# --extra data brings in supervision and huggingface-hub, needed by the dataset
# fetch scripts. They live in an extra rather than the base deps because the
# service itself never needs them — only the tooling that assembles the eval set.
#
# The dev group (pytest and two small pure-Python packages) IS installed, and
# deliberately. The container is the supported way to run this — "nothing needs
# to be installed on the host" is a promise the repo makes — so an image that
# cannot run its own checks pushes the client back onto a host toolchain to
# answer "does this work here". That is a few hundred kilobytes against the one
# question a client asks first.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra data

# Then the project itself.
COPY src ./src
COPY scripts ./scripts
COPY config.yaml eventfinder.yaml ./
COPY tests ./tests
RUN uv sync --frozen --extra data

# PYTHONPATH puts the mounted source ahead of the copy uv installed into .venv.
# Without it the ./src bind mount is inert: the container keeps running whatever
# was baked at build time, so every edit silently needs a full rebuild and it is
# easy to test stale code while believing otherwise.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    VRS_DATA_DIR=/data \
    VRS_OUT_DIR=/out

# Mount points. Videos are mounted read-only at run time — the service must not
# be able to modify a client's footage.
RUN mkdir -p /data /out

ENTRYPOINT []
CMD ["video-reasoning", "--help"]
