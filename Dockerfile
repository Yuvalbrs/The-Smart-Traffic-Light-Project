# Smart-Traffic-RL — the whole backend in one image.
#
# Two stages, because the dashboard is a Node build artefact and the hub is a Python service, and
# shipping Node into the runtime image just to serve static files would double its size for nothing.
#
# What is NOT here, deliberately:
#
#   * The Unity client. It is a Windows GUI binary; these containers are Linux and headless. It
#     runs natively and connects to the published port, exactly as the Android client does in
#     Exercise-Advanced-Programming's compose.
#   * A database server. This project embeds SQLite — a file opened in-process, not a service.
#     `data/` is a volume. Making it a real server means porting the WAL/PRAGMA layer in
#     src/db/engine.py to Postgres, which is a project, not a Dockerfile.
#
# Base images are pinned BY DIGEST, not by tag. `python:3.11-slim` is a moving target that will be
# a different image next month. This repository pre-registers its hypotheses and records git_sha
# and sumo_version on every single run — an environment that silently changes underneath it would
# break that contract in the one place nobody thinks to look.

# ---------------------------------------------------------------------------
# Stage 1 — the React dashboard
# ---------------------------------------------------------------------------
FROM node:20-slim@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0 AS frontend

WORKDIR /build

# package.json + lockfile first, so editing a .tsx does not re-run the install layer.
COPY frontend/package.json frontend/package-lock.json ./

# `npm ci`, with NO `|| npm install` fallback. The whole value of `ci` is that it fails loudly when
# the lockfile and package.json disagree; falling back to `install` would paper over that and
# silently build the dashboard from a different dependency tree than the one committed.
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ---------------------------------------------------------------------------
# Stage 2 — the hub: FastAPI + SUMO + the trained models
# ---------------------------------------------------------------------------
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534 AS hub

# The SUMO wheels do NOT declare their system library dependencies: pip installs happily and then
# both the `sumo` binary and `import libsumo` die at load time with
# "libXrender.so.1: cannot open shared object file". Verified with ldd inside this image —
# libXrender and libatomic are the two that python:3.11-slim genuinely lacks. Dropping either
# breaks the simulator at RUN time, not at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 \
        libatomic1 \
        libxml2 \
        libgl1 \
        libglu1-mesa \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The point of the stack: libsumo in-process rather than the socket client — roughly 10x
    # faster, and the difference between a live demo that keeps up and one that crawls.
    LIBSUMO_AS_TRACI=1

WORKDIR /app

# The commit this image was built from. Without it every run recorded inside the container stores
# git_sha = NULL, which drops the provenance stamp AND makes /comparison return nothing for that
# mode, because NULL never matches in SQL. Pass it at build time:
#     GIT_SHA=$(git rev-parse HEAD) docker compose up --build
ARG GIT_SHA=""
ENV SMART_TRAFFIC_GIT_SHA=${GIT_SHA}

# One exact, fully pinned dependency set (see the header of the lock file for why). `--no-compile`
# skips writing ~12k .pyc files that PYTHONDONTWRITEBYTECODE means we never benefit from anyway.
COPY requirements-docker.lock ./
RUN pip install --no-cache-dir --no-compile \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-docker.lock

# Fail the BUILD, not the first episode. The libXrender bug shipped a green build that died on
# startup; importing the native extensions here means that class of fault can never leave the
# builder again. `sumo --version` is checked too because sumo_env.py resolves the BINARY through
# sumolib.checkBinary(), which is a separate failure mode from the Python bindings loading.
RUN python -c "import libsumo, traci, sumolib, torch, fastapi, sqlalchemy, numpy; print('imports OK, torch', torch.__version__)" \
    && sumo --version | head -1

# Source last: everything above is cached across ordinary code edits.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY tests/ ./tests/
COPY pyproject.toml ./
# The reproducibility contract: tests/test_repro.py hashes a reference run against it. Without it
# two tests fail inside the container with FileNotFoundError - which would be a poor advertisement
# for a project whose whole argument is that its results reproduce.
COPY golden_hashes.json ./
COPY docker/entrypoint.sh ./docker/entrypoint.sh

COPY --from=frontend /build/dist ./frontend/dist

# Run as a normal user. `app` owns /app so the entrypoint can still write the database, the route
# files and the JSONL traces into the bind mounts.
RUN chmod +x ./docker/entrypoint.sh \
    && useradd --create-home --uid 10001 app \
    && chown -R app:app /app
USER app

EXPOSE 8000

# Reports unhealthy until the hub genuinely answers, so `depends_on: condition: service_healthy`
# means something to anything added later.
HEALTHCHECK --interval=10s --timeout=4s --start-period=40s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
