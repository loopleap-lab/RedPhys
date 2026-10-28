# Scenario agent root (env and code mounted read-only); built on first use by redphys.target
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
