# Sandbox root (env and code mounted read-only); built on first use
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
# a real /sbin for docker --init when a system-Python venv mounts the host's /usr
RUN rm /sbin && mkdir /sbin
