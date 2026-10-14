# Minimal root for the scenario agent (realm.target); the Python env and the code are mounted read-only.
# Built on first use: docker build -t realm-agent -f realm/agent/docker/agent.Dockerfile realm/agent/docker
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
