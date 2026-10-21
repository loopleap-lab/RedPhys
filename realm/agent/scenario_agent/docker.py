"""The agent runs in Docker: host env and code read-only, only the output dir writable, only API keys passed."""

import json
import os
import subprocess
import sys
from pathlib import Path

from ...data import DATASETS, PROJECT_ROOT
from .run import case_image, scenario_config

DOCKER = Path(__file__).resolve().parents[1] / "docker"
IMAGE = "realm-agent"


def run_in_docker(args) -> Path:
    """run(args) inside the container."""
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = scenario_config(args)
    local = cfg.generator == "local" or any(h in u for u in (cfg.server_url, cfg.checker_server_url) for h in ("localhost", "127.0.0.1", "0.0.0.0"))
    keys = {cfg.api_key_env, cfg.checker_api_key_env} - {""}
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode:
        subprocess.run(["docker", "build", "-t", IMAGE, "-f", str(DOCKER / "agent.Dockerfile"), str(DOCKER)], check=True)
    files = [Path(p).resolve() for p in (args.cases, args.labels_file, args.config) if p]
    ro = {p.parent for p in files} | ({Path(args.source).resolve()} if args.source else set())
    if args.cases:
        ro |= {Path(case_image(c["image"])).resolve().parent for c in json.loads(Path(args.cases).read_text())}
    if args.dataset:  # resolved inside the container
        ro |= {Path(DATASETS[args.dataset]["source"]), Path(DATASETS[args.dataset]["labels"]).parent}
    mounts = [Path(sys.prefix), Path(sys.base_prefix), PROJECT_ROOT / "realm", PROJECT_ROOT / "scripts", PROJECT_ROOT / "configs", *ro]
    argv = [str(PROJECT_ROOT / "scripts" / "generate_targets.py")]
    for k, v in vars(args).items():
        if k == "params":
            argv += ["--params", json.dumps(v)] if v else []
        elif v not in (None, False):
            argv += [f"--{k}"] + ([] if v is True else [str(Path(v).resolve() if k in ("cases", "labels_file", "config", "source", "output") else v)])
    cmd = ["docker", "run", "--rm", "--init", "--pull=never", "--user", f"{os.getuid()}:{os.getgid()}", "--read-only", "--tmpfs", "/tmp:size=2g",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "512", "--memory", "8g",
           *(["--network", "host"] if local else []), "-e", "HOME=/tmp", "-e", "REALM_AGENT_LOCAL=1", "-e", f"PYTHONPATH={PROJECT_ROOT}", *(a for k in keys for a in ("-e", k)),
           *(f"--volume={p}:{p}:ro" for p in dict.fromkeys(mounts) if p.exists()), f"--volume={out}:{out}", "-w", str(Path.cwd()),
           "--entrypoint", sys.executable, IMAGE, *argv]
    subprocess.run(cmd, check=True)
    return out
