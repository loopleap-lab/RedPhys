import hashlib
import os
import subprocess
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parent / "docker" / "agent.Dockerfile"
IMAGE = "redphys-agent:" + hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest()[:12]  # rebuilt when the Dockerfile changes


def sandbox_image(name: str = IMAGE) -> str:
    """The sandbox image (env and code are mounted read-only), built on first use."""
    if subprocess.run(["docker", "image", "inspect", name], capture_output=True).returncode:
        subprocess.run(["docker", "build", "-t", name, "-f", str(DOCKERFILE), str(DOCKERFILE.parent)], check=True)
    return name


def python_dirs(python) -> list[Path]:
    """What `python` needs mounted: its env, its base interpreter, each symlink to it."""
    python = Path(python)
    base = subprocess.run([python, "-c", "import sys; print(sys.base_prefix)"], capture_output=True, text=True, check=True).stdout.strip()
    dirs, p = [python.parents[1], Path(base)], python
    while p.is_symlink():
        p = p.parent / os.readlink(p)
        dirs.append(p.parents[1])
    return list(dict.fromkeys(dirs))
