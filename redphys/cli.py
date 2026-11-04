"""redphys <command>: target, attack, defend, evaluate (scripts/), report, methods, serve (a PhyAgent)."""

import argparse
import sys

from .data import PROJECT_ROOT

SCRIPTS = {"target": "target", "attack": "attack", "defend": "defense", "evaluate": "evaluate"}


def main():
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd in SCRIPTS:
        path, sys.argv = PROJECT_ROOT / "scripts" / f"{SCRIPTS[cmd]}.py", [f"redphys {cmd}", *args]  # runpy would reset argv[0], the usage name
        exec(compile(path.read_text(), path, "exec"), {"__name__": "__main__", "__file__": str(path)})
    elif cmd == "report":
        p = argparse.ArgumentParser(prog="redphys report", description="Table of evaluate/score runs")
        p.add_argument("results", nargs="+", help="Run dirs or eval_results.json files")
        p.add_argument("--output", "-o", help="Also write it: .md or .csv")
        a = p.parse_args(args)
        from .evaluation import report

        print(report(a.results, a.output))
    elif cmd == "serve":
        p = argparse.ArgumentParser(prog="redphys serve", description="A PhyAgent behind an OpenAI-compatible chat API")
        p.add_argument("victim", help="phymodel (the PhyModel alone), tools, sketchpad or spatialclaw")
        p.add_argument("--port", type=int, default=9000)
        p.add_argument("--model", help="The PhyModel")
        p.add_argument("--server-url", help="The PhyModel's API base")
        a = p.parse_args(args)
        from .victims import serve

        print(f"{a.victim} on http://localhost:{a.port}")
        serve(a.victim, a.port, background=False, **{k: v for k, v in (("model", a.model), ("server_url", a.server_url)) if v})
    elif cmd == "methods":
        from . import methods

        for kind, names in methods().items():
            print(f"{kind}: {' '.join(names)}")
    else:
        sys.exit(f"usage: redphys {{{','.join([*SCRIPTS, 'report', 'methods', 'serve'])}}} ...  (redphys <command> -h)")
