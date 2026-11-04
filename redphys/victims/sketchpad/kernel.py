"""The sandboxed kernel: JSON lines in, {exit_code, output, images} out, <<IMG:n>> where image n was displayed."""

import ast
import base64
import contextlib
import io
import json
import sys
import traceback
import types

from PIL import Image

shown = []


def display(*objs):
    for o in objs:
        if isinstance(o, Image.Image):
            print(f"<<IMG:{len(shown)}>>")
            shown.append(o)
        elif o is not None:
            print(repr(o))


try:
    import IPython.display  # its display() replaced: `from IPython.display import display` in a cell gets this one
except ImportError:
    sys.modules["IPython"], sys.modules["IPython.display"] = types.ModuleType("IPython"), types.ModuleType("IPython.display")
sys.modules["IPython.display"].display = display
ns = {"__name__": "__main__"}
exec("from PIL import Image\nfrom IPython.display import display\nfrom tools import *", ns)

for line in sys.stdin:
    req, out = json.loads(line), io.StringIO()
    shown.clear()
    exit_code = 0
    with contextlib.redirect_stdout(out):
        try:
            if "images" in req:
                for i, b in enumerate(req["images"], 1):
                    ns[f"image_{i}"] = Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
            else:
                tree = ast.parse(req["code"])
                last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                exec(compile(tree, "<cell>", "exec"), ns)
                if last:
                    display(eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns))
                if "matplotlib.pyplot" in sys.modules:  # inline figures, as Jupyter shows them
                    plt = sys.modules["matplotlib.pyplot"]
                    for n in plt.get_fignums():
                        buf = io.BytesIO()
                        plt.figure(n).savefig(buf, format="png", bbox_inches="tight")
                        display(Image.open(buf))
                    plt.close("all")
        except BaseException as e:
            exit_code = 1
            print("".join(traceback.format_exception(type(e), e, e.__traceback__.tb_next)))  # from the cell's frame on
    imgs = []
    for im in shown:
        b = io.BytesIO()
        im.convert("RGB").save(b, "PNG")
        imgs.append(base64.b64encode(b.getvalue()).decode())
    sys.stdout.write(json.dumps({"exit_code": exit_code, "output": out.getvalue(), "images": imgs}) + "\n")
    sys.stdout.flush()
