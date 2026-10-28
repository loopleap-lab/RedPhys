#!/usr/bin/env bash
#   bash install.sh                                       # environment + weights
#   bash install.sh env                                   # environment only
#   bash install.sh weights [--check|--download-all|--delete]   # --check: verify only
cd "$(dirname "$0")"
BOOT="${PYTHON:-python3}"  # runs uv
PY=$BOOT                   # gets the packages
# system Python (not writable, no venv): install into ./.venv
[ -z "${PYTHON:-}" ] && ! $BOOT -c 'import os, site, sys; sys.exit(not (sys.prefix != sys.base_prefix or os.access(site.getsitepackages()[0], os.W_OK)))' 2>/dev/null && PY="$PWD/.venv/bin/python"
UV="$BOOT -m uv"
PIP="$UV pip install --python $PY"

install_env() {
  $BOOT -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "!! Python >= 3.10 required"; exit 1; }
  $UV --version >/dev/null 2>&1 || $BOOT -m pip install -q uv 2>/dev/null || $BOOT -m pip install -q --user uv 2>/dev/null \
    || $BOOT -m pip install -q --user --break-system-packages uv || { echo "!! pip is missing: apt install python3-pip"; exit 1; }
  [ "$PY" = "$BOOT" ] || [ -x "$PY" ] || $UV venv .venv --python "$BOOT"
  # highest CUDA build <= the driver's
  local cuda arch tag=cpu t
  cuda=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9.]+" | grep -oE "[0-9.]+$" || true)
  arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
  for t in 13.0:cu130 12.0:cu128 11.0:cu118; do  # minor-compatible
    [ -n "$cuda" ] && [ "$(printf '%s\n' "${t%:*}" "$cuda" | sort -V | head -1)" = "${t%:*}" ] && { tag=${t#*:}; break; }
  done
  echo ">> PyTorch build: $tag${arch:+ (compute capability $arch)}"
  $PIP --index-url "https://download.pytorch.org/whl/$tag" torch torchvision
  $PIP -e . "git+https://github.com/openai/CLIP.git" "git+https://github.com/facebookresearch/segment-anything.git"
  if [ "$tag" != cpu ]; then
    TORCH_CUDA_ARCH_LIST="$arch" $PIP --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git" \
      || echo "!! Detectron2 build failed (PhysPatch and SceneTAP only): https://detectron2.readthedocs.io"
    # own venv; > 0.19.1 needs CUDA 13
    local major=${cuda%%.*} vllm=vllm
    [ "$major" -lt 13 ] && vllm="vllm==0.19.1"
    if [ "$major" -lt 12 ]; then
      echo "!! CUDA $cuda is too old for vLLM: serve models elsewhere and pass --server_url"
    else
      $UV venv .venv-vllm --python "$PY" --allow-existing && $UV pip install --python .venv-vllm/bin/python "$vllm"
    fi
    # target generator server: own venv
    $UV venv .venv-target-generator --python 3.12 --allow-existing && $UV pip install --python .venv-target-generator/bin/python \
      --torch-backend=cu129 torch==2.13.0 torchvision==0.28.0 transformers==5.17.0 accelerate==1.15.0 safetensors==0.8.0 \
      huggingface-hub==1.33.0 fastapi==0.141.1 uvicorn==0.54.0 python-multipart==0.0.32 numpy==2.5.3 pillow==12.3.0 \
      "diffusers @ git+https://github.com/huggingface/diffusers@e0abab83b5df05de9e7abd788643c1a7c1e42e28"
  fi
  $PY -c "import torch, redphys.attacks, redphys.defense; print('>> torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '| RedPhys OK')"
  [ "$PY" = "$BOOT" ] || echo ">> installed in ./.venv: run  source .venv/bin/activate"
}

install_weights() {
  local MODE=default arg
  for arg in "$@"; do
    case "$arg" in --check) MODE=check ;; --download-all) MODE=download-all ;; --delete) MODE=delete ;;
      *) echo "unknown weights option: $arg"; return 1 ;; esac
  done
  local W=redphys CLIP=$HOME/.cache/clip COCO=dataset/coco missing=0 HUB  # CLIP: clip.load's default
  HUB=$($PY -c "from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)") || return 1  # HF_HOME as Python sees it
  local FP=$W/defense/freqpure/assets/pretrained/guided_diffusion/256x256_diffusion_uncond.pt
  local BLUE_SG=$W/defense/bluesuffix/assets/suffix_generator/adapter_model.safetensors
  local FONT=$W/attacks/figstep/assets/FreeMonoBold.ttf  # FigStep's font, from GNU FreeFont
  local SFONT=$W/attacks/scenetap/assets/LiberationSans-Bold.ttf  # SceneTAP's text box font: Arial Bold's metrics
  local FILES=(  # destination|sha256|url (gdrive:<id> via gdown)
    "$W/attacks/physpatch/assets/checkpoints/sam_vit_h_4b8939.pth|a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e|https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    "$W/defense/pad/assets/models/sam_vit_l_0b3195.pth|3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622|https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    "$W/attacks/advdiffvlm/assets/checkpoints/ldm_cin256-v2.ckpt|5b96bd47dd37d51a8ee15633dda0d4bc5dfb80c88e36d1e6e40b2c534336e6a1|https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt"
    "$W/attacks/anyattack/assets/checkpoints/coco_cos.pt|bed1f25d46b4eaef0c3d4eb39c3853b2241991cc048b3b2fff21b7b8c38e0070|https://huggingface.co/jiamingzz/anyattack/resolve/main/checkpoints/coco_cos.pt"
    "$FP|a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1|https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
    "$W/defense/oscp/assets/checkpoints/oscp_lora.safetensors|248457189e21a5dc0beba27f9f8c0142807ada18a010859608f2a38d7589ebde|gdrive:1JF7zLb1OqlxjKkIL2S5_b68L8B5UfYGW"
    "$W/attacks/coa/assets/conceptual_weights.pt|f09faf2b3b390c201ec3b80b223ea0baa2b303074d43dc3dec5663a9ecd34607|gdrive:14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT"
  )
  local QWEN GEN=.venv-target-generator/bin/python  # at server.py's REVISION
  QWEN=Qwen/Qwen-Image-2.1@$(sed -n 's/^REVISION = "\([0-9a-f]*\)".*/\1/p' redphys/agent/target_generator/server.py)
  local HF=(  # repo|patterns to skip: only the files the code loads
    "laion/CLIP-ViT-g-14-laion2B-s12B-b42K|pytorch_model.bin open_clip_pytorch_model.safetensors"  # open_clip reads the .bin
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k|*.bin pytorch_model-*"
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K|*.bin model.safetensors"
    "openai/clip-vit-large-patch14-336|*.msgpack *.h5"  # loads pytorch_model.bin (no safetensors on main)
    "openai/clip-vit-base-patch16|*.msgpack *.h5"
    "openai/clip-vit-base-patch32|*.msgpack *.h5"
    "OpenGVLab/InternVL3-1B-hf|"  # MPCAttack
    "facebook/dinov2-base|pytorch_model.bin"  # MPCAttack
    "chs20/fare4-clip|"  # ET3
    "stable-diffusion-v1-5/stable-diffusion-v1-5|*.bin *.ckpt v1-5-pruned* safety_checker/* *non_ema* unet/diffusion_pytorch_model.safetensors text_encoder/model.safetensors"  # OSCP: fp16 weights; SceneTAP: the fp32 VAE
    "lllyasviel/sd-controlnet-canny|*.bin images/* sd.png"  # OSCP
    "JingyeChen22/textdiffuser2-full-ft-inpainting@44ba2e12ef20b8959f942d2d42d0138e4491c46a|*.pkl *scaler.pt *scheduler.bin"  # SceneTAP
    "timm/vit_large_patch14_clip_224.openai|*.bin"
    "timm/vit_base_patch16_clip_224.openai|*.bin"
    "timm/vit_base_patch32_clip_224.openai|*.bin"
    "timm/ViT-SO400M-14-SigLIP|*.bin"
    "timm/ViT-B-16-SigLIP|*.bin *.safetensors"  # AdvEDM: open_clip SigLIP tokenizer only
    "gpt2|*.bin *.h5 *.msgpack *.ot *.tflite onnx/*"
    "Qwen/Qwen2.5-VL-3B-Instruct|"  # CoA / OAttack captions
  )
  local e path sum url repo skip

  if [[ $MODE == delete ]]; then  # ~/.cache/clip, torch hub: shared, kept
    local targets=("$BLUE_SG" "$FONT" "$SFONT" "$COCO/val2014")
    for e in "${FILES[@]}"; do targets+=("${e%%|*}"); done
    for e in "${HF[@]}"; do repo=${e%%|*}; targets+=("$HUB/models--${repo//\//--}"); done
    printf '  %s\n' "${targets[@]}"
    read -rp "Type YES to delete the above: " e
    [[ $e == YES ]] && rm -rf "${targets[@]}" && echo "Deleted."
    return
  fi

  # need LABEL TEST...: report; true = download now
  need() {
    if [[ $MODE != download-all ]] && "${@:2}" 2>/dev/null; then echo "  OK       $1"; return 1; fi
    echo "  MISSING/BAD  $1"; missing=$((missing + 1)); [[ $MODE != check ]]
  }
  sha_ok() { pyw check file "$1" "$2"; }
  fetch() {  # url dest sha256; keep only if the hash matches
    mkdir -p "$(dirname "$2")"
    if [[ $1 == gdrive:* ]]; then $PY -m gdown "${1#gdrive:}" -O "$2.part"; else wget -q --show-progress -O "$2.part" "$1"; fi \
      && sha_ok "$2.part" "$3" && mv "$2.part" "$2" || { echo "!! $2: download failed or sha256 mismatch"; rm -f "$2.part"; }
  }
  pyw() { ${PYW:-$PY} - "$@" <<'PY'
# check|get file|clip|resnet|hf ...
import fnmatch, hashlib, os, sys
from pathlib import Path
def sha(p):  # of the bytes on disk (page cache dropped first)
    fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24): h.update(b)
    return h.hexdigest()
def sha_ok(p, want):  # re-read once: a bad read can be transient
    return any(sha(p).startswith(want) for _ in range(2))
mode, what, *a = sys.argv[1:]
get, ok = mode == "get", True
if what == "file":  # a: path, sha256
    ok = Path(a[0]).is_file() and sha_ok(a[0], a[1])
elif what == "clip":  # OpenAI CLIP x5 in a[0]; the sha256 is in the URL
    from clip.clip import _MODELS, _download
    for n in ("RN50", "RN101", "ViT-B/16", "ViT-B/32", "ViT-L/14"):
        url = _MODELS[n]
        if get: _download(url, a[0])  # re-fetches on a hash mismatch
        else: p = Path(a[0], os.path.basename(url)); ok &= p.is_file() and sha_ok(p, url.split("/")[-2])
elif what == "resnet":  # ResNet50 for AdvDiffVLM Grad-CAM; filename has a sha256 prefix
    import torch
    from torchvision.models import ResNet50_Weights
    url = ResNet50_Weights.DEFAULT.url; p = Path(torch.hub.get_dir(), "checkpoints", os.path.basename(url))
    good = p.is_file() and sha_ok(p, p.stem.split("-")[-1])
    if not get: ok = good
    elif not good: p.unlink(missing_ok=True); torch.hub.load_state_dict_from_url(url, check_hash=True)
elif what == "hf":  # a: repo[@rev], skip patterns; blobs are named by sha256
    from huggingface_hub import list_repo_files, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    (repo, _, rev), skip = a[0].partition("@"), a[1:]
    rev = rev or None
    snaps = Path(HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "snapshots")
    blob = lambda f: f.parent / os.readlink(f)  # one hop: the blob may itself link into a shared store
    bad = lambda: [f for f in snaps.rglob("*") if f.is_symlink() and len(blob(f).name) == 64 and not sha_ok(f, blob(f).name)]
    if get:
        for f in bad(): blob(f).unlink(); f.unlink()  # corrupt: fetch again
        snapshot_download(repo, revision=rev, ignore_patterns=skip)
    else:
        try:
            snap = Path(snapshot_download(repo, revision=rev, local_files_only=True))
            try: want = [f for f in list_repo_files(repo, revision=rev) if not any(fnmatch.fnmatch(f, s) for s in skip + ["*.md", ".gitattributes"])]
            except Exception: want = []  # offline: check what is there
            ok = all((snap / f).exists() for f in want) and not bad()
        except Exception: ok = False
sys.exit(not ok)
PY
  }
  coco_ok() { [[ $(ls "$COCO/val2014" 2>/dev/null | wc -l) -ge 3000 ]]; }

  for e in "${FILES[@]}"; do
    IFS='|' read -r path sum url <<< "$e"
    need "$path" sha_ok "$path" "$sum" && fetch "$url" "$path" "$sum"
  done
  fetch_member() {  # url member dest sha256: one file out of a zip or tar archive (gdrive:<id> via gdown)
    local tmp; tmp=$(mktemp -d)
    if [[ $1 == gdrive:* ]]; then $PY -m gdown "${1#gdrive:}" -O "$tmp/archive"; else wget -q -O "$tmp/archive" "$1"; fi \
      && $PY -c "import sys, tarfile, zipfile; a, m, o = sys.argv[1:]
data = zipfile.ZipFile(a).read(m) if zipfile.is_zipfile(a) else tarfile.open(a).extractfile(m).read()
open(o, 'wb').write(data)" "$tmp/archive" "$2" "$tmp/file" \
      && sha_ok "$tmp/file" "$4" && mkdir -p "$(dirname "$3")" && mv "$tmp/file" "$3" \
      || echo "!! $3: download failed or sha256 mismatch"
    rm -rf "$tmp"
  }
  local SG_SHA=96afced00b0220f8e95bb1f39efad8fb4f965e87ff4d2d0197a58dfc150275c4 FONT_SHA=acd5fbb1985f9c2266c96a995b6cfba981609d598c9ce815388c1d5a8a23962a
  local SFONT_SHA=788abee4c806d660e8aee46689dd8540cd4bb98da03dcc9d171ce3efd99a9173
  need "$BLUE_SG (official BlueSuffix release)" sha_ok "$BLUE_SG" $SG_SHA \
    && fetch_member gdrive:15X6RbcjuvIE3SfgnzhrGJZXU9TgpHk2O fine-tuned_suffix_generator/adapter_model.safetensors "$BLUE_SG" $SG_SHA \
    || echo "!! get it from https://drive.google.com/drive/folders/1WQYXE0hD_b5xfkg-Nqvm-IonMgj0o6d4"
  need "$FONT (GNU FreeFont 20120503)" sha_ok "$FONT" $FONT_SHA \
    && fetch_member https://ftp.gnu.org/gnu/freefont/freefont-ttf-20120503.zip freefont-20120503/FreeMonoBold.ttf "$FONT" $FONT_SHA
  need "$SFONT (Liberation Fonts 2.1.5)" sha_ok "$SFONT" $SFONT_SHA \
    && fetch_member https://github.com/liberationfonts/liberation-fonts/files/7261482/liberation-fonts-ttf-2.1.5.tar.gz \
      liberation-fonts-ttf-2.1.5/LiberationSans-Bold.ttf "$SFONT" $SFONT_SHA
  need "~/.cache/clip (OpenAI CLIP x5, clip.load's default dir)" pyw check clip "$CLIP" && pyw get clip "$CLIP"
  need "torchvision ResNet50 (AdvDiffVLM Grad-CAM)" pyw check resnet && pyw get resnet
  for e in "${HF[@]}"; do
    repo=${e%%|*}; read -ra skip <<< "${e#*|}"
    need "HF $repo" pyw check hf "$repo" "${skip[@]}" && pyw get hf "$repo" "${skip[@]}"
  done
  [ -x "$GEN" ] && PYW=$GEN need "HF $QWEN (target_generator)" pyw check hf "$QWEN" && PYW=$GEN pyw get hf "$QWEN"  # its venv, its HF cache; none on CPU
  # for PA-Attack's prototypes
  need "$COCO/val2014 (3000 images)" coco_ok && $PY - "$COCO" <<'PY'
import io, json, random, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
coco = Path(sys.argv[1]); ann = coco / "annotations" / "captions_val2014.json"; out = coco / "val2014"
if not ann.exists():
    zf = zipfile.ZipFile(io.BytesIO(requests.get("http://images.cocodataset.org/annotations/annotations_trainval2014.zip").content))
    zf.extract("annotations/captions_val2014.json", coco)
random.seed(42)
picked = random.sample(json.loads(ann.read_text())["images"], 3000)
out.mkdir(parents=True, exist_ok=True)
def get(img):
    dest = out / img["file_name"]
    if not dest.exists():
        r = requests.get(img["coco_url"], timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
with ThreadPoolExecutor(16) as pool:
    list(pool.map(get, picked))
PY
  echo; [[ $missing -eq 0 ]] && echo "All weights present." \
    || echo "$missing item(s) $([[ $MODE == check ]] && echo "missing or corrupt; run without --check to fetch" || echo "fetched; run --check to verify")."
}

enter_venv() {  # a script cannot activate the caller's shell
  [ "$PY" = "$PWD/.venv/bin/python" ] && [ -z "${REDPHYS_NO_SHELL:-}" ] && [ -t 0 ] && [ -t 1 ] || return 0
  echo ">> opening a shell with .venv active (type exit to leave)"
  exec bash --rcfile <(printf '[ -f ~/.bashrc ] && . ~/.bashrc\n. %q/.venv/bin/activate\n' "$PWD") -i
}

case "${1:-all}" in
  all)     REDPHYS_NO_SHELL=1 bash "$0" env && install_weights && enter_venv ;;  # separate process: set -e is ignored before &&
  env)     (set -e; install_env) && enter_venv ;;
  weights) shift; install_weights "$@" ;;
  *)       sed -n '2,4p' "$0"; exit 1 ;;
esac
