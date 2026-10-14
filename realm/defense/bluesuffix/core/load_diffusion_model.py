from pathlib import Path

import yaml
import torch

from ..._guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from .utils import dict2namespace


def load_diffusion_models(model_src, device):
    with open(Path(__file__).parent / 'diffusion_configs' / 'imagenet.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config = dict2namespace(config)
    model_config = model_and_diffusion_defaults()
    model_config.update(vars(config.model))
    diffusion, gaussian_diffusion = create_model_and_diffusion(**model_config)
    diffusion.load_state_dict(torch.load(model_src, map_location='cpu'))
    if model_config['use_fp16']:
        diffusion.convert_to_fp16()  # as DiffPure
    diffusion.eval().to(device)
    return diffusion, gaussian_diffusion
