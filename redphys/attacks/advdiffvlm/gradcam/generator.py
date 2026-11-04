"""GradCAM mask generator for AdvDiffVLM spatial attention."""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


class GradCAMGenerator:
    """GradCAM masks from ResNet50."""

    def __init__(self, layer_name: str = "layer4", resolution: int = 64, device: str = "cuda"):
        self.device = device
        self.resolution = resolution
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights).to(device).eval().requires_grad_(False)  # input grads only
        layer = getattr(self.model, layer_name)
        layer.register_forward_hook(lambda _m, _i, out: setattr(self, "_feat", out))
        layer.register_full_backward_hook(lambda _m, _gi, go: setattr(self, "_grad", go[0]))
        t = weights.transforms()
        self.preprocess = transforms.Compose([
            transforms.Resize([t.crop_size[0]] * 2, interpolation=t.interpolation, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(t.mean, t.std),
        ])

    def generate(self, image: Image.Image, class_label: int) -> torch.Tensor:
        """GradCAM mask of class_label at self.resolution, in [0, 1]."""
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device).requires_grad_(True)
        self.model(img_tensor)[0, class_label].backward()
        cam = F.relu((self._grad.mean(dim=(2, 3), keepdim=True) * self._feat).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False).squeeze()
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-6)).detach()
