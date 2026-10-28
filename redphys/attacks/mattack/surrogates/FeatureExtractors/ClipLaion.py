import torch
from transformers import CLIPModel
from .Base import BaseFeatureExtractor
from torchvision import transforms


class ClipLaionFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(ClipLaionFeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
        res = self.model.config.vision_config.image_size
        self.normalizer = transforms.Compose(
        [
            transforms.Resize((res, res), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

    def forward(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        image_features = self.model.get_image_features(**inputs)
        # support both old (tensor) and new transformers output types
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features
