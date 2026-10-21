"""CoA dataset: clean/target images and captions, paired by stem."""

from pathlib import Path
from typing import Dict, List

from ....data import image_files, load_rgb


class CoADataset:
    def __init__(self, clean_images_dir: str, clean_captions_path: str, target_images_dir: str, target_captions_path: str):
        self.clean_images_dir, self.target_images_dir = Path(clean_images_dir), Path(target_images_dir)
        self.clean_captions = self._load_captions(clean_captions_path)  # line i: image i of the sorted dir
        self.target_captions = self._load_captions(target_captions_path)
        self.clean_image_paths = image_files(self.clean_images_dir)
        self.target_image_paths = image_files(self.target_images_dir)
        self._clean, self._target = {}, {}
        for i, p in enumerate(self.clean_image_paths):
            self._clean.setdefault(p.stem, i)
        for i, p in enumerate(self.target_image_paths):
            self._target.setdefault(p.stem, i)
        self._validate_dataset()

    def _load_captions(self, captions_path: str) -> List[str]:
        """One caption per line; [] if the file is missing."""
        if captions_path is None:
            return []
        captions_path = Path(captions_path)
        if not captions_path.exists():
            print(f"  Warning: Captions file not found: {captions_path}")
            print("  Captions will be auto-generated during attack")
            return []
        return [line.strip() for line in captions_path.read_text().splitlines() if line.strip()]

    def _validate_dataset(self):
        n_clean_imgs, n_target_imgs = len(self.clean_image_paths), len(self.target_image_paths)
        n_clean_caps, n_target_caps = len(self.clean_captions), len(self.target_captions)
        if n_clean_imgs == 0:
            raise ValueError(f"No clean images found in {self.clean_images_dir}")
        if n_target_imgs == 0:
            raise ValueError(f"No target images found in {self.target_images_dir}")
        if n_clean_caps > n_clean_imgs:  # missing clean captions are generated on demand
            raise ValueError(f"Too many captions: {n_clean_caps} captions but only {n_clean_imgs} images")
        if n_target_caps > 0 and n_target_imgs != n_target_caps:
            raise ValueError(f"Mismatch: {n_target_imgs} target images but {n_target_caps} target captions")

        def _cap_status(n_caps, n_imgs):
            if n_caps == 0:
                return "auto-generate on demand"
            return "all pre-loaded" if n_caps == n_imgs else f"{n_caps} pre-loaded, {n_imgs - n_caps} auto-generate"

        print(f"✓ CoA dataset validated: {n_clean_imgs} clean, {n_target_imgs} target images")
        print(f"  Clean captions:  {_cap_status(n_clean_caps, n_clean_imgs)}")
        print(f"  Target captions: {_cap_status(n_target_caps, n_target_imgs)}")

    def get(self, stem: str) -> Dict:
        """Captions and paths of the clean image and the target with this stem."""
        if stem not in self._clean:
            raise ValueError(f"CoA: '{stem}' not found in {self.clean_images_dir}")
        if stem not in self._target:
            raise ValueError(f"CoA: no target for '{stem}' in {self.target_images_dir}")
        i, j = self._clean[stem], self._target[stem]
        target_img_path = self.target_image_paths[j]
        return {
            "clean_caption": self.clean_captions[i] if i < len(self.clean_captions) else "",
            "target_image": load_rgb(target_img_path),  # EXIF-upright, as the sources
            "target_caption": self.target_captions[j] if j < len(self.target_captions) else "",
            "target_image_path": str(target_img_path),
            "image_path": str(self.clean_image_paths[i]),
        }
