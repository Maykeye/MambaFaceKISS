import torch.nn as nn
from torch import Tensor
from pathlib import Path
import torch
import random
import torchvision.io as VIO
import torchvision.transforms.functional as VF
from dataclasses import dataclass
from tqdm.auto import tqdm

# https://huggingface.co/datasets/huggan/anime-faces
RAW_IMAGES_PATH = Path(
    "~/Downloads/datasets/anime/anime-faces/images").expanduser()
RESOLUTIONS = [64, 8]

AS_TENSORS_64 = Path(f"data/all_images_64.bin")
AS_TENSORS_8 = Path(f"data/all_images_8.bin")


@dataclass
class ImageBatch:
    im8: Tensor
    im64: Tensor
    loss: Tensor

    @property
    def n_batch(self):
        return self.im8.shape[0]

    def as_1d(self):
        return ImageBatch(
            im8=self.im8.view(self.n_batch, 8*8, self.im8.shape[-1]),
            im64=self.im64.view(self.n_batch, 64*64, self.im64.shape[-1]),
            loss=self.loss
        )

    def as_2d(self):
        return ImageBatch(
            im8=self.im8.view(self.n_batch, 8, 8, self.im8.shape[-1]),
            im64=self.im64.view(self.n_batch, 64, 64, self.im64.shape[-1]),
            loss=self.loss
        )


class ImageDB:
    def __init__(self, val_ratio=0.05, dtype=None) -> None:
        if not AS_TENSORS_64.exists():
            self.make_tensor_version()
        print("Load tensors file")
        self.dtype = dtype or torch.bfloat16
        self.all_images_64 = torch.load(AS_TENSORS_64).to(self.dtype)
        self.all_images_8 = torch.load(AS_TENSORS_8).to(self.dtype)
        self.n_val = int(len(self.all_images_64) * val_ratio)

    def split(self, s: str):
        if s == "train":
            return {
                8: self.all_images_8[:-self.n_val],
                64: self.all_images_64[:-self.n_val]
            }
        if s == "valid":
            return {
                8: self.all_images_8[-self.n_val:],
                64: self.all_images_64[-self.n_val:]
            }
        raise ValueError(f"Invalid split {s}")

    @property
    def train_ds(self):
        return self.split("train")

    @property
    def valid_ds(self):
        return self.split("valid")

    @torch.no_grad()
    def make_tensor_version(self, path=RAW_IMAGES_PATH):
        items = list(path.glob("*.png"))
        all_tensors = [load_single_image(item) for item in tqdm(items)]
        t64 = torch.stack([t[64] for t in all_tensors])
        t8 = torch.stack([t[8] for t in all_tensors])
        torch.save(t64, AS_TENSORS_64)
        torch.save(t8, AS_TENSORS_8)
        return {8: t8, 64: t64}

    def random_batch(self, bs: int, split: str = "train"):
        split_dict = self.split(split)
        im8 = split_dict[8]
        im64 = split_dict[64]
        keys = list(range(len(im8)))
        random.shuffle(keys)
        keys = keys[: bs]
        return ImageBatch(
            im64=im64[keys].cuda(),
            im8=im8[keys].cuda(),
            loss=torch.tensor(-1))


def load_single_image(path: Path):
    im = VIO.read_image(str(path))
    im = im / 255.0
    # resize to 8x8
    im8 = VF.resize(im, [8, 8], VF.InterpolationMode.NEAREST_EXACT)
    # C H W -> H W C
    im = im.permute(1, 2, 0).contiguous()
    im8 = im8.permute(1, 2, 0).contiguous()

    return {64: im, 8: im8}


class RGBToModel(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.fc = nn.Linear(3, d_model, device=device, dtype=dtype)

    def forward(self, x):
        return self.fc(x)


class ModelToRGB(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.fc = nn.Linear(d_model, 3, device=device, dtype=dtype)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc(x)
        x = x.sigmoid()
        return x
