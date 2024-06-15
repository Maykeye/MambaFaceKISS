from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2_simple import Mamba2Simple
from mamba_ssm.modules.mamba2 import Mamba2
import torch
from torch import Tensor
import random
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
from einops import rearrange, repeat
from typing import Optional


from image_utils import ImageDB, ImageBatch, RGBToModel
from image_utils import ModelToRGB
from torch_utils import model_numel

epochs = 10_000
bs = 16
d_model = 768
weights_path = Path(f"data/weights-{d_model}.bin")

OPTS = {
    'device': "cuda",
    'dtype': torch.float32
}


class MambaWrap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mamba = Mamba2Simple(d_model, **OPTS, headdim=64)
        self.norm = nn.LayerNorm(d_model, **OPTS)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = residual + x
        return x


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.from_rgb = RGBToModel(d_model, **OPTS)
        self.to_rgb = ModelToRGB(d_model, **OPTS)
        self.s0 = nn.Parameter(torch.randn(1, 1, d_model, **OPTS))
        self.suffix = nn.Parameter(torch.randn(64*64, d_model, **OPTS))
        self.layers = nn.ModuleList([MambaWrap() for _ in range(4)])
        self.norm0 = nn.LayerNorm(d_model, **OPTS)

    def forward(self, batch: ImageBatch):
        B = batch.n_batch
        batch = batch.as_1d()
        batch.im8 = self.from_rgb(batch.im8)

        s0 = self.s0.repeat(B, 1, 1)
        s1 = self.zoom(batch.im8)

        x = torch.cat((s0, batch.im8, s1), 1)
        x = self.norm0(x)
        x = self.mamba(x)
        x = x[:, -64*64:]
        y_hat = self.to_rgb(x)
        y_true = batch.im64
        batch.loss = F.mse_loss(y_hat, y_true)
        batch.im64 = y_hat
        return batch.as_2d()

    def zoom(self, im8):
        im8 = im8.view(im8.shape[0], 8, 8, im8.shape[-1])
        im8 = repeat(
            im8, "B H W C -> B (H 8) (W 8) C").view(im8.shape[0], 64*64, im8.shape[-1])
        im8 = im8 + self.suffix
        return im8

    def mamba(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main_":
    image_db = ImageDB(dtype=OPTS["dtype"])
    model = Model()
    if weights_path.exists():
        print(f"*** Load {weights_path:s}")
        model.load_state_dict(torch.load(weights_path))
    opt = torch.optim.AdamW(model.parameters(), fused=True)
    
    for e in (bar := tqdm(range(epochs))):
        b = model(image_db.random_batch(bs))
        b.loss.backward()
        opt.step()
        opt.zero_grad()
        bar.set_description(f'L:{b.loss.item():.4f}')
        if e and e % 100 == 0:
            torch.save(model.state_dict(), weights_path)
    torch.save(model.state_dict(), weights_path)
