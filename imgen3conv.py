from mamba_ssm.modules.mamba2_simple import Mamba2Simple, rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
from einops import repeat


from image_utils import ImageDB, ImageBatch, RGBToModel
from image_utils import ModelToRGB

epochs = 1000
bs = 16
d_model = 768
headdim = 64
n_layer = 4

OPTS = {
    'device': "cuda",
    'dtype': torch.bfloat16
}
# Since we have KISS flip/flop think that number of mamba layers are actually 2 times higher
# This is somewhat relatable to LLM model where 1 block had two mamba layers: one replaced ATTN, one replaced MLP

weights_path = Path(
    f"data/image-conv-weights-{d_model}x{n_layer}-{str(OPTS['dtype'])}.bin")
print(f"Weight path is {str(weights_path)}")


class MambaWrap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mamba = Mamba2Simple(d_model, **OPTS, headdim=headdim)
        self.norm = nn.LayerNorm(d_model, **OPTS)
        self.norm_conv = nn.LayerNorm(d_model, **OPTS)
        self.conv = nn.Conv2d(d_model, d_model, 5, padding="same", **OPTS)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        image = x[:, -64*64:]
        image = self.norm_conv(image)
        image = rearrange(image, "B (H W) C -> B C H W", H=64)
        image = self.conv(image)
        tail = rearrange(image, "B C H W -> B (H W) C")
        head = x[:, :-64*64]
        x = torch.cat((head, tail), 1)
        x = residual + x
        return x


class MambaFlipFlop(nn.Module):
    def __init__(self, n_values) -> None:
        super().__init__()
        self.mb_forward = MambaWrap()
        self.mb_backward = MambaWrap()
        self.n_values = n_values

    def forward(self, x):
        x = self.mb_forward(x)
        x = self.swap_order(x)
        x = self.mb_backward(x)
        x = self.swap_order(x)
        return x

    def swap_order(self, x):
        T = x.shape[1]
        head = torch.arange(0, T - self.n_values)
        tail = torch.arange(T - 1, T - self.n_values - 1, -1)
        seq = torch.cat((head, tail))
        x = x[:, seq]
        return x


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.from_rgb = RGBToModel(d_model, **OPTS)
        self.to_rgb = ModelToRGB(d_model, **OPTS)
        self.s0 = nn.Parameter(torch.randn(1, 1, d_model, **OPTS))
        self.suffix = nn.Parameter(torch.randn(64*64, d_model, **OPTS))
        self.layers = nn.ModuleList([MambaFlipFlop(64*64)
                                    for _ in range(n_layer)])
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
        im8 = repeat(im8, "B H W C -> B (H 8) (W 8) C")
        im8 = im8.view(im8.shape[0], 64*64, im8.shape[-1])
        im8 = im8 + self.suffix
        return im8

    def mamba(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    image_db = ImageDB(dtype=OPTS["dtype"])
    model = Model()
    if weights_path.exists():
        print(f"*** Load {str(weights_path)}")
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
