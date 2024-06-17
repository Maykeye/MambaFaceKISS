from mamba_ssm.modules.mamba2_simple import Mamba2Simple, rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
from einops import repeat
from image_utils import ImageDB, ImageBatch, ModelToRGB, RGBToModel, VF

epochs = 1_000
bs = 16

d_model = 768
headdim = 64
n_layer = 2

OPTS = {
    'device': "cuda",
    'dtype': torch.bfloat16
}
# Since we have KISS flip/flop think that number of mamba layers are actually 2 times higher
# This is somewhat relatable to LLM model where 1 block had two mamba layers: one replaced ATTN, one replaced MLP

weights_path = Path(
    f"data/image-flip-kisser-weights-{d_model}x{n_layer}-{str(OPTS['dtype'])}.bin")
print(f"Weight path is {str(weights_path)}")


class MambaWrap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mamba = Mamba2Simple(d_model, **OPTS, headdim=headdim)
        self.norm = nn.LayerNorm(d_model, **OPTS)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = residual + x
        return x


class MambaFlipVlop(nn.Module):
    def __init__(self, n_values) -> None:
        super().__init__()
        self.mb_forward = MambaWrap()
        self.mb_backward = MambaWrap()
        self.n_values = n_values
        self.tail_order = self.make_tail_order()

    def make_tail_order(self):
        # tail order := (default row by row starting from top left)
        # [0 1]
        # [2 3]
        rows = int(self.n_values**0.5)
        tail_order = torch.arange(0, self.n_values).view(rows, -1)
        # tail order := (go column by column starting from top left)
        # [0 2]
        # [1 3]
        tail_order = tail_order.mT
        # tail order = (go column by column starting from bottom right)
        # [3 1]
        # [2 0]
        tail_order = (self.n_values - 1) - tail_order
        # Tail order = (same but in 1D space)
        # [3 1 2 0]
        tail_order = tail_order.reshape(-1)

        # assert that we can get original using the same transform.
        # Sample is negative numbers so no value of it is equal to the value of index
        verification_sample = -1-torch.arange(0.0, self.n_values)
        verification_sample = verification_sample.view(1, self.n_values, 1)
        reordered = verification_sample[:, tail_order]
        reordered_back = reordered[:, tail_order]
        assert verification_sample.equal(reordered_back)
        return tail_order

    def forward(self, x):
        left_right = self.mb_forward(x)  # left to right
        x = self.swap_order(x)
        down_up = self.mb_backward(x)  # bottom to top
        down_up = self.swap_order(down_up)  # restore the order
        y = left_right + down_up
        return y

    def swap_order(self, x):
        # Reorder last n_values values
        T = x.shape[1]
        head = torch.arange(0, T - self.n_values)
        tail = self.tail_order + (T - self.n_values)
        seq = torch.cat((head, tail))
        x = x[:, seq]
        return x


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.from_rgb = RGBToModel(d_model, **OPTS)
        self.to_rgb = ModelToRGB(d_model, **OPTS)
        self.s0 = nn.Parameter(torch.randn(1, 1, d_model, **OPTS))
        self.s1 = nn.Parameter(torch.randn(1, 1, d_model, **OPTS))
        self.suffix = nn.Parameter(torch.randn(64*64, d_model, **OPTS))
        self.layers = nn.ModuleList([MambaFlipVlop(64*64)
                                    for _ in range(n_layer)])

    def forward(self, batch: ImageBatch, im64_2d=None, steps=3):
        for _ in range(steps):
            im64_2d = self.step(batch.im8, im64_2d)
        assert im64_2d is not None
        y_true = batch.im64
        result_batch = ImageBatch(
            im8=batch.im8,
            im64=im64_2d,
            loss=F.mse_loss(im64_2d, y_true))
        return result_batch

    def step(self, im8_2d, im64_2d) -> torch.Tensor:
        B = im8_2d.shape[0]
        if im64_2d is None:
            im64_2d = self.zoom(im8_2d)
        im8_1d = rearrange(im8_2d, "B H W C -> B (H W) C")
        im64_1d = rearrange(im64_2d, "B H W C -> B (H W) C")
        im8_emb = self.from_rgb(im8_1d)
        im64_emb = self.from_rgb(im64_1d) + self.suffix

        prefix = self.s0.repeat(B, 1, 1)
        separator = self.s1.repeat(B, 1, 1)
        inputs = torch.cat((prefix, im8_emb, separator, im64_emb), 1)
        result_emb = self.mamba(inputs)
        result_emb = result_emb[:, -64*64:]
        result_1d = self.to_rgb(result_emb)
        result_2d = rearrange(result_1d, "B (H W) C -> B H W C", H=64)
        return result_2d

    def zoom(self, x):
        x = rearrange(x, "B H W C -> B C H W", H=8, W=8)
        x = VF.resize(x, [64, 64])  # bicubic resize
        x = rearrange(x, "B C H W -> B H W C")
        return x

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
