from imgen3flip import weights_path, Model, ImageBatch, OPTS
import torch
import torchvision as TV
import torchvision.transforms.functional as VF
import sys


assert weights_path.exists(), "Model weights do not exist"

assert len(sys.argv) == 3, f"Usage: {
    sys.argv[0]} <input-filename> <output-filename>"

input_filename = sys.argv[1]
output_filename = sys.argv[2]

assert input_filename != output_filename, f"Use different file names"

print("Loading the model")
model = Model()
model.load_state_dict(torch.load(weights_path))

print(f"Loading 8x8 input image from {input_filename}")
# read image and ditch alpha-channel if it presents
image = TV.io.read_image(input_filename)[:3]
# Convert range from 0..255 to 0.0..1.0
image = image / 255.0
assert image.shape[0] == 3, "RGB image expected"
# Convert C H W -> H W C
image = image.permute(1, 2, 0)
# Now add batch dimension(B=1): H W C -> 1 H W C
# We also specify H, W, C explicitly as model expect them to be 8x8x3
image = image.view(1, 8, 8, 3)

# Now construct batch that model uses
# Target and loss are not used in inference, as model code always calculates loss
dummy_target = torch.zeros(1, 64, 64, 3, **OPTS)
dummy_loss = torch.tensor(-1, **OPTS)
inference_batch = ImageBatch(
    im8=image.to(**OPTS),
    im64=dummy_target,
    loss=dummy_loss)
result = model(inference_batch)

# Now convert image to PIL format so we can save it
new_image = result.im64.detach().float().cpu()
# new_image: 1 H W C -> H W C
new_image = new_image[0]
# new_image: H W C -> C H W
new_image = new_image.permute(2, 0, 1)
assert new_image.shape == (3, 64, 64)
img = VF.to_pil_image(new_image)
# Save
print(f"Writing {img.height}x{img.width} image to {output_filename}")
img.save(output_filename)
