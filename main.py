import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"

images = pipe(prompt, width=256, height=256).images

for i in range(len(images)):
    images[i].save(f"astronaut_rides_horse{i}.png")