import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
# assert torch.cuda.is_available()
hub_token = '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'
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=hub_token)  

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")