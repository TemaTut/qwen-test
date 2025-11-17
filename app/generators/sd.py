from PIL import Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    cache_dir="./models/stable_diffusion_xl_refiner"
)

pipe.enable_model_cpu_offload()
init_image = Image.open("./megan.png").convert("RGB")
prompt = "undressed, ultrarealistic, no dress, nuked, realistic, tits, vagina"
negative_prompt = "clothes, dressed"
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    strength=0.8,
    guidance_scale=7,
    num_inference_steps=300,
).images[0]
image.save("refined.png")
