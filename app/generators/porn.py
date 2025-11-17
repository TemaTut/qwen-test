import uuid
from pathlib import Path

from diffusers import StableDiffusionXLPipeline
import torch

def generate_image(
        task_id: uuid.UUID,
        prompt: str,
        negative: str,
        num_inference_steps: int,
) -> None:
    cache_path = Path(__file__).parent.parent.parent / "models/cyberrealisticPony_v140.safetensors"
    pipe = StableDiffusionXLPipeline.from_single_file(str(cache_path), torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    images = pipe(
        prompt=prompt,
        negative=negative,
        num_images_per_prompt=1,
        num_inference_steps=num_inference_steps,
    ).images
    image = images[0]
    save_path = Path(__file__).parent.parent.parent / "output" / f"{task_id}.png"
    image.save(save_path)

