from io import BytesIO
from pathlib import Path
from uuid import UUID
import torch
from PIL import Image
from diffusers import DiffusionPipeline


def generate_image(
    task_id: UUID,
    image_1: bytes,
    image_2: bytes | None = None,
    prompt: str = "",
    negative: str = "",
    num_inference_steps: int = 30,
):
    cache_path = Path(__file__).parent.parent.parent / "models"

    # 🟢 КОРРЕКТНАЯ ЗАГРУЗКА — РАБОТАЕТ НА A6000 (48GB)
    pipe = DiffusionPipeline.from_pretrained(
        str(cache_path / "Qwen/Qwen-Image-Edit-2509"),
        torch_dtype=torch.float16,  # экономный формат
        device_map="auto",  # автоматический offload
    )

    # memory-efficient attention (экономия VRAM)
    pipe.enable_xformers_memory_efficient_attention()

    # НЕ НУЖНО pipe.to("cuda") — device_map="auto" сам распределит модель

    pipe.set_progress_bar_config(disable=None)

    # 🖼️ preprocessing
    images = [Image.open(BytesIO(image_1)).convert("RGB")]
    if image_2:
        images.append(Image.open(BytesIO(image_2)).convert("RGB"))

    inputs = {
        "image": images,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "negative_prompt": negative,
        "num_inference_steps": num_inference_steps,
        "num_images_per_prompt": 1,
    }

    # 🔥 Генерация (safe offload)
    with torch.inference_mode():
        output = pipe(**inputs)

        image = output.images[0]
        save_path = Path(__file__).parent.parent.parent / "output" / f"{task_id}.png"
        image.save(save_path)
