from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import UUID

import torch
from PIL import Image
from diffusers import DiffusionPipeline


# Путь к модели и тип тензоров
CACHE_PATH = (
    Path(__file__).parent.parent.parent / "models" / "Qwen" / "Qwen-Image-Edit-2509"
)
TORCH_DTYPE = torch.bfloat16  # 16 бит, как и было


# Глобальный пайплайн (инициализируется один раз)
_PIPELINE: DiffusionPipeline | None = None


def get_pipeline() -> DiffusionPipeline:
    global _PIPELINE

    if _PIPELINE is None:
        # Загружаем модель один раз
        _PIPELINE = DiffusionPipeline.from_pretrained(
            str(CACHE_PATH),
            torch_dtype=TORCH_DTYPE,  # в новых версиях можно будет поменять на dtype=...
        )

        # Никакого .to("cuda") здесь — вместо этого offload
        # Модель будет лежать в RAM и по слоям ездить на GPU
        _PIPELINE.enable_model_cpu_offload()

        # Прогрессбар оставим включённым
        _PIPELINE.set_progress_bar_config(disable=None)

    return _PIPELINE


def generate_image(
    task_id: UUID,
    image_1: bytes,
    image_2: bytes | None = None,
    prompt: str = "",
    negative: str = "",
    num_inference_steps: int = 15,
) -> None:
    pipeline = get_pipeline()

    images = [Image.open(BytesIO(image_1)).convert("RGB")]
    if image_2 is not None:
        images.append(Image.open(BytesIO(image_2)).convert("RGB"))

    inputs = {
        "image": images,
        "prompt": prompt,
        "negative_prompt": negative,
        "num_inference_steps": num_inference_steps,
        "num_images_per_prompt": 1,
        "generator": torch.manual_seed(0),
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        image = output.images[0]

    save_dir = Path(__file__).parent.parent.parent / "output"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{task_id}.png"
    image.save(save_path)
