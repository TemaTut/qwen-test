from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import UUID

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from nunchaku import NunchakuQwenImageTransformer2DModel


# Локальный путь к базовой модели (как у тебя уже скачано)
BASE_MODEL_PATH = (
    Path(__file__).parent.parent.parent / "models" / "Qwen" / "Qwen-Image-Edit-2509"
)

# ID квантованного трансформера на Hugging Face
NUNCHAKU_MODEL_ID = "nunchaku-tech/nunchaku-qwen-image-edit-2509"

# 4-битная модель, rank 32 (можно поднять до 128 для качества)
NUNCHAKU_RANK = 32

TORCH_DTYPE = torch.bfloat16  # 16-бит, как и было


# Глобальный пайплайн (инициализируем один раз)
_PIPELINE: QwenImageEditPlusPipeline | None = None


def get_pipeline() -> QwenImageEditPlusPipeline:
    global _PIPELINE

    if _PIPELINE is None:
        # 1. грузим 4-битный квантованный трансформер из репо Nunchaku
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            f"{NUNCHAKU_MODEL_ID}/svdq-int4_r{NUNCHAKU_RANK}-qwen-image-edit-2509.safetensors"
        )

        # 2. создаём QwenImageEditPlusPipeline, подсовывая свой transformer
        _PIPELINE = QwenImageEditPlusPipeline.from_pretrained(
            str(BASE_MODEL_PATH),
            transformer=transformer,
            torch_dtype=TORCH_DTYPE,
        )

        # 3. Гоним всё на GPU.
        # Квантизованный UNet нормально влазит в 48 ГБ VRAM,
        # так что CPU offload нам больше НЕ нужен.
        _PIPELINE.to("cuda")
        _PIPELINE.set_progress_bar_config(disable=None)

    return _PIPELINE


def generate_image(
    task_id: UUID,
    image_1: bytes,
    image_2: bytes | None = None,
    prompt: str = "",
    negative: str = "",
    num_inference_steps: int = 8,
) -> None:
    """
    Редактируем изображение через Qwen-Image-Edit-2509 (4-битная версия).

    По умолчанию делаю 8 шагов — для lightning/квантизованных
    моделей это норм. Если хочешь, можешь поднять до 12–15.
    """
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
