from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import UUID

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline


# Базовая модель Qwen Image Edit 2509
BASE_MODEL_PATH = (
    Path(__file__).parent.parent.parent / "models" / "Qwen" / "Qwen-Image-Edit-2509"
)

# Путь к INT4 LoRA адаптеру Nunchaku
NUNCHAKU_ADAPTER_PATH = Path("/workspace/models/nunchaku-qwen/int4_r32.safetensors")

TORCH_DTYPE = torch.bfloat16

_PIPELINE: QwenImageEditPlusPipeline | None = None


def get_pipeline() -> QwenImageEditPlusPipeline:
    global _PIPELINE

    if _PIPELINE is None:
        # 1) Загружаем базовый пайплайн
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            str(BASE_MODEL_PATH),
            torch_dtype=TORCH_DTYPE,
        )

        # 2) Загружаем 4-битный адаптер, который заменяет heavy UNet
        pipe.unet.load_attn_procs(NUNCHAKU_ADAPTER_PATH)

        # 3) Переезжаем на GPU
        pipe.to("cuda")

        # 4) Включаем прогресс-бар
        pipe.set_progress_bar_config(disable=None)

        _PIPELINE = pipe

    return _PIPELINE


def generate_image(
    task_id: UUID,
    image_1: bytes,
    image_2: bytes | None = None,
    prompt: str = "",
    negative: str = "",
    num_inference_steps: int = 12,
):
    pipe = get_pipeline()

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
        result = pipe(**inputs)
        image = result.images[0]

    out_dir = Path(__file__).parent.parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / f"{task_id}.png")
