from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import UUID
import threading
import gc

import torch
from PIL import Image
from diffusers import DiffusionPipeline

# Корень проекта: /qwen-test
ROOT_DIR = Path(__file__).parent.parent.parent

# Путь к модели и к выходной папке (как было в старом коде)
MODEL_DIR = ROOT_DIR / "models" / "Qwen" / "Qwen-Image-Edit-2509"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# dtype как в твоём исходном коде
TORCH_DTYPE = torch.bfloat16

# Глобальный пайплайн и лок для потокобезопасности
_pipe_lock = threading.Lock()
_pipe: DiffusionPipeline | None = None


def _get_pipeline() -> DiffusionPipeline:
    """
    Ленивая инициализация пайплайна:
    - модель грузится ОДИН раз при первом обращении;
    - enable_model_cpu_offload() вызывается один раз;
    - дальше пайплайн только переиспользуется.
    """
    global _pipe

    if _pipe is not None:
        return _pipe

    with _pipe_lock:
        if _pipe is not None:
            return _pipe

        pipe = DiffusionPipeline.from_pretrained(
            str(MODEL_DIR),
            torch_dtype=TORCH_DTYPE,
        )
        # CPU offload оставляем, но только один раз при старте,
        # а не на каждый запрос.
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        _pipe = pipe
        return _pipe


def generate_image(
    task_id: UUID,
    image_1: bytes,
    image_2: bytes | None = None,
    prompt: str = "",
    negative: str = "",
    num_inference_steps: int = 30,
) -> None:
    """
    Основная функция генерации.

    Исправления:
    - НЕ создаёт пайплайн заново, а берёт готовый из _get_pipeline();
    - вызов пайплайна защищён локом (_pipe_lock), чтобы не было гонок при параллельных запросах;
    - после генерации подчистка временных объектов и кэша.
    """
    pipe = _get_pipeline()

    # готовим входные картинки
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

    # Один вызов пайплайна за раз
    with _pipe_lock:
        with torch.inference_mode():
            output = pipe(**inputs)

    image = output.images[0]
    save_path = OUTPUT_DIR / f"{task_id}.png"
    image.save(save_path)

    # --- зачистка мусора, чтобы не росла лишняя память ---
    del output
    del image
    del images
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
