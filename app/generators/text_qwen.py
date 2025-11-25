from __future__ import annotations

from pathlib import Path
from uuid import UUID
import threading
import gc

import torch
from diffusers import DiffusionPipeline

# Корень проекта: /qwen-test
ROOT_DIR = Path(__file__).parent.parent.parent

# Путь к модели и к выходной папке
MODEL_DIR = ROOT_DIR / "models" / "Qwen" / "Qwen-Image"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        _pipe = pipe
        return _pipe


def generate_image(
    task_id: UUID,
    image_1: bytes | None = None,  # игнорируем, оставлено ради совместимости
    image_2: bytes | None = None,  # игнорируем, оставлено ради совместимости
    prompt: str = "",
    negative: str = "",
    num_inference_steps: int = 50,
    seed: int = 0,
    true_cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,  # для Qwen-Image можно не использовать, просто принимаем
) -> None:
    """
    Qwen-Image: обычный text-to-image.

    На вход берём только prompt + negative + настройки.
    Любые image_1 / image_2 игнорируем.
    """
    pipe = _get_pipeline()

    # генератор для воспроизводимости
    generator = torch.manual_seed(seed)
    negative_prompt = negative if negative.strip() else " "

    with _pipe_lock:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                # guidance_scale у Qwen-Image не обязателен, можем просто не передавать
                generator=generator,
                num_images_per_prompt=1,
            )
            image = result.images[0]

    save_path = OUTPUT_DIR / f"{task_id}.png"
    image.save(save_path)

    # --- зачистка мусора ---
    del result
    del image
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
