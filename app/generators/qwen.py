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

# Путь к модели и к выходной папке
MODEL_DIR = ROOT_DIR / "models" / "Qwen" / "Qwen-Image-Edit-2509"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TORCH_DTYPE = torch.bfloat16

# Промт для BFS (активирует лору Best Face Swap)
BFS_HEAD_V3_PROMPT = (
    "head_swap: start with Picture 1 as the base image, keeping its lighting, "
    "environment, and background. remove the head from Picture 1 completely and "
    "replace it with the head from Picture 2. ensure the head and body have "
    "correct anatomical proportions, and blend the skin tones, shadows, and "
    "lighting naturally so the final result appears as one coherent, realistic person."
)

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

        # Подключаем LoRA BFS (Best Face Swap) для Qwen-Image-Edit-2509
        pipe.load_lora_weights(
            "Alissonerdx/BFS-Best-Face-Swap",
            weight_name="bfs_head_v3_qwen_image_edit_2509.safetensors",
        )

        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        _pipe = pipe
        return _pipe


def generate_image(
    task_id: UUID,
    image_1: bytes,
    image_2: bytes | None = None,  # <- игнорируем, оставлено ради совместимости
    prompt: str = "",
    negative: str = "",
    num_inference_steps: int = 30,
    seed: int = 0,
    true_cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
) -> None:
    """
    ЛОГИКА СЕЙЧАС ВСЕГДА ОДНА:

    1) Берём image_1 (оригинал с твоим лицом).
    2) Шаг 1: Qwen редактирует фото по prompt (фон, одежда, поза и т.п.).
    3) Шаг 2: BFS меняет лицо у результата на лицо из исходного image_1.
    4) Отдаём готовую картинку.

    Для фронта: как и раньше — ОДНО фото + prompt.
    """
    pipe = _get_pipeline()

    # Оригинальное фото с лицом пользователя
    orig_image = Image.open(BytesIO(image_1)).convert("RGB")

    # Для воспроизводимости можно использовать один и тот же сид
    generator = torch.manual_seed(seed)

    with _pipe_lock:
        with torch.inference_mode():
            # ---------- ШАГ 1: Qwen делает новую сцену / одежду / позу ----------
            step1_inputs = {
                "image": [orig_image],
                "prompt": prompt,
                "generator": generator,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": 1,
            }
            step1_output = pipe(**step1_inputs)
            body_image = step1_output.images[0]

            # ---------- ШАГ 2: BFS возвращает лицо с исходного фото ----------
            # Picture 1 = сгенерированное тело/фон/поза
            # Picture 2 = оригинальное лицо
            step2_inputs = {
                "image": [body_image, orig_image],
                "prompt": BFS_HEAD_V3_PROMPT,
                "generator": generator,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": 1,
            }
            step2_output = pipe(**step2_inputs)
            final_image = step2_output.images[0]

    save_path = OUTPUT_DIR / f"{task_id}.png"
    final_image.save(save_path)

    # --- зачистка мусора ---
    del step1_output
    del step2_output
    del final_image
    del body_image
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
