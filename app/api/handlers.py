import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
from starlette.background import BackgroundTasks
from starlette.responses import FileResponse

from app.generators.qwen import generate_image as generate_qwen_image
from app.generators.text_qwen import generate_image as generate_text_qwen_image

# общий роутер, который потом подключается в main.py
router = APIRouter()

# --- Qwen-Image-Edit-2509 (i2i + BFS) ---
qwen_router = APIRouter(prefix="/qwen", tags=["QWEN"])


@qwen_router.post("/generate")
async def generate_qwen(
    background_task: BackgroundTasks,
    prompt: str = Form(...),
    negative: str = Form(" "),
    num_inference_steps: int = Form(30),
    seed: int = Form(0),
    true_cfg_scale: float = Form(4.0),
    guidance_scale: float = Form(1.0),
    photo_1: UploadFile = File(...),
    photo_2: UploadFile | None = File(None),
):
    task_id = uuid.uuid4()
    image_1 = await photo_1.read()
    image_2 = await photo_2.read() if photo_2 else None

    background_task.add_task(
        generate_qwen_image,
        task_id,
        image_1,
        image_2,
        prompt,
        negative,
        num_inference_steps,
        seed,
        true_cfg_scale,
        guidance_scale,
    )

    return {"task_id": task_id}


@qwen_router.get("/get_file")
async def get_file_qwen(task_id: str):
    save_path = Path(__file__).parent.parent.parent / "output" / f"{task_id}.png"
    if save_path.exists():
        return FileResponse(path=save_path, filename=f"{task_id}.png", status_code=200)
    else:
        return {"error": "Processing or doesn't exist"}


# --- Qwen-Image (text-to-image) ---
text_qwen_router = APIRouter(prefix="/text-qwen", tags=["QWEN_TEXT"])


@text_qwen_router.post("/generate")
async def generate_text_qwen(
    background_task: BackgroundTasks,
    prompt: str = Form(...),
    negative: str = Form(" "),
    num_inference_steps: int = Form(50),
    seed: int = Form(0),
    true_cfg_scale: float = Form(4.0),
    guidance_scale: float = Form(1.0),
):
    """
    Чистый text-to-image: только prompt/negative + настройки, без загрузки фото.
    """
    task_id = uuid.uuid4()

    background_task.add_task(
        generate_text_qwen_image,
        task_id,
        None,  # image_1 (игнорируется моделью)
        None,  # image_2 (игнорируется)
        prompt,
        negative,
        num_inference_steps,
        seed,
        true_cfg_scale,
        guidance_scale,
    )

    return {"task_id": task_id}


@text_qwen_router.get("/get_file")
async def get_file_text_qwen(task_id: str):
    save_path = Path(__file__).parent.parent.parent / "output" / f"{task_id}.png"
    if save_path.exists():
        return FileResponse(path=save_path, filename=f"{task_id}.png", status_code=200)
    else:
        return {"error": "Processing or doesn't exist"}


# подключаем оба под-роутера
router.include_router(qwen_router)
router.include_router(text_qwen_router)
