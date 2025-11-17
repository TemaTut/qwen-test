import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
from starlette.background import BackgroundTasks
from starlette.responses import FileResponse

from app.generators.qwen import generate_image

router = APIRouter(prefix="/qwen", tags=["QWEN"])


@router.post("/generate")
async def generate(
        background_task: BackgroundTasks,
        prompt: str = Form(...),
        negative: str = Form(" "),
        num_inference_steps: int = Form(30),
        photo_1: UploadFile = File(...),
        photo_2: UploadFile | None = File(None),
):
    task_id = uuid.uuid4()
    image_1 = await photo_1.read()
    image_2 = await photo_2.read() if photo_2 else None
    background_task.add_task(
        generate_image,
        task_id,
        image_1,
        image_2,
        prompt,
        negative,
        num_inference_steps,
    )
    return {"task_id": task_id}


@router.get("/get_file")
async def get_file(task_id: str):
    save_path = Path(__file__).parent.parent.parent / "output" / f"{task_id}.png"
    if save_path.exists():
        return FileResponse(path=save_path, filename=f"{task_id}.png", status_code=200)
    else:
        return {"error": "Processing or doesn't exist"}
