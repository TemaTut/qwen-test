import os
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.handlers import router

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")

app.include_router(router=router)

def main() -> None:
    """
    Точка входа, если запускать так:
        python3 -m app.main
    """
    port = int(os.getenv("PORT", "8888"))

    try:
        uvicorn.run(app=app, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        os._exit(0)

if __name__ == "__main__":
    main()
