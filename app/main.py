import os

import uvicorn
from fastapi import FastAPI

from app.api.handlers import router

app = FastAPI()
app.include_router(router=router)


def main() -> None:
    """
    Точка входа, если запускать так:
        python3 -m app.main

    Порт берём из переменной окружения PORT (по умолчанию 8000).
    """
    port = int(os.getenv("PORT", "8888"))

    try:
        uvicorn.run(app=app, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        os._exit(0)


if __name__ == "__main__":
    main()
