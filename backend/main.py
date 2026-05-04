from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from loguru import logger

from backend.config import settings
from backend.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title       = settings.APP_NAME,
        version     = settings.APP_VERSION,
        description = "AI-driven underwater surveillance pipeline",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],
        allow_credentials = False,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # API and WebSocket routes first
    app.include_router(router)

    # Serve built frontend
    dist = Path("frontend/dist")
    if dist.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(dist / "assets")),
            name="assets",
        )

        @app.get("/")
        async def serve_frontend():
            return FileResponse(str(dist / "index.html"))

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            file = dist / full_path
            if file.exists() and file.is_file():
                return FileResponse(str(file))
            return FileResponse(str(dist / "index.html"))

        logger.info("Serving frontend from frontend/dist")
    else:
        logger.warning("frontend/dist not found — run: cd frontend && npm run build")

    @app.on_event("startup")
    async def startup():
        logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} starting")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host             = settings.HOST,
        port             = settings.PORT,
        reload           = False,
        workers          = 1,
        ws_ping_interval = 20,
        ws_ping_timeout  = 30,
    )