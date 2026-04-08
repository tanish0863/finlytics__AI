"""Root server entrypoint for the Finlytics OpenEnv environment."""

from __future__ import annotations

from openenv.server import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
