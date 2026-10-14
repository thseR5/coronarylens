"""Hugging Face Space entry point (Gradio SDK, ZeroGPU hardware).

Runs the unchanged CoronaryLens app (FastAPI API + built React frontend) on the port the Space expects (7860).
A small Gradio page linking to the app is mounted at /gradio. The model runs on CPU; ZeroGPU expects at least one
@spaces.GPU function, so a no-op one is registered and never called.
"""

# ruff: noqa: I001  (order matters: spaces must be imported before gradio on ZeroGPU)
import os

# Hugging Face enables Gradio server-side rendering, whose Node server would take port 7860 before our app.
os.environ["GRADIO_SSR_MODE"] = "false"

import spaces  # (must be imported before other CUDA-aware packages on ZeroGPU)

import gradio as gr
import uvicorn
from fastapi import FastAPI

from backend.app.main import app as coronarylens


@spaces.GPU
def _gpu_noop() -> str:
    return "unused"


with gr.Blocks(title="CoronaryLens") as status:
    gr.Markdown(
        "# CoronaryLens\n\n"
        "The interactive app is at the [root of this Space](/). "
        "Code and report: https://github.com/thseR5/coronarylens"
    )

# mounted apps do not run their own lifespan, so the outer app runs CoronaryLens startup (model load, worker)
root = FastAPI(title="CoronaryLens Space", lifespan=coronarylens.router.lifespan_context)
root = gr.mount_gradio_app(root, status, path="/gradio", ssr_mode=False)
root.mount("/", coronarylens)  # API under /api, frontend at /

if __name__ == "__main__":
    # ZeroGPU reports @spaces.GPU functions from inside gr.Blocks.launch(); we serve with uvicorn instead, so run the
    # same startup report here (only defined when the Space runs on ZeroGPU).
    from spaces import zero

    if hasattr(zero, "startup"):
        zero.startup()
    uvicorn.run(root, host="0.0.0.0", port=7860)
