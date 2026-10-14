# Live demo on a Hugging Face Gradio Space (ZeroGPU)

Docker Spaces need a paid plan on this account, so the demo runs as a Gradio Space: `app.py` starts the unchanged
CoronaryLens app on port 7860 (the model runs on CPU; one no-op `@spaces.GPU` function satisfies ZeroGPU).
Tested locally: fresh Python 3.12 environment with `requirements.txt` + `gradio==6.30.0`, `python app.py`,
`/api/health`, `/api/analyze`, `/api/explain` and the full UI work.

1. Create the Space at https://huggingface.co/new-space: name `coronarylens`, SDK **Gradio** (Blank), hardware
   **ZeroGPU (Free)**, Public.
2. Write token: https://huggingface.co/settings/tokens → New token → Write. Keep it private.
3. In Terminal:
   ```bash
   brew install git-lfs && git lfs install
   cd ~/dev/coronarylens
   tools/prepare_hf_gradio.sh .scratch/hf-gradio
   git clone https://huggingface.co/spaces/<hf-username>/coronarylens .scratch/hf-remote
   rsync -a --delete --exclude .git .scratch/hf-gradio/ .scratch/hf-remote/
   cd .scratch/hf-remote
   git add -A && git commit -m "CoronaryLens demo" && git push   # password = the Write token
   ```
4. Open the Space → **Logs**. Build takes about 5–10 min and ends with `Uvicorn running on http://0.0.0.0:7860`.
   The app is at `https://<hf-username>-coronarylens.hf.space/`.
