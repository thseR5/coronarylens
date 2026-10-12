# Deploying the live demo to a Hugging Face Docker Space (free CPU)

Nothing here runs automatically. Deploy only after you have approved it.

## One-time setup

1. Create a free account at https://huggingface.co/join and verify your email.
2. Create a write token: https://huggingface.co/settings/tokens → **New token** → type **Write** → copy it.
3. Install the tools (macOS):
   ```bash
   brew install git-lfs huggingface-cli    # or: pip install -U "huggingface_hub[cli]"
   git lfs install
   huggingface-cli login                   # paste the token
   ```
4. Create the Space: https://huggingface.co/new-space
   - Name: `coronarylens`
   - SDK: **Docker** → template **Blank**
   - Hardware: **CPU basic** (free)
   - Visibility: Public (or Private while testing)

## Build and push

From the repository root (`~/dev/coronarylens`), with everything committed:

```bash
tools/prepare_hf_space.sh .scratch/hf-space                  # assembles the Space from HEAD
git clone https://huggingface.co/spaces/<your-hf-username>/coronarylens .scratch/hf-remote
rsync -a --delete --exclude .git .scratch/hf-space/ .scratch/hf-remote/
cd .scratch/hf-remote
git lfs track "*.joblib" "*.glb" "*.npy" "*.png" "*.pdf" "*.xlsx"
git add -A && git commit -m "CoronaryLens demo"
git push                                                      # the Space builds the Dockerfile (~5-10 min)
```

The Space card (`README.md` in the Space) comes from `deploy/hf-space/SPACE_README.md` and sets `sdk: docker` and `app_port: 8000`.

## Check

- Open `https://huggingface.co/spaces/<your-hf-username>/coronarylens`. The **Logs** tab shows the build and then `Uvicorn running on http://0.0.0.0:8000`.
- The app loads, preset **Three-vessel disease** colours the vessels, and clicking LAD shows its explanation.
- Free CPU Spaces sleep after inactivity. The first request after sleep takes about 30–60 s while the container starts.

Then add the Space URL to the README ("Live demo") and to Devpost.
