# SLM Distillation Lab — Setup & Deployment

## Option A: Free hosted demo with Streamlit Community Cloud (recommended first)

The current Hugging Face creation screen can mark Docker and ordinary Gradio Spaces as paid. Static Spaces are free but cannot run this Python/PyTorch model-inference app. For the existing Streamlit app, try Streamlit Community Cloud first.

1. Create a GitHub repository, e.g. `slm-distillation-lab`.
2. Upload the contents of this folder to the repository root; preserve `adapters/` and `metrics/` directories.
3. Visit <https://share.streamlit.io/> and sign in with GitHub.
4. Create/deploy an app from your repo, branch `main`, file path `app.py`.
5. Wait for package installation. The first model load downloads `HuggingFaceTB/SmolLM2-360M-Instruct`; first inference may take longer.

Official instructions: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app>.

**Resource caveat:** Streamlit Community Cloud is free, but its documented resource range tops out at approximately 2.7 GB RAM. The app currently loads the base model in FP32 on CPU, which may be close to the limit. If the app reports an out-of-memory error, share the deployment logs before changing packages or model weights.

## Option B: Run locally

1. Install Python 3.11.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

The first generation downloads the public base model from Hugging Face. Keep internet access enabled for the first launch.

## Option C: Hugging Face Spaces

The account screen may mark standard Gradio and Docker Spaces as paid. Do not select Static for this app: Static Spaces do not run a Python/Streamlit inference backend.

A possible free HF route is to convert the Streamlit interface to Gradio and use ZeroGPU, if the Hugging Face personal account is in good standing (verified email and account older than 30 days). Free personal accounts can host up to two ZeroGPU Spaces. This requires a Gradio-compatible app; the current Streamlit app cannot be deployed unchanged to ZeroGPU.

Official docs: <https://huggingface.co/docs/hub/spaces-overview> and <https://huggingface.co/docs/hub/spaces-zerogpu>.

## Troubleshooting

- **Missing adapter files:** confirm all four adapter folders exist and contain `adapter_config.json` and `adapter_model.safetensors`.
- **Hugging Face download/network error:** retry after connectivity recovers. The base repository is public, so a token is not normally required.
- **Out of memory on Streamlit Community Cloud:** preserve the deployment logs and share them for a targeted lower-memory adjustment.
- **TorchAO/PEFT issue:** the app includes a defensive check for the specific stale TorchAO issue seen during Colab experiments.

## Results interpretation

Keyword-coverage proxy is not factual correctness or human evaluation. Live serving latency will differ from Colab benchmark timings. Response-level KD transfers teacher responses as training targets; it is not logit-level distillation.
