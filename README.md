# SLM Distillation Lab

An academic Q&A research demo for **HuggingFaceTB/SmolLM2-360M-Instruct**, comparing the base model with four LoRA adapters:

- Curated SFT v1
- Response-level KD v1
- Curated SFT v2
- Response-level KD v2

The app lets visitors ask a question to one model or compare multiple variants, and inspect the recorded Experiment 3 metrics.

## Run locally

Python 3.11 is recommended. Install the app dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app downloads the public base model from Hugging Face on first use. The four LoRA adapters and experiment metrics are included in this folder. First launch needs an internet connection.

## Free hosted deployment: Streamlit Community Cloud

Use this route if your Hugging Face Spaces creation screen marks Docker and Gradio as paid. Do **not** choose a Static Space for this app: Static hosting serves HTML/JavaScript and cannot run this Python/PyTorch inference application.

1. Create a GitHub repository for this project.
2. Upload the contents of this folder to the repository root, preserving the `adapters/` and `metrics/` subfolders.
3. Open <https://share.streamlit.io/> and sign in with GitHub.
4. Choose **Create app** (or **Deploy an app**), select your repository and `main` branch, and set the app file path to `app.py`.
5. Deploy and wait for dependencies to install. The public base model downloads on first model load; the initial request may take a while.

Official Streamlit instructions: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app>.

### Hosting limitation

Streamlit Community Cloud is free but resource-limited (the documented limit is approximately 2.7 GB RAM maximum). This app loads a 360M-parameter model in FP32 on CPU, so startup/inference could exceed the memory limit on some runs. If it does, do not keep redeploying unchanged; tell me the exact log and we can make a lower-memory version or convert the interface to Gradio for a qualifying Hugging Face ZeroGPU Space.

## Optional Hugging Face Spaces route

Hugging Face currently marks ordinary Docker and Gradio compute Spaces as requiring a paid plan. Free personal accounts that meet Hugging Face's eligibility requirements can host up to two **Gradio ZeroGPU** Spaces; this app is Streamlit, so it would need to be converted to Gradio and configured for ZeroGPU. Official docs: <https://huggingface.co/docs/hub/spaces-overview> and <https://huggingface.co/docs/hub/spaces-zerogpu>.

## Notes on interpretation

- Experiment 3's keyword-coverage proxy is **not** factual accuracy or human evaluation.
- The demo displays experiment measurements as recorded in the Colab benchmark. Live latency depends on serving hardware and should not be compared directly to benchmark timings.
- This is a research prototype. Small language models can generate inaccurate answers; verify important claims independently.
- No Gemini API key is used by this live demo.

## Model and license

The base model is published under the Apache-2.0 license: <https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct>.
