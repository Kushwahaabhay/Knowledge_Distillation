from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

APP_DIR = Path(__file__).resolve().parent
BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_DIRS = {
    "Curated SFT v1": APP_DIR / "adapters" / "curated_sft_lora",
    "Response-level KD v1": APP_DIR / "adapters" / "response_level_kd_lora",
    "Curated SFT v2": APP_DIR / "adapters" / "curated_sft_v2_lora",
    "Response-level KD v2": APP_DIR / "adapters" / "response_level_kd_v2_lora",
}
ADAPTER_NAMES = {
    "Curated SFT v1": "sft_v1",
    "Response-level KD v1": "kd_v1",
    "Curated SFT v2": "sft_v2",
    "Response-level KD v2": "kd_v2",
}
MODEL_CHOICES = ["Base SLM", *ADAPTER_DIRS.keys()]

st.set_page_config(
    page_title="SLM Distillation Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1250px; }
      .hero { padding: 1.7rem 1.8rem; border: 1px solid rgba(128,128,128,.24);
              border-radius: 22px; margin-bottom: 1.25rem;
              background: linear-gradient(125deg, rgba(46,90,160,.10), rgba(36,180,150,.08)); }
      .eyebrow { font-size: .74rem; font-weight: 750; letter-spacing: .14em;
                 text-transform: uppercase; opacity: .72; margin-bottom: .7rem; }
      .hero h1 { font-size: clamp(2rem, 4.3vw, 3.25rem); line-height: 1.04;
                 letter-spacing: -.045em; margin: 0 0 .8rem 0; }
      .hero p { max-width: 800px; font-size: 1.04rem; opacity: .82; margin: 0; }
      .section-note { opacity: .75; font-size: .9rem; }
      .result-box { border: 1px solid rgba(128,128,128,.25); border-radius: 16px;
                    padding: 1rem 1.1rem; min-height: 118px; margin-bottom: .8rem; }
      .small-label { text-transform: uppercase; font-size: .72rem; letter-spacing: .10em;
                      font-weight: 700; opacity: .68; }
      div[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.22);
                                     padding: .85rem 1rem; border-radius: 14px; }
      .stButton button { border-radius: 11px; font-weight: 650; min-height: 2.75rem; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.16); }
    </style>
    """,
    unsafe_allow_html=True,
)


def _apply_optional_torchao_compatibility_patch() -> None:
    """Avoid crashing on environments that have a stale TorchAO package preinstalled.

    These models use ordinary LoRA, not TorchAO quantization. Fresh Docker environments
    normally do not need this patch; it is primarily defensive for notebook runtimes.
    """
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.torchao as peft_lora_torchao
        import peft.utils.quantization_utils as peft_quant_utils
    except ImportError:
        return

    original_check = peft_import_utils.is_torchao_available
    state = {"warned": False}

    def safe_torchao_check() -> bool:
        try:
            return original_check()
        except Exception as exc:  # noqa: BLE001 - only suppress known TorchAO package mismatch
            message = str(exc).lower()
            if "torchao" in message and (
                "incompatible version" in message
                or "package metadata" in message
                or "distribution was not found" in message
            ):
                if not state["warned"]:
                    st.warning(
                        "An incompatible optional TorchAO installation was ignored; "
                        "this demo uses regular LoRA adapters."
                    )
                    state["warned"] = True
                return False
            raise

    peft_import_utils.is_torchao_available = safe_torchao_check
    peft_lora_torchao.is_torchao_available = safe_torchao_check
    peft_quant_utils.is_torchao_available = safe_torchao_check


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> tuple[Any, Any, torch.device]:
    """Load the base model and attach all local LoRA adapters once per app process."""
    from peft import PeftModel

    _apply_optional_torchao_compatibility_patch()

    missing = []
    for label, path in ADAPTER_DIRS.items():
        if not (path / "adapter_config.json").is_file() or not (path / "adapter_model.safetensors").is_file():
            missing.append(f"{label}: {path}")
    if missing:
        raise FileNotFoundError("Missing adapter files:\n" + "\n".join(missing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base.to(device)

    first_label = next(iter(ADAPTER_DIRS))
    model = PeftModel.from_pretrained(
        base,
        str(ADAPTER_DIRS[first_label]),
        adapter_name=ADAPTER_NAMES[first_label],
        is_trainable=False,
    )

    for label, adapter_path in list(ADAPTER_DIRS.items())[1:]:
        model.load_adapter(
            str(adapter_path),
            adapter_name=ADAPTER_NAMES[label],
            is_trainable=False,
        )
    model.eval()
    return tokenizer, model, device


def generate_answer(
    bundle: tuple[Any, Any, torch.device],
    model_label: str,
    question: str,
    max_new_tokens: int = 160,
) -> dict[str, Any]:
    tokenizer, model, device = bundle
    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic assistant for undergraduate Computer Science. "
                "Answer directly and accurately. Explain key concepts clearly. "
                "If you are unsure, state the uncertainty rather than inventing facts."
            ),
        },
        {"role": "user", "content": question.strip()},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    started = time.perf_counter()
    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "repetition_penalty": 1.08,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    with torch.inference_mode():
        if model_label == "Base SLM":
            with model.disable_adapter():
                output_ids = model.generate(**inputs, **generation_kwargs)
        else:
            model.set_adapter(ADAPTER_NAMES[model_label])
            output_ids = model.generate(**inputs, **generation_kwargs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - started, 1e-6)
    prompt_tokens = inputs["input_ids"].shape[1]
    new_tokens = max(0, output_ids.shape[1] - prompt_tokens)
    answer = tokenizer.decode(
        output_ids[0][prompt_tokens:],
        skip_special_tokens=True,
    ).strip()
    return {
        "model": model_label,
        "answer": answer or "The model returned an empty answer. Try a different question or output length.",
        "latency_seconds": elapsed,
        "generated_tokens": new_tokens,
        "tokens_per_second": new_tokens / elapsed,
        "device": str(device),
    }


def _set_single_question(value: str) -> None:
    st.session_state["single_question"] = value


def get_experiment_metrics() -> pd.DataFrame:
    path = APP_DIR / "metrics" / "experiment3_model_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    return df.rename(
        columns={
            "model": "Model",
            "questions": "Questions",
            "mean_latency_seconds": "Mean latency (s)",
            "median_latency_seconds": "Median latency (s)",
            "mean_tokens_per_second": "Mean throughput (tokens/s)",
            "mean_keyword_coverage_proxy": "Keyword-coverage proxy",
            "mean_generated_tokens": "Mean generated tokens",
            "outputs_at_300_token_cap": "Outputs at 300-token cap",
        }
    )


st.sidebar.markdown("### 🧪 SLM Distillation Lab")
st.sidebar.caption("Small Language Model research demo")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model family**")
st.sidebar.caption("SmolLM2 · 360M parameters")
st.sidebar.markdown("**Variants**")
st.sidebar.caption("Base · SFT v1/v2 · response-level KD v1/v2")
st.sidebar.markdown("---")
st.sidebar.caption("First inference downloads the public base model. No Gemini API key is required.")
st.sidebar.caption("Live speed depends on the hardware serving this app.")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Research prototype · Experiment series 01–03</div>
      <h1>Small models.<br>Measured, not assumed.</h1>
      <p>Explore how supervised fine-tuning and response-level knowledge distillation affect a compact academic Q&amp;A model. Run one model or compare their answers side by side.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_demo, tab_compare, tab_results, tab_method = st.tabs(
    ["Live demo", "Compare models", "Experiment results", "Methodology"]
)

with tab_demo:
    st.subheader("Ask an academic question")
    st.markdown(
        '<p class="section-note">Try core Computer Science topics: algorithms, databases, operating systems, networks or machine learning.</p>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.8, 1], gap="large")
    with left:
        with st.form("single_model_form"):
            question = st.text_area(
                "Question",
                placeholder="Example: Explain the four necessary conditions for deadlock.",
                height=125,
                key="single_question",
            )
            model_label = st.selectbox("Model variant", MODEL_CHOICES, index=0)
            max_tokens = st.slider("Maximum new tokens", min_value=64, max_value=300, value=160, step=16)
            submitted = st.form_submit_button("Generate answer", type="primary", use_container_width=True)
        if submitted:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                try:
                    with st.spinner("Loading model and generating answer… first load may take a little longer."):
                        bundle = load_model_bundle()
                        result = generate_answer(bundle, model_label, question, max_tokens)
                    st.session_state["single_result"] = result
                    st.session_state["single_question_used"] = question.strip()
                except Exception as exc:  # display deployment/runtime errors to the user
                    st.error(f"Model loading or generation failed: {exc}")
                    st.info("Check that all four adapter folders are present and the app has internet access to download the base model.")
    with right:
        st.markdown("#### Quick prompts")
        examples = [
            "Explain BFS versus DFS.",
            "What are the ACID properties?",
            "What are the four deadlock conditions?",
            "Explain overfitting in machine learning.",
        ]
        for idx, item in enumerate(examples):
            st.button(
                item,
                key=f"quick_{idx}",
                use_container_width=True,
                on_click=_set_single_question,
                args=(item,),
            )
        st.markdown("---")
        st.markdown("**Interpret results carefully**")
        st.caption("This 360M-parameter model is intended for a research demo. It can produce incorrect answers; do not treat its output as authoritative without verification.")

    result = st.session_state.get("single_result")
    if result:
        st.markdown("---")
        st.markdown(f"### {result['model']}")
        st.markdown(result["answer"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Generation latency", f"{result['latency_seconds']:.2f} s")
        c2.metric("Generated tokens", f"{result['generated_tokens']}")
        c3.metric("Generation throughput", f"{result['tokens_per_second']:.1f} tok/s")
        st.caption(f"Serving device: {result['device']}. Live timings vary by hardware and load.")

with tab_compare:
    st.subheader("Compare model variants")
    st.markdown(
        '<p class="section-note">Use the same prompt and deterministic decoding to inspect differences between variants.</p>',
        unsafe_allow_html=True,
    )
    with st.form("comparison_form"):
        compare_question = st.text_area(
            "Question to compare",
            value="Explain the four necessary conditions for deadlock and give a short example.",
            height=110,
            key="compare_question",
        )
        selected_models = st.multiselect(
            "Select 2–5 models",
            MODEL_CHOICES,
            default=["Base SLM", "Curated SFT v2", "Response-level KD v2"],
        )
        compare_max_tokens = st.slider(
            "Maximum new tokens per model", min_value=64, max_value=256, value=128, step=16,
            key="compare_max_tokens",
        )
        compare_submit = st.form_submit_button("Run comparison", type="primary", use_container_width=True)
    if compare_submit:
        if not compare_question.strip():
            st.warning("Enter a question to compare.")
        elif len(selected_models) < 2:
            st.warning("Choose at least two models.")
        else:
            results: list[dict[str, Any]] = []
            try:
                with st.spinner("Running selected variants. On CPU this may take a while."):
                    bundle = load_model_bundle()
                    for label in selected_models:
                        results.append(generate_answer(bundle, label, compare_question, compare_max_tokens))
                st.session_state["compare_results"] = results
                st.session_state["compare_question_used"] = compare_question.strip()
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")

    compare_results = st.session_state.get("compare_results", [])
    if compare_results:
        st.markdown("---")
        st.caption(f"Prompt: {st.session_state.get('compare_question_used', '')}")
        cols = st.columns(3)
        for idx, item in enumerate(compare_results):
            with cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(f"**{item['model']}**")
                    st.markdown(item["answer"])
                    st.caption(
                        f"{item['latency_seconds']:.2f} s · {item['generated_tokens']} tokens · "
                        f"{item['tokens_per_second']:.1f} tok/s"
                    )

with tab_results:
    st.subheader("Experiment 3 · 50-question held-out benchmark")
    metrics = get_experiment_metrics()
    if metrics.empty:
        st.warning("The bundled Experiment 3 metric file is missing.")
    else:
        metric_cols = st.columns(3)
        best_proxy = metrics.loc[metrics["Keyword-coverage proxy"].idxmax()]
        metric_cols[0].metric("Questions per variant", "50")
        metric_cols[1].metric("Highest keyword proxy", f"{best_proxy['Keyword-coverage proxy']:.1%}")
        metric_cols[1].caption(f"Highest result: {best_proxy['Model']}")
        metric_cols[2].metric("Model variants", str(len(metrics)))
        st.markdown("#### Recorded benchmark metrics")
        display = metrics.copy()
        display["Keyword-coverage proxy"] = display["Keyword-coverage proxy"].map(lambda x: f"{x:.1%}")
        display["Mean latency (s)"] = display["Mean latency (s)"].map(lambda x: f"{x:.2f}")
        display["Median latency (s)"] = display["Median latency (s)"].map(lambda x: f"{x:.2f}")
        display["Mean throughput (tokens/s)"] = display["Mean throughput (tokens/s)"].map(lambda x: f"{x:.2f}")
        display["Mean generated tokens"] = display["Mean generated tokens"].map(lambda x: f"{x:.1f}")
        st.dataframe(display, hide_index=True, use_container_width=True)

        chart_left, chart_right = st.columns(2, gap="large")
        with chart_left:
            st.markdown("#### Keyword-coverage proxy")
            coverage = metrics.set_index("Model")[["Keyword-coverage proxy"]].mul(100)
            coverage = coverage.rename(columns={"Keyword-coverage proxy": "Keyword-coverage proxy (%)"})
            st.bar_chart(coverage, use_container_width=True)
        with chart_right:
            st.markdown("#### Mean generation latency")
            latency = metrics.set_index("Model")[["Mean latency (s)"]]
            st.bar_chart(latency, use_container_width=True)
        st.info(
            "Keyword coverage measures overlap with selected expected terms. It is not factual correctness, "
            "completeness, or human evaluation. These are recorded Colab measurements; live demo timings will differ."
        )
        topic_path = APP_DIR / "metrics" / "experiment3_topic_summary.csv"
        if topic_path.exists():
            with st.expander("View subject-level benchmark metrics"):
                topic = pd.read_csv(topic_path)
                topic = topic.rename(
                    columns={
                        "topic": "Subject",
                        "model": "Model",
                        "questions": "Questions",
                        "mean_keyword_coverage_proxy": "Keyword-coverage proxy",
                        "mean_latency_seconds": "Mean latency (s)",
                        "mean_tokens_per_second": "Mean throughput (tokens/s)",
                    }
                )
                st.dataframe(topic, hide_index=True, use_container_width=True)

with tab_method:
    st.subheader("How the experiment is set up")
    st.markdown(
        """
        **Base model:** `HuggingFaceTB/SmolLM2-360M-Instruct` (approximately 360M parameters).

        **Compared variants:**
        - **Base SLM:** pretrained instruct model without a project adapter.
        - **Curated SFT v1/v2:** parameter-efficient LoRA adapters trained on curated question-answer examples.
        - **Response-level KD v1/v2:** parameter-efficient LoRA adapters trained to imitate teacher-generated answer targets.

        **Experiment 3 benchmark:** 50 held-out academic questions, 10 questions each for DSA, DBMS, Operating Systems, Computer Networks, and Machine Learning. The same question set was run through each of the five variants.

        **What is measured:** answer-generation latency, output token count, token throughput, and an automatic keyword-coverage proxy.

        **What is not established:** the proxy does not prove factual accuracy. The generated outputs showed factual mistakes during spot inspection, and the provided Experiment 3 sheet has not been independently human-scored. The demo therefore reports measured metrics without labeling them as factual accuracy or a human-evaluation score.

        **Method limitation:** response-level KD transfers teacher responses as training targets; it is not logit-level knowledge distillation. The benchmark is small, exact-text overlap checks cannot eliminate conceptual overlap, and serving-time measurements depend on hardware.
        """
    )
    st.markdown("#### Model reference")
    st.markdown(f"[{BASE_MODEL_ID}](https://huggingface.co/{BASE_MODEL_ID})")
    st.caption("Research prototype for academic demonstration. Verify important answers with trusted sources.")

st.markdown("---")
st.caption("SLM Distillation Lab · Academic prototype · Built with Streamlit, Transformers and PEFT")
