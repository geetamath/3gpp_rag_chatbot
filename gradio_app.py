"""
Gradio front-end for the 3GPP RAG chatbot -- deployment entry point for
Hugging Face Spaces (sdk: gradio).

Wraps the exact same pipeline used by app.py (CLI) and webapp/backend
(FastAPI) -- src/pipeline.py -- so the evidence gate / claim verification
behaviour is identical, just with a simpler hosted UI.

On first run it ingests + builds the index (from data/raw/ if present,
else falls back to the synthetic data/sample/ corpus) if the index
doesn't already exist on disk.
"""

import os
import subprocess
import sys

import gradio as gr

import config
from src.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Build the index at startup if it isn't already there (first boot on a
# fresh Space container).
# ---------------------------------------------------------------------------
if not os.path.exists(config.METADATA_PATH):
    print("[gradio_app] no index found -- running ingest + build_index ...")
    subprocess.run([sys.executable, "-m", "src.ingest"], check=True)
    subprocess.run([sys.executable, "-m", "src.build_index"], check=True)

pipeline = RAGPipeline()


def ask(question, history):
    if not question or not question.strip():
        return "", history

    result = pipeline.answer(question)
    answer = result["answer"]
    status = result["status"]

    if result.get("citations"):
        cite_lines = "\n".join(
            f"- **{c['ts_number']}** ({c['release']}), clause {c['clause']}"
            for c in result["citations"]
        )
        answer = f"{answer}\n\n**Citations:**\n{cite_lines}"

    if status != "ANSWERED":
        answer = f"⚠️ *{status}*\n\n{answer}\n\n_Reason: {result.get('gate_reason', '')}_"

    history = history + [(question, answer)]
    return "", history


with gr.Blocks(title="3GPP RAG Chatbot") as demo:
    gr.Markdown(
        "# 📡 3GPP RAG Chatbot — Near-Zero Hallucination\n"
        "Ask questions about the ingested 3GPP specification documents. "
        "Answers are grounded with a citation-verified evidence gate — "
        "if there isn't enough evidence, the bot refuses to answer rather "
        "than guess.\n\n"
        f"_Embedding backend: `{config.EMBEDDING_BACKEND}` · "
        f"LLM backend: `{config.LLM_BACKEND}`_"
    )
    chatbot = gr.Chatbot(height=450)
    msg = gr.Textbox(
        placeholder="e.g. What does the AMF do?",
        label="Your question",
    )
    clear = gr.Button("Clear chat")

    msg.submit(ask, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: (None, []), None, [msg, chatbot], queue=False)

if __name__ == "__main__":
    demo.launch()