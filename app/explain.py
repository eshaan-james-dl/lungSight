"""
LLM-based explainability layer for the Grad-CAM output.

Works with ANY OpenAI-compatible vision endpoint -- Groq, OpenRouter, NVIDIA NIM,
etc. -- by just changing environment variables. No code changes needed to switch
providers.

Environment variables:
    LLM_API_KEY   - required
    LLM_BASE_URL  - default: Groq's endpoint
    LLM_MODEL     - default: Qwen vision-capable model configured by the user

Install:
    pip install openai --break-system-packages
"""

import base64

import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b")
LLM_API_KEY = os.getenv("LLM_API_KEY")

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

SYSTEM_PROMPT = """
You are assisting with an educational chest X-ray explainability tool.

You will be shown a chest X-ray with a Grad-CAM heatmap overlay. Red/yellow
regions indicate areas that influenced an AI model's prediction. They do NOT
prove that a disease is present.

Write ONLY the final explanation. Do not output internal reasoning, analysis,
planning, chain-of-thought, or <think>/<\\think> tags.

Write one short paragraph of 3-5 sentences in plain, non-technical language.

Your explanation should:
1. Describe approximately where the heatmap is concentrated.
2. Explain what that highlighted pattern can generally be associated with,
   without claiming that the disease is confirmed.
3. Consider the model's confidence when describing the prediction. If the
   confidence is low, explicitly describe the prediction as uncertain.
4. Explicitly state that this is an AI model's output for educational purposes
   only, NOT a medical diagnosis, and that a radiologist or physician should
   interpret any real X-ray.

Do not invent radiographic findings or state that the heatmap confirms disease.
Keep the explanation cautious and non-diagnostic.
"""


def clean_explanation(text: str) -> str:
    """Remove Qwen reasoning/thinking blocks, including an unclosed <think> block."""
    if not text:
        return ""

    # Remove complete <think>...</think> blocks.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Qwen can sometimes return an opening <think> without a closing tag.
    text = re.sub(
        r"<think>.*$",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove any stray closing tag.
    text = re.sub(r"</think>", "", text, flags=re.IGNORECASE)

    return text.strip()


def generate_explanation(
    overlay_image_bytes: bytes,
    label: str,
    probability: float,
) -> str:
    """Generate a concise, cautious explanation of a Grad-CAM overlay."""
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY environment variable is not set.")

    image_b64 = base64.b64encode(overlay_image_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{image_b64}"

    confidence_text = f"{probability:.2%}"

    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"The classifier predicted '{label}' with a confidence "
                            f"of {confidence_text}. Explain the Grad-CAM heatmap "
                            "cautiously. The confidence is important: if it is low, "
                            "make clear that the prediction is uncertain. Return "
                            "ONLY the final explanation."
                        ),
                    },
                ],
            },
        ],
    )

    raw_text = response.choices[0].message.content or ""
    cleaned = clean_explanation(raw_text)

    if not cleaned:
        return (
            f"The AI model predicted {label} with a confidence of "
            f"{confidence_text}, which indicates an uncertain prediction. "
            "The highlighted regions show areas that influenced the model's "
            "prediction, but they do not confirm the presence of disease. "
            "This output is for educational purposes only and is not a medical "
            "diagnosis; a radiologist or physician should interpret any real X-ray."
        )

    return cleaned
