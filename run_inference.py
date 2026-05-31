import json
import os

# ── Configuration + Global Variables ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"                    # CUDA_VISIBLE_DEVICES
START_INDEX = 0
END_INDEX   = 943
DATA_PATH   = "data/private.jsonl"
OUTPUT_PATH = f"results/results_{START_INDEX}_{END_INDEX}.csv"
MAX_TOKENS  = 5000
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)
TEMPERATURE   = 0.3
TOP_P         = 0.95
TOP_K         = 40
REPETITION_PENALTY = 1.05

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import re
import sys
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import pandas as pd

# Function Definitions ────────────────────────────────────────────────────────────────────────────────
def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


# Data Loading ────────────────────────────────────────────────────────────────────────────────
data = [json.loads(line) for line in open(DATA_PATH)]

# Model Loading ────────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
)

# Generation with Transformer ───────────────────────────────────────────────────────────────────
prompts = []
for item in data[START_INDEX:END_INDEX]:
    system, user = build_prompt(item["question"], item.get("options"))
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts.append(prompt_text)

responses = []
for i, prompt in enumerate(prompts):
    print(f"\n── Generating Response {i} (id={data[i].get('id')}) ──")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=16384,
    ).to(llm.device)
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    with torch.no_grad():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            streamer=streamer,
        )
    # Decode only new tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()
    responses.append(response)
    print(f"\n── Finished Response {i} ──")


results = []
for item, response in zip(data[START_INDEX:END_INDEX], responses):
    results.append({
        "id":       item.get("id"),
        "response": response,
    })
df = pd.DataFrame(results)
df.to_csv(OUTPUT_PATH, index=False)