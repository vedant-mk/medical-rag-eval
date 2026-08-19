"""Generation module: Groq API client and prompt assembly."""

import os
import re
import time

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are a medical expert answering biomedical research questions. "
    "Based on the provided context, answer the question with exactly one word: yes, no, or maybe. "
    "Output ONLY that single word, nothing else."
)

SYSTEM_PROMPT_BLIND = (
    "You are a medical expert answering biomedical research questions. "
    "Answer the question with exactly one word: yes, no, or maybe. "
    "Output ONLY that single word, nothing else."
)


def build_rag_prompt(question: str, contexts: list[str]) -> list[dict]:
    ctx_block = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts))
    user_msg = f"{ctx_block}\n\nQuestion: {question}\nAnswer (yes/no/maybe):"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def build_blind_prompt(question: str) -> list[dict]:
    user_msg = f"Question: {question}\nAnswer (yes/no/maybe):"
    return [
        {"role": "system", "content": SYSTEM_PROMPT_BLIND},
        {"role": "user", "content": user_msg},
    ]


def parse_answer(text: str) -> str:
    text = text.strip().lower()
    for ans in ["maybe", "yes", "no"]:
        if ans in text:
            return ans
    return text


def call_groq(messages: list[dict], max_retries: int = 3) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=10,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
