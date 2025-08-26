# zai_client.py
import os
import requests
from dotenv import load_dotenv
load_dotenv()

ZAI_API_KEY = os.getenv("ZAI_API_KEY", "")
ZAI_BASE_URL = os.getenv("ZAI_BASE_URL", "https://api.z.ai/paas/v4/chat/completions")
MODEL = os.getenv("ZAI_MODEL", "glm-4.5")
TIMEOUT = int(os.getenv("MODEL_TIMEOUT","30"))

if not ZAI_API_KEY:
    raise RuntimeError("Please set ZAI_API_KEY in your .env")

def chat_completion(system_prompt: str, messages: list, max_tokens:int=512, temperature:float=0.2):
    headers = {
        "Authorization": f"Bearer {ZAI_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": MODEL,
        "messages": [],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    if system_prompt:
        body["messages"].append({"role":"system","content": system_prompt})
    for m in messages:
        role = m.get("role","user")
        body["messages"].append({"role": role, "content": m.get("content","")})

    resp = requests.post(ZAI_BASE_URL, headers=headers, json=body, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"ZAI API error {resp.status_code}: {resp.text}")
    j = resp.json()
    # Parsing flexible: try common shapes
    if "choices" in j and j["choices"]:
        ch = j["choices"][0]
        if isinstance(ch.get("message"), dict):
            return ch["message"].get("content") or ch["message"].get("content", "")
        if ch.get("text"):
            return ch.get("text")
    if "data" in j and j["data"]:
        d = j["data"][0]
        return d.get("content") or d.get("text") or str(d)
    # fallback
    return str(j)
