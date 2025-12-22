# src/common.py
from __future__ import annotations
import os, re, json, time, hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import requests
from tqdm import tqdm
from pypdf import PdfReader

# ----------------------------
# IO: read docs
# ----------------------------
def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        parts.append(txt)
    return "\n".join(parts)

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# ----------------------------
# Chunking
# ----------------------------
def chunk_text(text: str, max_chars: int = 1800, overlap_chars: int = 200) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunk = text[i:j]
        chunks.append(chunk)
        if j == len(text):
            break
        i = max(0, j - overlap_chars)
    return chunks

# ----------------------------
# Robust JSON extraction
# ----------------------------
def extract_first_json_obj(s: str) -> Dict[str, Any]:
    """
    从模型输出中尽量提取第一个 JSON object。
    """
    s = s.strip()
    # 直接就是JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # 截取第一个 {...}
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in LLM output")
    candidate = m.group(0)
    # 去掉常见尾逗号
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return json.loads(candidate)

# ----------------------------
# LLM client (OpenAI-compatible chat/completions)
# ----------------------------
@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout_sec: int = 90
    max_retries: int = 3

class OpenAICompatibleClient:
    """
    兼容 /v1/chat/completions 的服务（可本地vLLM/LM Studio/Ollama网关等）。
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        last_err = None
        for k in range(self.cfg.max_retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_sec)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (k + 1))
        raise RuntimeError(f"LLM call failed: {last_err}")

# ----------------------------
# Prompts
# ----------------------------
SYSTEM_PROMPT = """你是信息抽取系统。你的任务是从给定文本中抽取“国际工程成本超支”相关的结构化知识。
必须遵守：
1) 只输出JSON，且必须满足schema；不要输出解释性文字。
2) 仅抽取文本中明确出现或可直接推断的事实；如果不确定，降低confidence或不输出。
3) 每条输出必须包含evidence.quote（原文摘录，<=200字符）与evidence.char_start/char_end（在chunk中的字符位置）。
4) 关系只能从允许集合中选择：CAUSES, INCREASES_COST_OF, LEADS_TO_CLAIM, DELAYS, MITIGATES, ASSOCIATED_WITH。
5) 节点类型只能从允许集合中选择：Project, Risk, Claim, Activity, Contract, Stakeholder, Resource, ExternalEvent。
"""

def build_user_prompt_entities(project_id: str, doc_id: str, chunk_id: str, chunk_date: str, text: str) -> str:
    return json.dumps({
        "project_id": project_id,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "chunk_date": chunk_date,
        "text": text
    }, ensure_ascii=False)

ENTITIES_SCHEMA_DESC = """输出JSON schema：
{
  "project_id": "...",
  "doc_id": "...",
  "chunk_id": "...",
  "entities": [
    {
      "temp_id": "E1",
      "type": "Risk|Claim|Stakeholder|Resource|Activity|Contract|ExternalEvent",
      "name": "规范化短名",
      "aliases": ["原文表述1","原文表述2"],
      "taxonomy": "若为Risk，填成本超支风险大类；否则null",
      "attributes": {
        "amount": null,
        "currency": null,
        "probability": null,
        "severity": "low|medium|high|null",
        "time_span": null
      },
      "confidence": 0.0
    }
  ],
  "notes": ""
}
要求：不要凭空创建实体；金额/币种只有在文本出现时才填。
"""

TRIPLES_SCHEMA_DESC = """输出JSON schema：
{
  "project_id": "...",
  "doc_id": "...",
  "chunk_id": "...",
  "triples": [
    {
      "head_temp_id": "E1",
      "relation": "CAUSES|INCREASES_COST_OF|LEADS_TO_CLAIM|DELAYS|MITIGATES|ASSOCIATED_WITH",
      "tail_temp_id": "E2",
      "polarity": "+|-|0",
      "weight_hint": "low|medium|high|null",
      "confidence": 0.0,
      "evidence": {
        "quote": "原文证据摘录（<=200字符）",
        "char_start": 0,
        "char_end": 0
      }
    }
  ]
}
要求：
- 若文本未明确表达关系，不输出。
- CAUSES/LEADS_TO_CLAIM 仅在出现因果触发词或清晰语义时输出。
- INCREASES_COST_OF 必须与成本/费用/超支/涨价/索赔金额等表述对应。
"""
