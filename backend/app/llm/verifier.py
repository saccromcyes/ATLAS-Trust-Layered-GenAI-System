import json
from typing import List, Dict, Any
from .client import OpenAIClient
from .prompts import SYSTEM_CLAIMS, SYSTEM_VERIFY

def _safe_json(text: str) -> Any:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    try:
        return json.loads(t)
    except Exception:
        return None

class Verifier:
    def __init__(self):
        self.llm = OpenAIClient()

    def extract_claims(self, answer: str) -> List[str]:
        out = self.llm.chat(
            [{"role":"system","content":SYSTEM_CLAIMS},
             {"role":"user","content":answer}],
            temperature=0.0,
        )
        data = _safe_json(out)
        if isinstance(data, list):
            claims = [str(x).strip() for x in data if str(x).strip()]
            return claims[:25]
        return [answer[:300]]

    def verify(self, claims: List[str], evidence: List[Dict[str, str]]) -> Dict[str, Any]:
        evidence_text = "\n\n".join([f"[{e['chunk_id']}] {e['text']}" for e in evidence])
        out = self.llm.chat(
            [{"role":"system","content":SYSTEM_VERIFY},
             {"role":"user","content":json.dumps({"claims": claims, "evidence": evidence_text})}],
            temperature=0.0,
        )
        data = _safe_json(out)
        if isinstance(data, dict) and "verdicts" in data and "groundedness" in data:
            return data
        return {"verdicts":[{"claim":c,"supported":False,"evidence_chunk_ids":[]} for c in claims], "groundedness":0.0}
