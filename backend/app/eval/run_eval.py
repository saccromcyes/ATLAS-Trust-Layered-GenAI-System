import json
import requests

API = "http://localhost:8000"

def load_golden(path="app/eval/golden.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def run():
    cases = list(load_golden())
    passed = 0

    for c in cases:
        r = requests.post(f"{API}/ask", json={"question": c["question"], "top_k": 8}, timeout=180)
        r.raise_for_status()
        out = r.json()

        ans = (out.get("answer") or "").lower()
        cites = out.get("citations") or []
        ver = out.get("verification") or {}

        ok = True
        for kw in c.get("must_include", []):
            if kw.lower() not in ans:
                ok = False
        if c.get("must_cite", False) and len(cites) == 0:
            ok = False

        print("\nQ:", c["question"])
        print("PASS" if ok else "FAIL")
        print("Groundedness:", ver.get("groundedness"))
        passed += int(ok)

    print(f"\nResult: {passed}/{len(cases)} passed")

if __name__ == "__main__":
    run()
