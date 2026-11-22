"""Does the server honour standard request parameters? PRACTICE endpoints only (free). One request at a time.

  temperature 0 (greedy)  -> if honoured, a plain copy answers identically every time, and identical weights give
                             identical text: the sharpest same-model test there is (as in Gao et al., model equality
                             testing, and Zhu et al., rank-based uniformity test, where sampling noise is the obstacle)
  seed                    -> same, for sampled answers
  logprobs / top_logprobs -> if returned, token probabilities compare models directly (quantisation shows up there)

Sends 4 endpoints x 7 requests = 28 short requests (max 60 tokens): ref-01 plain, ref-03 system prompt,
ref-08 temperature disguise, ref-02 fp8. Prints what comes back. Nothing is written except extra_probes/api_params_out.jsonl.
  python extra_probes/api_params.py          (needs API_KEY; stop other collectors first: the team limit is 4 requests at a time)
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = "http://34.147.229.55/33-model-equality/v1"
PROMPT = "Continue this text with one sentence: The history of the Roman Empire"
EPS = ["ref-01", "ref-03", "ref-08", "ref-02"]
VARIANTS = [("default", {}), ("default", {}), ("temp0", {"temperature": 0}), ("temp0", {"temperature": 0}),
            ("temp0+seed", {"temperature": 0, "seed": 1}), ("temp0+seed", {"temperature": 0, "seed": 1}),
            ("logprobs", {"logprobs": True, "top_logprobs": 3})]

key = os.environ.get("API_KEY", "").strip()
if not key:
    sys.exit("Set the API key first:  read -s API_KEY; export API_KEY")
s = requests.Session(); s.headers["X-API-Key"] = key
out = open(Path(__file__).resolve().parent / "params_out.jsonl", "a", encoding="utf-8")
res = {}
for ep in EPS:
    for name, extra in VARIANTS:
        for _ in range(10):
            r = s.post(f"{BASE}/chat/completions", timeout=180,
                       json={"model": ep, "max_tokens": 60, "messages": [{"role": "user", "content": PROMPT}], **extra})
            if r.status_code in (429, 502, 503, 504):
                time.sleep(int(r.headers.get("Retry-After", 10))); continue
            break
        if r.status_code != 200:
            print(ep, name, "HTTP", r.status_code, r.text[:200]); continue
        body = r.json(); ch = body["choices"][0]
        txt = ch["message"]["content"]
        res.setdefault((ep, name), []).append(txt)
        out.write(json.dumps({"endpoint": ep, "variant": name, "params": extra, "body": body}, ensure_ascii=False) + "\n")
        out.flush()
        if name == "logprobs":
            print(f"{ep} logprobs: choice keys {sorted(ch)}; logprobs field {'PRESENT' if ch.get('logprobs') else 'absent'}"
                  f"; top-level keys {sorted(body)}")
print("\nanswers (first 90 chars):")
for (ep, name), txts in res.items():
    same = len(txts) > 1 and len(set(txts)) == 1
    print(f"{ep} {name:11s} {'IDENTICAL x2' if same else '':12s} " + " || ".join(t[:90].replace(chr(10), ' ') for t in txts))
g = {ep: res.get((ep, "temp0"), [None])[0] for ep in EPS}
print("\ngreedy answers equal across endpoints: " +
      ", ".join(f"{a}={b}: {g[a] == g[b]}" for i, a in enumerate(EPS) for b in EPS[i + 1:] if g[a] and g[b]))
