"""Does the server accept the NATIVE OpenAI `tools` field (as in the organisers' AgentProv paper)? PRACTICE ONLY, free.
Sends 3 requests to ref-01: one with `tools` + tool_choice 'auto', one with tool_choice 'required', one plain control.
Prints the HTTP status and what comes back (text, tool_calls, or the error). Nothing else is saved.
  python me3/native_tools.py        (needs API_KEY; stop other collectors first: team limit 4 requests at a time)"""
import json, os, sys, requests

BASE = "http://34.147.229.55/33-model-equality/v1"
key = os.environ.get("API_KEY", "").strip() or sys.exit("Set the API key first:  read -s API_KEY; export API_KEY")
tools = [{"type": "function", "function": {"name": n, "description": "Send an email to a recipient.",
          "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
                         "required": ["to", "body"]}}} for n in ("send_email", "compose_mail", "email_send", "dispatch_message")]
msg = [{"role": "user", "content": "Send a short greeting to alice@example.com."}]
for name, extra in (("tools+auto", {"tools": tools, "tool_choice": "auto"}),
                    ("tools+required", {"tools": tools, "tool_choice": "required"}),
                    ("control, no tools", {})):
    r = requests.post(f"{BASE}/chat/completions", headers={"X-API-Key": key}, timeout=180,
                      json={"model": "ref-01", "max_tokens": 120, "messages": msg, **extra})
    print(f"\n=== {name}: HTTP {r.status_code}")
    try:
        body = r.json()
    except Exception:
        print(r.text[:400]); continue
    if r.status_code != 200:
        print(json.dumps(body)[:400]); continue
    m = body["choices"][0]["message"]
    print("finish_reason:", body["choices"][0].get("finish_reason"), "| message keys:", sorted(m))
    print("content:", (m.get("content") or "")[:300].replace("\n", " / "))
    if m.get("tool_calls"):
        print("tool_calls:", json.dumps(m["tool_calls"])[:400])
