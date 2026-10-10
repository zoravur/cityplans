"""Overnight pass: send descriptions the regex couldn't resolve to a local model via Ollama.

Setup (once):  ollama pull qwen2.5:7b-instruct      # or any instruct model you like
Run:           python analysis/llm_residue.py data/residue.csv data/llm.jsonl [--model qwen2.5:7b-instruct]
               (data/residue.csv is written by `python -m cityplans.run`)

Resumable: rows already in the output file are skipped, so you can stop and restart.
Uses only the standard library.
"""
import argparse
import csv
import json
import urllib.request
from pathlib import Path

SCHEMA = {
    "type": "object",
    "properties": {
        "residential_units": {"type": ["integer", "null"], "description": "total NEW residential units proposed"},
        "max_storeys": {"type": ["integer", "null"]},
        "n_buildings": {"type": ["integer", "null"]},
        "gfa_m2": {"type": ["number", "null"], "description": "total gross floor area in square metres"},
        "fsi": {"type": ["number", "null"]},
        "vehicle_parking": {"type": ["integer", "null"]},
        "bicycle_parking": {"type": ["integer", "null"]},
        "affordable_units": {"type": ["integer", "null"]},
        "rental_replacement_units": {"type": ["integer", "null"]},
        "tenure": {"type": "string", "enum": ["condo", "rental", "mixed", "freehold", "unknown"]},
        "primary_use": {"type": "string", "enum": [
            "residential", "mixed-use", "commercial", "office", "industrial", "institutional", "other"]},
    },
    "required": ["residential_units", "max_storeys", "tenure", "primary_use"],
}

PROMPT = """Extract facts from this City of Toronto development application description.
Rules: only use numbers stated in the text; use null if not stated. residential_units is the total
number of new homes (sum across buildings and types; exclude retail/commercial units, hotel rooms,
and beds). max_storeys is the tallest building. Respond with JSON only.

Description:
{d}"""


def call(model, desc, host):
    body = {
        "model": model, "stream": False, "format": SCHEMA,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": PROMPT.format(d=desc)}],
    }
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(json.loads(r.read())["message"]["content"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("residue")
    ap.add_argument("out")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=None, help="test on the first N rows")
    a = ap.parse_args()

    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["desc_id"] for l in out.open() if l.strip()}
    rows = [r for r in csv.DictReader(open(a.residue, encoding="utf-8")) if int(r["desc_id"]) not in done]
    if a.limit:
        rows = rows[: a.limit]
    print(f"{len(done)} already done, {len(rows)} to go with {a.model}")

    with out.open("a") as f:
        for i, r in enumerate(rows, 1):
            try:
                res = call(a.model, r["description"], a.host)
                err = None
            except Exception as e:  # keep going overnight; failures are logged and can be retried
                res, err = {}, str(e)[:200]
            f.write(json.dumps({"desc_id": int(r["desc_id"]), "error": err, **res}) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(rows)}")


if __name__ == "__main__":
    main()
