# -*- coding: utf-8 -*-
"""
LLM Screening Pipeline for Systematic Review (PICO-ready)
--------------------------------------------------------
What this script does:
1) Load a CSV with results from multiple databases (e.g., Scopus + WoS).
2) Deduplicate by DOI and normalized Title.
3) Call an LLM (OpenAI or Azure OpenAI) to score each record against a PICO-based rubric.
4) Add columns with scores, decision, and reason (reproducible, JSON-based output parsing).
5) Provide helper filters to shortlist the Top-10 most relevant titles for your review.
6) Export results to CSV/Excel.

How to use:
- pip install: pandas openai (or openai>=1.0.0 if using the new SDK), python-dotenv (optional)
- Set your API key in the environment: OPENAI_API_KEY="sk-..."
- Adjust PICO and weightings below if needed.
- Run in Jupyter or as a script: python llm_screening_pipeline.py --input yourfile.csv

Input CSV requirements (flexible, these columns are auto-detected when present):
- "source" (database name), "Title", "Author Keywords", "Source title", "DOI"
Optional columns (if you have them): "Abstract", "Document Type", "Publication Year"

Output columns (added):
- llm_decision (Include/Exclude/Maybe), llm_reason
- score_* fields + total_score
- extracted fields (ml_technique, control_task, domain, metrics, doi_confirmed)

Author: (your name)
"""

import os, sys, re, time, json, argparse
import pandas as pd

# --------- CONFIG: PICO, WEIGHTS, LLM MODEL ----------------------------------

# PICO (Edit these to reuse for another review)
PICO = {
    "P": "Procesos industriales con válvulas (well testing, separación trifásica, refinería y análogos).",
    "I": "Control basado en ML: redes neuronales (ANN/DL), aprendizaje por refuerzo (RL), híbridos neuro-PID, agentes/LLM.",
    "C": "Controladores PID/PI tradicionales (baseline o referencia).",
    "O": "Desempeño de lazo (overshoot, settling, IAE/ISE, error RMS), robustez, autonomía (menos intervención humana), viabilidad PLC/SCADA."
}

# Scoring weights (sum used to rank)
WEIGHTS = {
    "relevance_valve": 3,            # 0-3
    "ml_strength": 3,                # 0-3
    "vs_pid": 2,                     # 0-2
    "validation": 2,                 # 0-2
    "industrial_similarity": 2,      # 0-2
    "peer_review": 1,                # 0-1
    "data_code": 1                   # 0-1
}

# LLM settings
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")            # "openai" or "azure"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")       # choose your model
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")

DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"     # If true: simulate outputs, no API calls
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
SLEEP_BETWEEN_CALLS = float(os.environ.get("SLEEP_BETWEEN_CALLS", "0.8"))

# --------- PROMPT (PICO-aware, JSON output) ----------------------------------

PROMPT_TEMPLATE = """You are assisting a PRISMA-style systematic review. 
Rate the following record against the PICO question and produce STRICT JSON.

PICO:
- P (Population/Problem): {P}
- I (Intervention): {I}
- C (Comparison): {C}
- O (Outcomes): {O}

Rubric and scales (return integers only in the ranges indicated):
- relevance_valve: 0–3 (direct valve control with closed-loop metrics=3; tangential=0)
- ml_strength: 0–3 (RL/Deep NN/online learning=3; shallow/heuristic ML=1; none=0)
- vs_pid: 0–2 (explicit quantitative comparison vs PID=2; mentions PID as baseline=1; none=0)
- validation: 0–2 (conceptual=0; simulation=1; pilot/field or bench experiments=2)
- industrial_similarity: 0–2 (well testing/separation/ICV/choke=2; related process=1; other=0)
- peer_review: 0–1 (1 if peer-reviewed journal/conference; 0 otherwise/preprint)
- data_code: 0–1 (1 if datasets/code openly available or reproducibility noted; else 0)

Return JSON with this exact schema (no commentary):
{{
  "decision": "Include" | "Exclude" | "Maybe",
  "reason": "one-line reason for decision",
  "scores": {{
    "relevance_valve": int,
    "ml_strength": int,
    "vs_pid": int,
    "validation": int,
    "industrial_similarity": int,
    "peer_review": int,
    "data_code": int
  }},
  "notes": {{
    "ml_technique": "short string",
    "control_task": "level/pressure/flow/valve positioning/other",
    "domain": "well testing/ICV/choke/separation/other",
    "metrics": "e.g., overshoot/settling/IAE/ISE/error/robustness",
    "doi_confirmed": true | false
  }}
}}

Record:
- Title: {title}
- Source: {source_title}
- DOI: {doi}
- Keywords: {keywords}
- Abstract: {abstract}
"""

# --------- LLM client helpers ------------------------------------------------

def llm_client():
    if DRY_RUN:
        return None
    if LLM_PROVIDER == "openai":
        try:
            from openai import OpenAI
        except Exception:
            # fallback older SDK
            import openai
            return openai
        return OpenAI()
    elif LLM_PROVIDER == "azure":
        # Using Azure OpenAI via openai python SDK compatibility
        import openai
        openai.api_type = "azure"
        openai.api_base = AZURE_OPENAI_ENDPOINT
        openai.api_key = AZURE_OPENAI_API_KEY
        openai.api_version = "2024-02-15-preview"
        return openai
    else:
        raise RuntimeError("Unsupported LLM_PROVIDER. Use 'openai' or 'azure'.")

def llm_call(prompt: str) -> str:
    if DRY_RUN:
        # Simulate a mid-level Include/Maybe for testing WITHOUT API calls
        fake = {
            "decision": "Maybe",
            "reason": "DRY_RUN stub; please run with API to get real scores",
            "scores": {
                "relevance_valve": 2 if "valve" in prompt.lower() else 1,
                "ml_strength": 2 if ("reinforcement" in prompt.lower() or "neural" in prompt.lower() or "machine learning" in prompt.lower()) else 0,
                "vs_pid": 1 if "pid" in prompt.lower() else 0,
                "validation": 1,
                "industrial_similarity": 1,
                "peer_review": 1,
                "data_code": 0
            },
            "notes": {
                "ml_technique": "unknown (dry-run)",
                "control_task": "valve (guess)",
                "domain": "related",
                "metrics": "not parsed (dry-run)",
                "doi_confirmed": ("doi:" in prompt.lower())
            }
        }
        return json.dumps(fake, ensure_ascii=False)
    client = llm_client()
    # New SDK (OpenAI>=1.0.0)
    try:
        if LLM_PROVIDER == "openai":
            if hasattr(client, "chat"):  # new SDK
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role":"system","content":"You are a precise research assistant."},
                              {"role":"user","content": prompt}],
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()
            else:
                # legacy SDK
                import openai as _openai
                _openai.api_key = OPENAI_API_KEY
                resp = _openai.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[{"role":"system","content":"You are a precise research assistant."},
                              {"role":"user","content": prompt}],
                    temperature=0.0,
                )
                return resp["choices"][0]["message"]["content"].strip()
        else:
            # Azure OpenAI
            resp = client.ChatCompletion.create(
                engine=AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role":"system","content":"You are a precise research assistant."},
                          {"role":"user","content": prompt}],
                temperature=0.0,
            )
            return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return json.dumps({"decision":"Exclude","reason":f"LLM error: {e}",
                           "scores":{"relevance_valve":0,"ml_strength":0,"vs_pid":0,"validation":0,
                                     "industrial_similarity":0,"peer_review":0,"data_code":0},
                           "notes":{"ml_technique":"","control_task":"","domain":"","metrics":"","doi_confirmed":False}})

# --------- Utility: dedupe and normalization ---------------------------------

def normalize_title(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t

def is_peer_reviewed(source_title: str) -> int:
    s = (source_title or "").lower()
    if any(k in s for k in ["journal", "transactions", "proceedings", "conference", "letters", "acta", "control", "engineering", "ifac", "ieee", "elsevier", "springer", "wiley", "mdpi", "acs"]):
        return 1
    return 0

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["Title","Source title","DOI","Abstract","Author Keywords","Document Type","Publication Year","source"]:
        if c not in df.columns:
            df[c] = None
    df["title_norm"] = df["Title"].apply(normalize_title)
    # Priority: unique DOI, else by normalized title + source
    df["_doi_norm"] = df["DOI"].astype(str).str.strip().str.lower()
    # Keep best row among duplicates (prefer with DOI, with peer-review heuristic)
    df["_peer"] = df["Source title"].apply(is_peer_reviewed)
    # Collapse by DOI first
    df = df.sort_values(["_doi_norm","_peer"], ascending=[True, False])
    df = df.drop_duplicates(subset=["_doi_norm","title_norm"], keep="first")
    # If DOI missing, dedupe by title_norm alone
    df = df.sort_values(["title_norm","_peer"], ascending=[True, False])
    df = df.drop_duplicates(subset=["title_norm"], keep="first")
    return df.drop(columns=["_peer"])

# --------- Main scoring loop --------------------------------------------------

def score_with_llm(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, r in df.iterrows():
        title = str(r.get("Title",""))
        source_title = str(r.get("Source title",""))
        doi = str(r.get("DOI",""))
        abstract = str(r.get("Abstract","")) if "Abstract" in df.columns else ""
        keywords = str(r.get("Author Keywords",""))

        prompt = PROMPT_TEMPLATE.format(
            P=PICO["P"], I=PICO["I"], C=PICO["C"], O=PICO["O"],
            title=title, source_title=source_title, doi=doi, keywords=keywords, abstract=abstract
        )
        raw = llm_call(prompt)
        try:
            parsed = json.loads(raw)
        except Exception:
            # Try to extract JSON substring
            m = re.search(r"\{.*\}", raw, re.S)
            parsed = json.loads(m.group(0)) if m else {"decision":"Exclude","reason":"JSON parse fail","scores":{k:0 for k in WEIGHTS},"notes":{}}

        sc = parsed.get("scores", {})
        # compute total using our WEIGHTS (robust)
        total = 0
        for k, w in WEIGHTS.items():
            v = int(sc.get(k, 0))
            total += v  # each subscore already encodes its max, weights used in scale design
        out = {
            "llm_decision": parsed.get("decision",""),
            "llm_reason": parsed.get("reason","").strip(),
            "score_relevance_valve": int(sc.get("relevance_valve",0)),
            "score_ml_strength": int(sc.get("ml_strength",0)),
            "score_vs_pid": int(sc.get("vs_pid",0)),
            "score_validation": int(sc.get("validation",0)),
            "score_industrial_similarity": int(sc.get("industrial_similarity",0)),
            "score_peer_review": int(sc.get("peer_review",0)),
            "score_data_code": int(sc.get("data_code",0)),
            "total_score": int(total),
            "ml_technique": (parsed.get("notes",{}) or {}).get("ml_technique",""),
            "control_task": (parsed.get("notes",{}) or {}).get("control_task",""),
            "domain": (parsed.get("notes",{}) or {}).get("domain",""),
            "metrics": (parsed.get("notes",{}) or {}).get("metrics",""),
            "doi_confirmed": (parsed.get("notes",{}) or {}).get("doi_confirmed", False)
        }
        rows.append(out)
        time.sleep(SLEEP_BETWEEN_CALLS)
    add = pd.DataFrame(rows, index=df.index)
    return pd.concat([df.reset_index(drop=True), add.reset_index(drop=True)], axis=1)

# --------- Filters to reach Top-10 -------------------------------------------

def shortlist_top10(scored: pd.DataFrame) -> pd.DataFrame:
    # Example strict filter: must mention valve+control in title OR have high relevance score
    has_valve_control = scored["Title"].str.contains(r"valv", case=False, na=False) & scored["Title"].str.contains(r"control", case=False, na=False)
    strong_scores = scored["total_score"] >= 7
    include_like = scored["llm_decision"].isin(["Include","Maybe"])
    prelim = scored[ (has_valve_control | strong_scores) & include_like ].copy()

    # Prefer explicit vs PID, higher validation, then total_score
    prelim = prelim.sort_values(by=["score_vs_pid","score_validation","total_score"], ascending=[False, False, False])
    return prelim.head(10)

# --------- CLI / Notebook entry ----------------------------------------------

def run_pipeline(input_csv: str, output_csv: str = None, output_xlsx: str = None):
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")
    ddf = deduplicate(df)
    print(f"After dedup: {len(ddf)} rows")
    sdf = score_with_llm(ddf)
    print("LLM scoring completed.")
    top10 = shortlist_top10(sdf)
    print(f"Top-10 shortlisted.")

    if output_csv:
        sdf.to_csv(output_csv, index=False)
        print(f"Saved scored results to {output_csv}")
    if output_xlsx:
        with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as w:
            sdf.to_excel(w, sheet_name="All_Scored", index=False)
            top10.to_excel(w, sheet_name="Top10", index=False)
        print(f"Saved Excel with Top-10 to {output_xlsx}")
    return sdf, top10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=False, default="titles_result - Hoja 1.csv",
                        help="Path to input CSV (merged Scopus+WoS)")
    parser.add_argument("--out_csv", "-oc", required=False, default="scored_results.csv",
                        help="Path to write full scored CSV")
    parser.add_argument("--out_xlsx", "-ox", required=False, default="scored_results.xlsx",
                        help="Path to write Excel with Top-10")
    args = parser.parse_args()

    run_pipeline(args.input, args.out_csv, args.out_xlsx)
