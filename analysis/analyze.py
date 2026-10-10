"""Text + structured analysis of Toronto development applications: TF-IDF, NMF topics, appeal classifier.

Usage (after `python -m cityplans.run toronto`):
    python analysis/analyze.py data/normalized/toronto.parquet data/
Writes data/findings.json and prints a summary.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

APPEAL = {"OMB Appeal", "OMB Approved", "Appeal Received"}
STOP_EXTRA = [
    "application", "applications", "proposal", "proposed", "proposes", "permit", "file", "files",
    "street", "avenue", "road", "st", "ave", "rd", "drive", "dr", "west", "east", "north", "south",
    "related", "lands", "site", "subject", "development", "amendment", "zoning", "law", "plan",
    "official", "storey", "units", "unit", "building", "buildings", "metres", "square", "sq", "m2",
    "toronto", "city", "address", "include", "includes", "including", "containing", "consisting",
    "new", "existing", "total", "approximately", "provide", "provided", "proposed", "located",
    # outcome words that leak the label into the text
    "appeal", "appealed", "appeals", "olt", "omb", "lpat", "tribunal", "settlement", "approved",
    "refused", "hearing", "decision", "ontario", "land",
]


def clean(t):
    t = t.lower()
    t = re.sub(r"\d[\d,\.]*", " ", t)
    t = re.sub(r"\b[a-z]{1,2}\b", " ", t)
    return t


def top_terms(vec, X, mask_a, mask_b, n=15):
    """Terms most over-represented in group A vs B (difference of mean TF-IDF)."""
    a = np.asarray(X[mask_a].mean(axis=0)).ravel()
    b = np.asarray(X[mask_b].mean(axis=0)).ravel()
    d = a - b
    terms = np.array(vec.get_feature_names_out())
    idx = d.argsort()[::-1][:n]
    return [(terms[i], round(float(d[i]), 4)) for i in idx]


def main(src, out_dir):
    out_dir = Path(out_dir)
    apps = pd.read_parquet(src)
    # map the normalized cityplans schema onto the names this script was written against
    apps = apps.rename(columns={"submitted_date": "date_submitted", "district": "ward_name"})
    apps["application_type"] = apps["raw_type"]
    apps["status"] = apps["raw_status"]
    for c in ["description", "application_type", "status", "ward_name"]:  # arrow strings -> plain object
        apps[c] = apps[c].astype(object)
    for c in apps.columns:
        if str(apps[c].dtype) in ("boolean", "bool[pyarrow]"):
            apps[c] = apps[c].fillna(False).astype(bool)
    apps = apps[apps["description"].str.len() > 5].copy()
    apps["year"] = apps["date_submitted"].dt.year
    apps["appealed"] = apps["tribunal"].fillna(False).astype(bool)
    F = {}

    # ---- Rezoning / OPA applications: the contested ones, one row per unique proposal ----
    oz = apps[apps["application_type"] == "OZ"].sort_values("date_submitted").drop_duplicates("description")
    oz = oz[oz["year"] >= 2008]
    F["n_oz"] = int(len(oz))
    F["status_oz"] = oz["status"].value_counts().to_dict()

    yr = oz.groupby("year").agg(
        proposals=("description", "size"),
        units=("units", "sum"),
        median_storeys=("max_storeys", "median"),
        share_31plus=("max_storeys", lambda s: float((s > 30).sum() / max(s.notna().sum(), 1))),
        appeal_rate=("appealed", "mean"),
    ).reset_index()
    F["by_year"] = yr.round(3).to_dict(orient="records")

    bf = oz.groupby("built_form").agg(n=("description", "size"), appeal_rate=("appealed", "mean"),
                                       median_units=("units", "median")).reset_index()
    F["by_form"] = bf.round(3).sort_values("n", ascending=False).to_dict(orient="records")

    ward = oz.groupby("ward_name").agg(n=("description", "size"), units=("units", "sum"),
                                       appeal_rate=("appealed", "mean")).reset_index()
    F["by_ward"] = ward.round(3).sort_values("units", ascending=False).to_dict(orient="records")

    # ---- TF-IDF over all unique descriptions ----
    uniq = apps.sort_values("date_submitted").drop_duplicates("description").copy()
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = list(ENGLISH_STOP_WORDS.union(STOP_EXTRA))
    vec = TfidfVectorizer(preprocessor=clean, stop_words=stop, ngram_range=(1, 2), min_df=8, max_df=0.5,
                          sublinear_tf=True)
    X = vec.fit_transform(uniq["description"])
    F["vocab_size"] = int(X.shape[1])

    early = (uniq["year"] <= 2013).values
    late = (uniq["year"] >= 2021).values
    F["terms_rising"] = top_terms(vec, X, late, early)
    F["terms_fading"] = top_terms(vec, X, early, late)

    ozmask = (uniq["application_type"] == "OZ").values
    app_m = ozmask & uniq["appealed"].values
    non_m = ozmask & ~uniq["appealed"].values
    F["terms_appealed"] = top_terms(vec, X, app_m, non_m)

    # ---- Topics (NMF) ----
    nmf = NMF(n_components=10, random_state=0, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    terms = vec.get_feature_names_out()
    topics = []
    dom = W.argmax(axis=1)
    for k, comp in enumerate(nmf.components_):
        top = [terms[i] for i in comp.argsort()[::-1][:8]]
        sel = dom == k
        topics.append({
            "topic": k, "terms": top, "n_docs": int(sel.sum()),
            "oz_appeal_rate": round(float(uniq["appealed"].values[sel & ozmask].mean()), 3) if (sel & ozmask).any() else None,
            "median_year": int(np.median(uniq["year"].values[sel])) if sel.any() else None,
        })
    F["topics"] = topics

    # ---- Baseline classifier: does the description predict a tribunal appeal? ----
    # OZ only, decided-era proposals (2008-2022) so pending ones don't dilute labels
    m = ozmask & (uniq["year"] <= 2022).values
    Xm, y = X[m], uniq["appealed"].values[m]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
    p_text = cross_val_predict(clf, Xm, y, cv=cv, method="predict_proba")[:, 1]
    sub = uniq[m]
    struct = pd.DataFrame({
        "log_storeys": np.log1p(sub["max_storeys"].fillna(0)),
        "has_storeys": sub["max_storeys"].notna().astype(int),
        "log_units": np.log1p(sub["units"].fillna(0)),
        "opa": sub["mentions_opa"].astype(int),
        "townhouse": sub["townhouse"].astype(int),
        "heritage": sub["heritage"].astype(int),
        "year": sub["year"] - 2015,
    }).values
    p_struct = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), struct, y, cv=cv,
                                 method="predict_proba")[:, 1]
    F["classifier"] = {
        "n": int(m.sum()), "base_rate": round(float(y.mean()), 3),
        "auc_text_tfidf": round(float(roc_auc_score(y, p_text)), 3),
        "auc_structured": round(float(roc_auc_score(y, p_struct)), 3),
    }
    clf.fit(Xm, y)
    coef = clf.coef_.ravel()
    F["classifier"]["top_pos"] = [terms[i] for i in coef.argsort()[::-1][:15]]
    F["classifier"]["top_neg"] = [terms[i] for i in coef.argsort()[:15]]

    # extraction coverage, for honesty
    r = uniq[uniq["residential_signal"]]
    F["coverage"] = {c: round(float(r[c].notna().mean()), 3) for c in ["units", "max_storeys", "gfa_m2", "parking", "fsi"]}
    F["n_unique"] = int(len(uniq))
    F["n_apps"] = int(len(apps))

    (out_dir / "findings.json").write_text(json.dumps(F, indent=1, default=str))
    print(json.dumps({k: F[k] for k in ["n_oz", "classifier", "coverage"]}, indent=1))
    print("rising:", [t for t, _ in F["terms_rising"]])
    print("fading:", [t for t, _ in F["terms_fading"]])
    print("appealed:", [t for t, _ in F["terms_appealed"]])
    for t in topics:
        print(t)
    print(pd.DataFrame(F["by_year"]).to_string())
    print(pd.DataFrame(F["by_form"]).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
