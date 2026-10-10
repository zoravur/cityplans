"""The normalized application schema every city ingestor must produce."""
import pandas as pd

# Normalized application kinds
KINDS = {"rezoning", "site_plan", "subdivision", "condominium", "heritage", "other"}

# Normalized outcomes. Keep these honest: never upgrade an ambiguous status to "approved".
STATUSES = {
    "pending",         # still in review
    "approved",        # approved (by council, planning commission, or a tribunal)
    "refused",         # denied / refused / disapproved
    "withdrawn",       # withdrawn, cancelled, expired, terminated by applicant inactivity
    "appealed",        # currently before an appeal body (e.g. Ontario Land Tribunal)
    "completed",       # review finished but the data doesn't say which way (mostly approvals)
    "closed_unknown",  # closed with no outcome information at all
    "other",
}

COLUMNS = {
    "city": "string",            # slug: toronto, calgary, nyc, austin
    "app_id": "string",          # the city's own application number
    "kind": "string",            # one of KINDS
    "status": "string",          # one of STATUSES
    "raw_type": "string",
    "raw_status": "string",
    "tribunal": "boolean",       # ever seen at an appeal tribunal (where knowable)
    "submitted_date": "datetime64[ns]",
    "decision_date": "datetime64[ns]",  # date the outcome was reached, if published
    "address": "string",
    "district": "string",        # ward / council district / community district
    "lat": "float64",
    "lon": "float64",
    "from_zoning": "string",
    "to_zoning": "string",
    "description": "string",     # free text the extractor reads
    "url": "string",
}


def conform(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Coerce an ingestor's output to the schema and check vocabularies."""
    df = df.copy()
    df["city"] = city
    for c, t in COLUMNS.items():
        if c not in df:
            df[c] = pd.NA
        if t.startswith("datetime"):
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True).dt.tz_localize(None)
        elif t == "float64":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        elif t == "boolean":
            df[c] = df[c].astype("boolean")
        else:
            df[c] = df[c].astype(t)
    bad_k = set(df["kind"].dropna()) - KINDS
    bad_s = set(df["status"].dropna()) - STATUSES
    if bad_k or bad_s:
        raise ValueError(f"{city}: unknown kinds {bad_k} or statuses {bad_s}")
    df["description"] = df["description"].fillna("").str.strip()
    return df[list(COLUMNS)]


def status_map(raw: pd.Series, mapping: dict, city: str, default: str = "other") -> pd.Series:
    """Map raw statuses; report anything unmapped so new city vocab doesn't slip through silently."""
    s = raw.fillna("").str.strip()
    unmapped = sorted(set(s) - set(mapping))
    if unmapped:
        print(f"  [{city}] unmapped statuses -> '{default}': {unmapped}")
    return s.map(mapping).fillna(default)
