"""Rule-based extraction of structured fields from free-text planning application descriptions.

City-agnostic: handles metric (m2) and imperial (sf) areas, storey/story spellings, number words.
"""
import re

import pandas as pd

NUMWORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}
NUM = r"\d{1,3}(?:,\d{3})+|\d+"


def normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\(\(.*?\)\)", " ", t)  # ((temporary address ...)) notes
    # "eleven (11)" -> "11"
    t = re.sub(r"\b(" + "|".join(NUMWORDS) + r")\s*\((\d+)\)", r"\2", t)
    # standalone number words -> digits ("twenty-two" handled as twenty + two is rare; skip)
    t = re.sub(r"\b(" + "|".join(NUMWORDS) + r")\b", lambda m: str(NUMWORDS[m.group(1)]), t)
    t = t.replace("m²", "m2").replace("sq. m", "sq.m").replace("square metres", "sq.m").replace("square meters", "sq.m")
    t = re.sub(r"\bstor(?:ey|y|ies|eys)\b", "storey", t)
    t = re.sub(r"(\d)-?\s*(?:and|to)\s*(\d)", r"\1, \2", t)  # "32and 38", "45- and 12" -> "32, 38"
    t = re.sub(r"(\d)-\s*,", r"\1,", t)
    t = re.sub(r"\s+", " ", t)
    return t


def to_int(s: str) -> int:
    return int(s.replace(",", ""))


STOREY_RE = re.compile(r"((?:(?:" + NUM + r")\s*(?:,|&|to|/)\s*)*(?:" + NUM + r"))\s*(?:-|\s)?\s*storey")
UNIT_RE = re.compile(
    r"(?P<pre>[a-z\-]+\s+[a-z\-]+\s+)?(?P<n>" + NUM + r")\s*(?:-|\s)\s*"
    r"(?P<mid>(?:[a-z\-]+\s+){0,3}?)(?:units?|dwellings?|dwelling units|suites?)\b"
)
EXCLUDE_UNIT_CONTEXT = re.compile(
    r"retail|commercial|office|industrial|replacement|affordable|retained|existing|demolish|"
    r"guest|hotel|bedroom|storage|locker|parking|live-work|non-residential|employment"
)
TOTAL_UNITS_RE = re.compile(r"(?:total(?:ling|ing)?|for a total of|totaling)\s*(?:of\s*)?(" + NUM + r")\s*(?:\w+\s+){0,3}?units")
GFA_RE = re.compile(r"(?:gross floor area|gfa)[^.;]{0,60}?(" + NUM + r"(?:\.\d+)?)\s*(?:sq\.?\s?m|m2)")
GFA_RE2 = re.compile(r"(" + NUM + r"(?:\.\d+)?)\s*(?:sq\.?\s?m|m2)\s*(?:of\s+)?(?:total\s+)?(?:gross floor area|gfa)")
GFA_SF_RE = re.compile(r"(" + NUM + r")\s*(?:sf|sq\.? ?ft|square feet|gsf|zsf)\b")
FSI_RE = re.compile(r"(?:fsi|floor space index|density)\D{0,25}?(\d{1,2}(?:\.\d+)?)\s*(?:times|x)?|(\d{1,2}(?:\.\d+)?)\s*(?:times the lot area|x the lot area|fsi)")
PARK_RE = re.compile(r"(" + NUM + r")\s*(?P<mid>(?:\w+\s+){0,2}?)parking (?:spaces|stalls)")
BIKE_RE = re.compile(r"(" + NUM + r")\s*(?:\w+\s+){0,1}?bi(?:cycle|ke) parking")
AFF_RE = re.compile(r"(" + NUM + r")\s*(?:\w+\s+){0,2}?affordable")
RR_RE = re.compile(r"(" + NUM + r")\s*(?:\w+\s+){0,1}?rental replacement")

FLAGS = {
    "mentions_opa": r"official plan amendment|amend the official plan|official plan and zoning",
    "mentions_rezoning": r"zoning by-?law amendment|rezoning|amend the zoning|zoning amendment|official plan and zoning",
    "townhouse": r"town ?house|townhome|back-to-back",
    "condominium": r"condominium|\bcondo\b",
    "rental": r"\brental\b|purpose-built rental",
    "mixed_use": r"mixed[- ]use",
    "retail": r"retail|commercial",
    "office": r"\boffice",
    "industrial": r"industrial|warehouse",
    "heritage": r"heritage",
    "demolition": r"demoli",
    "seniors_ltc": r"senior|retirement|long[- ]term care",
    "hotel": r"\bhotel",
    "institutional": r"school|place of worship|church|hospital|college|university",
    "tower": r"\btower",
    "podium": r"podium",
    "laneway_garden": r"laneway|garden suite",
    "multiplex": r"multiplex|fourplex|triplex|duplex",
    "affordable_mention": r"affordable",
}


def extract(text: str) -> dict:
    t = normalize(text or "")
    out = {}

    storeys = []
    for m in STOREY_RE.finditer(t):
        for n in re.findall(NUM, m.group(1)):
            v = to_int(n)
            if 1 <= v <= 110:
                storeys.append(v)
    out["max_storeys"] = max(storeys) if storeys else None
    out["n_storey_mentions"] = len(storeys)

    total = [to_int(m.group(1)) for m in TOTAL_UNITS_RE.finditer(t)]
    cands = []
    for m in UNIT_RE.finditer(t):
        ctx = (m.group("pre") or "") + (m.group("mid") or "")
        if EXCLUDE_UNIT_CONTEXT.search(ctx) or "storey" in (m.group("mid") or ""):
            continue
        v = to_int(m.group("n"))
        if 1 <= v <= 10000:
            cands.append(v)
    if total:
        out["units"], out["units_rule"] = max(total), "total"
    elif cands:
        out["units"], out["units_rule"] = max(cands), ("single" if len(set(cands)) == 1 else "max_of_many")
    else:
        out["units"], out["units_rule"] = None, None
    out["n_unit_candidates"] = len(set(cands))

    g = GFA_RE.search(t) or GFA_RE2.search(t)
    if g:
        out["gfa_m2"] = float(g.group(1).replace(",", ""))
    else:  # imperial descriptions (US cities): take the largest square-foot figure
        sf = [to_int(x) for x in GFA_SF_RE.findall(t)]
        out["gfa_m2"] = round(max(sf) * 0.092903, 1) if sf else None
    f = FSI_RE.search(t)
    fv = None
    if f:
        fv = float(f.group(1) or f.group(2))
        if not (0.1 <= fv <= 60):
            fv = None
    out["fsi"] = fv
    out["parking"] = max((to_int(m.group(1)) for m in PARK_RE.finditer(t)
                          if not re.search(r"bi(?:cycle|ke)", m.group("mid"))), default=None)
    out["bike_parking"] = max((to_int(x) for x in BIKE_RE.findall(t)), default=None)
    out["affordable_units"] = max((to_int(x) for x in AFF_RE.findall(t)), default=None)
    out["rental_replacement_units"] = max((to_int(x) for x in RR_RE.findall(t)), default=None)
    for k, pat in FLAGS.items():
        out[k] = bool(re.search(pat, t))

    residential = bool(re.search(r"residential|dwelling|\bunits?\b|town ?house|condominium|apartment|rental", t))
    out["residential_signal"] = residential
    # Needs an LLM look: residential text where units are missing or ambiguous
    out["needs_llm"] = residential and (out["units"] is None or out["units_rule"] == "max_of_many")
    return out


def form(row) -> str:
    s = row["max_storeys"]
    if row["laneway_garden"] or row["multiplex"]:
        return "gentle density"
    if row["townhouse"] and (pd.isna(s) or s <= 4):
        return "townhouse"
    if pd.isna(s):
        return "unknown"
    if s <= 4:
        return "low-rise (1-4)"
    if s <= 11:
        return "mid-rise (5-11)"
    if s <= 30:
        return "tall (12-30)"
    return "very tall (31+)"


def enrich(df: pd.DataFrame, text_col: str = "description") -> pd.DataFrame:
    """Add extracted columns to df. Extraction runs once per unique description."""
    uniq = pd.DataFrame({text_col: df[text_col].fillna("").unique()})
    feats = pd.DataFrame([extract(d) for d in uniq[text_col]])
    uniq = pd.concat([uniq, feats], axis=1)
    uniq["built_form"] = uniq.apply(form, axis=1)
    return df.merge(uniq, on=text_col, how="left")
