# cityplans

Ingest planning / rezoning applications from city open-data portals into one normalized schema.

| City | Dataset | Platform | What it covers |
|---|---|---|---|
| Toronto | [Development Applications](https://open.toronto.ca/dataset/development-applications/) | CKAN | Rezonings/OPAs, site plans, subdivisions, condos (2008–) |
| Calgary | [Land Use Redesignation Applications](https://data.calgary.ca/d/33vi-ew4s) | Socrata | Rezonings (1982–) |
| New York City | [ZAP Project Data](https://data.cityofnewyork.us/d/hgx4-8ukb) | Socrata | All land-use applications; rezonings = ZM actions |
| Austin | [Zoning Cases](https://datahub.austintexas.gov/d/edir-dcnf) | Socrata | Rezonings, PUDs, historic zoning |

## Run

```bash
pip install -r requirements.txt
python -m cityplans.run              # fetch all cities -> data/cityplans.duckdb
python -m cityplans.run --offline    # re-normalize from cached raw downloads
python -m cityplans.run calgary      # one city
```

Outputs: `data/raw/<city>.parquet` (as fetched), `data/normalized/<city>.parquet`, and
`data/cityplans.duckdb` with table `applications` and views `rezonings` (adds `days_to_decision`)
and `city_summary`.

Data is not committed; the run regenerates it in under a minute.

## Analysis

`analysis/` holds the Toronto prototype that started this:

- `analyze.py` — TF-IDF, NMF topics, and a baseline tribunal-appeal classifier over Toronto descriptions:
  `python analysis/analyze.py data/normalized/toronto.parquet data`
- `llm_residue.py` — overnight pass with a local model (Ollama) over the ~4,400 descriptions the regex
  rules couldn't resolve (`data/residue.csv`). Rerun `cityplans.run --offline` afterwards and the
  `applications_best` view fills in from `data/llm.jsonl`.
- `toronto_findings.html` — a write-up of the first Toronto pass (open in a browser).

## Schema

`city, app_id, kind, status, raw_type, raw_status, tribunal, submitted_date, decision_date, address,
district, lat, lon, from_zoning, to_zoning, description, url` plus the extracted fields from
`extract.py` (`units, max_storeys, gfa_m2, fsi, parking, built_form, …`) and `source`.

**kind**: rezoning, site_plan, subdivision, condominium, heritage, other

**status**: pending, approved, refused, withdrawn, appealed, completed, closed_unknown, other.
Ambiguous statuses are never upgraded to approved. `completed` means the review finished but the
source doesn't say which way (NYC). `closed_unknown` means closed with no outcome (Toronto "Closed",
Austin "Closed" without an approval date).

## Adding a city

Create `cityplans/cities/<slug>.py` with `CITY`, `SOURCE`, `fetch()` and `normalize(raw)`, and add the
slug to `CITIES` in `run.py`. Use `status_map()` so unknown status values are printed rather than
silently dropped. Fetch only the fields you need; several portals publish applicant phone numbers and
home addresses that this project has no use for.

## What each city can and can't tell you

| | Outcome | Decision date | Units in text | Zoning from→to |
|---|---|---|---|---|
| Toronto | weak (74% "Closed") | no | yes (72%) | no |
| Calgary | strong | yes (95%) | no | yes |
| NYC | partial ("Complete") | yes (91%) | often (58%) | in text only |
| Austin | partial | for approvals | rarely | yes |

Calgary and NYC supply the timeline and outcome labels Toronto lacks; Toronto supplies the richest
project descriptions.

## Data sources and licences

This repository contains code only. When you run it, it downloads data published by:

- City of Toronto, under the [Open Government Licence – Toronto](https://www.toronto.ca/city-government/data-research-maps/open-data/open-data-licence/)
- City of Calgary, under the [Open Government Licence – City of Calgary](https://data.calgary.ca/stories/s/Open-Calgary-Terms-of-Use/u45n-7awa)
- City of New York, [NYC Open Data terms of use](https://opendata.cityofnewyork.us/)
- City of Austin, [Austin open data](https://data.austintexas.gov/)

Contains information licensed under the Open Government Licence – Toronto and the Open Government
Licence – City of Calgary.

Code is MIT licensed.
