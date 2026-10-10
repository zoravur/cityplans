"""Ingest cities, normalize, extract, and load into one DuckDB.

  python -m cityplans.run                 # all cities
  python -m cityplans.run calgary nyc     # some cities
  python -m cityplans.run --offline       # reuse cached raw downloads
"""
import argparse
import importlib
import time
from pathlib import Path

import duckdb
import pandas as pd

from .extract import enrich
from .schema import conform

CITIES = ["toronto", "calgary", "nyc", "austin"]
DATA = Path("data")


def ingest(city: str, offline: bool) -> pd.DataFrame:
    mod = importlib.import_module(f"cityplans.cities.{city}")
    raw_path = DATA / "raw" / f"{city}.parquet"
    if offline and raw_path.exists():
        raw = pd.read_parquet(raw_path)
    else:
        t = time.time()
        raw = mod.fetch()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(raw_path, index=False)
        print(f"  [{city}] fetched {len(raw):,} rows in {time.time() - t:.0f}s")
    df = conform(mod.normalize(raw), city)
    df = enrich(df)
    df["source"] = mod.SOURCE
    out = DATA / "normalized" / f"{city}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def load_db():
    con = duckdb.connect(str(DATA / "cityplans.duckdb"))
    con.execute(f"CREATE OR REPLACE TABLE applications AS SELECT * FROM read_parquet('{DATA}/normalized/*.parquet', union_by_name=true)")
    # descriptions the regex rules couldn't resolve, for analysis/llm_residue.py
    con.execute(f"""
        COPY (SELECT row_number() OVER (ORDER BY description) - 1 AS desc_id, description
              FROM (SELECT DISTINCT description FROM applications WHERE needs_llm AND length(description) > 5))
        TO '{DATA}/residue.csv' (HEADER)
    """)
    llm = DATA / "llm.jsonl"
    if llm.exists():
        con.execute(f"""
            CREATE OR REPLACE TABLE llm_extract AS
            SELECT r.description, l.* EXCLUDE (desc_id, error)
            FROM read_json_auto('{llm}') l JOIN read_csv_auto('{DATA}/residue.csv') r USING (desc_id)
            WHERE l.error IS NULL
        """)
        con.execute("""
            CREATE OR REPLACE VIEW applications_best AS
            SELECT a.* REPLACE (COALESCE(l.residential_units, a.units) AS units,
                                COALESCE(l.max_storeys, a.max_storeys) AS max_storeys),
                   l.tenure, l.primary_use
            FROM applications a LEFT JOIN llm_extract l USING (description)
        """)
    con.execute("""
        CREATE OR REPLACE VIEW rezonings AS
        SELECT *, date_diff('day', submitted_date, decision_date) AS days_to_decision
        FROM applications WHERE kind = 'rezoning'
    """)
    con.execute("""
        CREATE OR REPLACE VIEW city_summary AS
        SELECT city, count(*) AS applications,
               count(*) FILTER (kind = 'rezoning') AS rezonings,
               min(submitted_date)::DATE AS earliest, max(submitted_date)::DATE AS latest,
               round(avg((decision_date IS NOT NULL)::INT), 3) AS share_with_decision_date,
               round(avg((units IS NOT NULL)::INT) FILTER (residential_signal), 3) AS units_coverage
        FROM applications GROUP BY city ORDER BY city
    """)
    return con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cities", nargs="*", default=CITIES)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    for c in a.cities:
        print(f"{c}:")
        df = ingest(c, a.offline)
        print(f"  {len(df):,} applications; status: {df['status'].value_counts().to_dict()}")
    con = load_db()
    print(con.sql("SELECT * FROM city_summary"))
    con.close()


if __name__ == "__main__":
    main()
