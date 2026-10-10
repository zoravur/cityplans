"""Fetch helpers for the open-data platforms cities use (Socrata, CKAN)."""
import io
import time

import pandas as pd
import requests

UA = {"User-Agent": "cityplans/0.1 (open-data research)"}


def _get(url, **kw):
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=120, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def socrata(domain: str, dataset: str, fields: list[str], where: str | None = None, page: int = 50000) -> pd.DataFrame:
    """Page through a Socrata dataset. Only the listed fields are requested, so personal
    contact columns many cities publish (applicant phone numbers, etc.) are never downloaded."""
    rows, offset = [], 0
    while True:
        params = {"$select": ",".join(fields), "$limit": page, "$offset": offset, "$order": ":id"}
        if where:
            params["$where"] = where
        batch = _get(f"https://{domain}/resource/{dataset}.json", params=params).json()
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return pd.DataFrame(rows, columns=fields)


def ckan_datastore_csv(base: str, resource_id: str, usecols: list[str]) -> pd.DataFrame:
    """Download a CKAN datastore resource as CSV, keeping only usecols."""
    r = _get(f"{base}/datastore/dump/{resource_id}")
    return pd.read_csv(io.StringIO(r.content.decode("utf-8")), dtype=str, usecols=usecols)
