"""Calgary: Land Use Redesignation Applications (Socrata). Calgary's equivalent of a rezoning."""
import pandas as pd

from ..schema import status_map
from ..sources import socrata

CITY = "calgary"
SOURCE = "https://data.calgary.ca/d/33vi-ew4s"
FIELDS = ["permittype", "permitnum", "description", "statuscurrent", "applieddate", "completeddate",
          "fromlud", "proposedlud", "address", "latitude", "longitude"]
STATUS = {
    "Submitted": "pending", "New": "pending", "Under Review": "pending",
    "Approved": "approved", "Refused": "refused", "Cancelled": "withdrawn",
    "Miscellaneous": "other",
}


def fetch() -> pd.DataFrame:
    return socrata("data.calgary.ca", "33vi-ew4s", FIELDS)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    r = raw
    out = pd.DataFrame()
    out["app_id"] = r["permitnum"]
    out["kind"] = "rezoning"
    out["raw_type"] = r["permittype"]
    out["raw_status"] = r["statuscurrent"]
    out["status"] = status_map(r["statuscurrent"], STATUS, CITY)
    out["submitted_date"] = r["applieddate"]
    out["decision_date"] = r["completeddate"].where(out["status"].isin(["approved", "refused", "withdrawn"]))
    out["address"] = r["address"]
    out["lat"] = pd.to_numeric(r["latitude"], errors="coerce")
    out["lon"] = pd.to_numeric(r["longitude"], errors="coerce")
    out["from_zoning"] = r["fromlud"]
    out["to_zoning"] = r["proposedlud"]
    out["description"] = r["description"].fillna("")
    return out
