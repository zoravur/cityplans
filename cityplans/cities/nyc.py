"""New York City: Zoning Application Portal (ZAP) project data (Socrata).

Rezonings are projects with a zoning map amendment (ZM) action, which is encoded in the ULURP number
(e.g. C220123ZMK). ZAP marks finished projects "Complete" without saying approved or disapproved, so
those map to status "completed".
"""
import pandas as pd

from ..schema import status_map
from ..sources import socrata

CITY = "nyc"
SOURCE = "https://data.cityofnewyork.us/d/hgx4-8ukb"
FIELDS = ["project_id", "project_name", "project_brief", "project_status", "public_status", "ulurp_non",
          "actions", "ulurp_numbers", "borough", "community_district", "app_filed_date", "certified_referred",
          "completed_date"]
STATUS = {
    "Active": "pending", "On-Hold": "pending",
    "Complete": "completed",
    "Withdrawn-Other": "withdrawn", "Terminated": "withdrawn", "Terminated-Applicant Unresponsive": "withdrawn",
    "Record Closed": "closed_unknown",
}


def fetch() -> pd.DataFrame:
    return socrata("data.cityofnewyork.us", "hgx4-8ukb", FIELDS)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    r = raw.fillna("")
    out = pd.DataFrame()
    out["app_id"] = r["project_id"]
    zm = (r["ulurp_numbers"].str.contains("ZM", regex=False) | r["actions"].str.contains(r"\bZM\b")
          | r["project_brief"].str.contains("zoning map amendment", case=False))
    out["kind"] = zm.map({True: "rezoning", False: "other"})
    out["raw_type"] = (r["ulurp_non"] + " " + r["actions"]).str.strip()
    out["raw_status"] = r["project_status"]
    out["status"] = status_map(r["project_status"], STATUS, CITY)
    out["submitted_date"] = r["app_filed_date"].where(r["app_filed_date"] != "", r["certified_referred"])
    out["decision_date"] = r["completed_date"].where(out["status"].isin(["completed", "withdrawn"]))
    out["district"] = r["borough"] + " " + r["community_district"]
    out["description"] = (r["project_name"] + ". " + r["project_brief"]).str.strip(". ")
    out["url"] = "https://zap.planning.nyc.gov/projects/" + r["project_id"]
    return out
