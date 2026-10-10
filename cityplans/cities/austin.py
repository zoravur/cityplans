"""Austin: Zoning Cases (Socrata). Applicant and owner contact columns are deliberately not fetched."""
import pandas as pd

from ..schema import status_map
from ..sources import socrata

CITY = "austin"
SOURCE = "https://datahub.austintexas.gov/d/edir-dcnf"
FIELDS = ["case_number", "case_name", "sub_type", "detailed_status", "description_of_work", "proposed_zoning",
          "existing_zoning", "site_address", "council_district", "application_start_date", "approval_date",
          "final_date", "latitude", "longitude"]
KIND = {
    "Zoning/Rezoning": "rezoning", "Planned Unit Development (PUD)": "rezoning",
    "Planned Development Agreement (PDA)": "rezoning", "Traditional Neighborhood District (TND)": "rezoning",
    "Historical": "heritage",
}
STATUS = {
    "Approved": "approved", "Approved and Released": "approved", "Reading Approved": "approved",
    "Partial Approval": "approved",
    "Denied": "refused",
    "Withdrawn": "withdrawn", "Expired": "withdrawn", "Aborted": "withdrawn", "Cancelled": "withdrawn", "VOID": "withdrawn",
    "In Review": "pending", "Pending": "pending", "Scheduled for Hearing": "pending",
    "Scheduled for Council Hearing": "pending", "Recommended for Approval": "pending", "Notice Sent": "pending",
    "Case Assigned": "pending", "Postponed": "pending", "Awaiting Update": "pending",
    "Notice/Hearing Determination": "pending", "Notice Requested": "pending",
    "Closed": "closed_unknown",
}


def fetch() -> pd.DataFrame:
    return socrata("datahub.austintexas.gov", "edir-dcnf", FIELDS)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    r = raw
    out = pd.DataFrame()
    out["app_id"] = r["case_number"]
    out["kind"] = r["sub_type"].map(KIND).fillna("other")
    out["raw_type"] = r["sub_type"]
    out["raw_status"] = r["detailed_status"]
    st = status_map(r["detailed_status"], STATUS, CITY)
    # A closed case with an approval date was approved
    st = st.mask((st == "closed_unknown") & r["approval_date"].notna(), "approved")
    out["status"] = st
    out["submitted_date"] = r["application_start_date"]
    out["decision_date"] = r["approval_date"].where(st == "approved", r["final_date"].where(st.isin(["refused", "withdrawn"])))
    out["address"] = r["site_address"].fillna("").str.replace(r"\s*,\s*,TX$", "", regex=True).str.split().str.join(" ")
    out["district"] = "District " + r["council_district"].fillna("?")
    out["lat"] = pd.to_numeric(r["latitude"], errors="coerce")
    out["lon"] = pd.to_numeric(r["longitude"], errors="coerce")
    out["from_zoning"] = r["existing_zoning"]
    out["to_zoning"] = r["proposed_zoning"]
    out["description"] = (r["case_name"].fillna("") + ". " + r["description_of_work"].fillna("")).str.strip(". ")
    return out
