"""Toronto: Development Applications (CKAN). Rezonings/OPAs, site plans, subdivisions, condos."""
import numpy as np
import pandas as pd
from pyproj import Transformer

from ..schema import status_map
from ..sources import ckan_datastore_csv

CITY = "toronto"
SOURCE = "https://open.toronto.ca/dataset/development-applications/"
FIELDS = ["APPLICATION_TYPE", "APPLICATION#", "STREET_NUM", "STREET_NAME", "STREET_TYPE", "STREET_DIRECTION",
          "DATE_SUBMITTED", "STATUS", "X", "Y", "DESCRIPTION", "WARD_NAME", "APPLICATION_URL"]

KIND = {"OZ": "rezoning", "SA": "site_plan", "SB": "subdivision", "CD": "condominium", "PL": "other"}
STATUS = {
    "Application Received": "pending", "Under Review": "pending", "Circulated": "pending",
    "Council Approved": "approved", "OMB Approved": "approved", "Approved": "approved",
    "NOAC Issued": "approved", "Draft Plan Approved": "approved", "Final Approval Completed": "approved",
    "Amend Drft Plan App": "approved",
    "OMB Appeal": "appealed", "Appeal Received": "appealed",
    "Refused": "refused", "OMB Refused": "refused", "OMB Partially Approved": "approved",
    "Closed": "closed_unknown",  # approved-and-done and withdrawn are indistinguishable here
}
TRIBUNAL = {"OMB Appeal", "OMB Approved", "Appeal Received", "OMB Refused", "OMB Partially Approved"}


def fetch() -> pd.DataFrame:
    return ckan_datastore_csv("https://ckan0.cf.opendata.inter.prod-toronto.ca",
                              "8907d8ed-c515-4ce9-b674-9f8c6eefcf0d", FIELDS)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    r = raw.fillna("")
    out = pd.DataFrame()
    out["app_id"] = r["APPLICATION#"]
    out["kind"] = r["APPLICATION_TYPE"].map(KIND).fillna("other")
    out["raw_type"] = r["APPLICATION_TYPE"]
    out["raw_status"] = r["STATUS"].str.strip()
    out["status"] = status_map(r["STATUS"], STATUS, CITY)
    out["tribunal"] = out["raw_status"].isin(TRIBUNAL)
    out["submitted_date"] = r["DATE_SUBMITTED"]
    out["address"] = (r["STREET_NUM"] + " " + r["STREET_NAME"] + " " + r["STREET_TYPE"] + " " + r["STREET_DIRECTION"]).str.split().str.join(" ")
    out["district"] = r["WARD_NAME"]
    # Toronto publishes NAD27 / MTM zone 10 coordinates
    x = pd.to_numeric(r["X"], errors="coerce").to_numpy()
    y = pd.to_numeric(r["Y"], errors="coerce").to_numpy()
    lon, lat = Transformer.from_crs("EPSG:2019", "EPSG:4326", always_xy=True).transform(x, y)
    ok = np.isfinite(lat) & (lat > 43.55) & (lat < 43.9) & (lon > -79.7) & (lon < -79.1)  # inside Toronto
    out["lat"], out["lon"] = np.where(ok, lat, np.nan), np.where(ok, lon, np.nan)
    out["description"] = r["DESCRIPTION"]
    out["url"] = r["APPLICATION_URL"].replace("", pd.NA)
    return out
