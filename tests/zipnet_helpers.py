"""Deterministic, synthetic ZIPNET-shaped fixtures.

Field names come from ``source_keys.json`` (captured from the live endpoints on
2026-10-26, keys only). All values are invented. Sensitive keys are filled with
obviously fake "canary" values so tests can prove they never reach storage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "zipnet_fixtures"
SOURCE_KEYS: dict[str, dict[str, Any]] = json.loads((FIXTURES / "source_keys.json").read_text())

# Canary values for fields that must be dropped.
CANARY_PHONE = "9876500001"
CANARY_ADDRESS = "CANARY HOUSE 42 SECRET LANE"
CANARY_PERSON = "CANARY GUARDIAN"
CANARY_EMAIL = "canary@example.invalid"

SENSITIVE_FILL = {
    "Address": CANARY_ADDRESS,
    "ComplainantPhoneNo": CANARY_PHONE,
    "ComplainantPhoneNumber": CANARY_PHONE,
    "ComplainantContactNo": CANARY_PHONE,
    "MobileNo": CANARY_PHONE,
    "MobileNoOther": CANARY_PHONE,
    "SHOMobNo": CANARY_PHONE,
    "ReportInform": f"SHO/TEST, Mobile No.:{CANARY_PHONE}",
    "GuardianName": CANARY_PERSON,
    "Parentage": CANARY_PERSON,
    "MotherName": CANARY_PERSON,
    "SpouseName": CANARY_PERSON,
    "Complainant": CANARY_PERSON,
    "EmailId": CANARY_EMAIL,
    "Religion": "CANARYRELIGION",
    "KeptPlaceAddress": CANARY_ADDRESS,
    "Remarks": f"IO CANARY OFFICER {CANARY_PHONE}",
    "VehicleEngineNo": "CANARYENGINE123",
    "VehicleChasisNo": "CANARYCHASSIS456",
    "PostMortemResult": "CANARYPM",
    "HouseNo": CANARY_ADDRESS,
    "RecoveredFrom": CANARY_PERSON,
    "CreatedFromIP": "10.0.0.99",
    "PolicePost": "CANARY POLICE POST",
}
CANARIES = ["9876500001", "CANARY", "canary@", "10.0.0.99"]

KEY_MAP = {
    "missing_persons": "missing_persons",
    "unidentified_dead_bodies": "unidentified_dead_bodies",
    "unidentified_persons": "unidentified_persons",
    "stolen_vehicles": "stolen_vehicles",
    "unclaimed_seized_vehicles": "unclaimed_seized_vehicles",
    "missing_mobiles": "missing_mobiles",
}

ID_FIELD = {
    "missing_persons": "MissingPersonId",
    "unidentified_dead_bodies": "UnIdentifiedDeadBodyId",
    "unidentified_persons": "UnIdentifiedPersonId",
    "stolen_vehicles": "MissingVehiclesId",
    "unclaimed_seized_vehicles": "UnclaimedSeizedVehiclesId",
    "missing_mobiles": "MissingMobileId",
}


def blank_row(category: str) -> dict[str, Any]:
    row = {k: None for k in SOURCE_KEYS[KEY_MAP[category]]["keys"]}
    for k, v in SENSITIVE_FILL.items():
        if k in row:
            row[k] = v
    return row


def make_row(category: str, n: int, **over: Any) -> dict[str, Any]:
    """Row number ``n`` (higher n = newer). Values are synthetic but use real source formats."""
    r = blank_row(category)
    r[ID_FIELD[category]] = 1000 + n
    day = f"{(n % 27) + 1:02d}/10/2026"
    r.update(
        {
            "State": "DELHI",
            "District": "NORTH WEST DISTRICT",
            "PoliceStation": f"TEST PS {n % 3}",
            "CreatedOn": day,
            "UpdatedOn": None,
            "IsDeleted": False,
        }
    )
    if category == "missing_persons":
        r.update(
            {
                "Name": f"TEST PERSON {n}",
                "Sex": "Female" if n % 2 else "Male",
                "BirthYear": "2001",
                "DateFrom": day,
                "ReportingDate": day,
                "DD_Date": day,
                "FIRNo": f"{100 + n}",
                "SerialNumber": f"MP-TEST-{n:04d}",
                "MissingFrom": "near test market",
                "TracingStatus": "Untraced",
                "Height": "157",
                "ImageUrls": [f"/ImageUploads/2026/MissingUploads/{n:08d}-aaaa-bbbb-cccc-000000000000.jpg"],
            }
        )
    elif category in ("unidentified_dead_bodies", "unidentified_persons"):
        serial_key = "UIDBSerialNumber" if category == "unidentified_dead_bodies" else "UIPFSerialNumber"
        r.update(
            {
                serial_key: f"UI-TEST-{n:04d}",
                "FoundDateTime": f"{day} 14:30:00",
                "FoundPlace": "test park gate",
                "Sex": "Male",
                "AgeFrom": "30",
                "AgeTo": "35",
                "DD_Date": day,
                "DDNo": str(n),
                "ImageUrls": [f"/ImageUploads/{n:08d}-aaaa-bbbb-cccc-000000000000.jpg"],
            }
        )
        if category == "unidentified_persons":
            r["Name"] = None
    elif category == "stolen_vehicles":
        r.update(
            {
                "VehicleRegistrationNo": f"DL 3C AB {1000 + n}",
                "VehicleType": "TWO WHEELER",
                "VehicleMake": "TEST MOTORS",
                "VehicleColor": "BLACK",
                "VehicleModel": "2022",
                "Year": "2022",
                "MissingStatus": "Stolen",
                "FIRNo": f"{200 + n}",
                "FIRDate": day,
                "StolenDate": day,
                "StolenFrom": "test parking",
                "CreatedOn": f"{day} 16:52:51",
            }
        )
    elif category == "unclaimed_seized_vehicles":
        r.update(
            {
                "VehicleRegistrationNo": f"HR26DK{2000 + n}",
                "VehicleType": "CAR",
                "VehicleMake": "TEST CARS",
                "VehicleModel": "OTHER",
                "Status": "Seized",
                "DDNo": str(n),
                "DD_Date": day,
            }
        )
    elif category == "missing_mobiles":
        r.update(
            {
                "IMEINo": "490154203237518",
                "IMEINoOther": None,
                "MobileMake": "TestPhone",
                "Status": "Theft/Stolen",
                "FIRNo": f"{300 + n}",
                "FIRDate": day,
                "ReportingDate": day,
                "StolenDate": day,
            }
        )
    r.update(over)
    return r


def page(rows: list[dict[str, Any]], total: int | None = None) -> dict[str, Any]:
    t = len(rows) if total is None else total
    return {"draw": "1", "recordsFiltered": t, "recordsTotal": t, "data": rows}


def listing_html(grid_id: str, script_path: str) -> str:
    """Sanitized skeleton of a ZIPNET listing page (structure as observed; content removed)."""
    return (
        (FIXTURES / "listing_template.html").read_text().replace("{GRID_ID}", grid_id).replace("{SCRIPT}", script_path)
    )
