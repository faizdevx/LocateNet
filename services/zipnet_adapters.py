"""ZIPNET section adapters: deny-by-default field allow-lists.

Each adapter reads ONLY the fields listed in its ``normalize`` method. Everything else in the
source JSON (complainant names and phone numbers, home addresses, guardian/parent/spouse names,
police officers' mobile numbers, e-mail, religion, engine/chassis numbers, the phone numbers of
missing mobiles, shelter addresses, post-mortem results, police remarks, ...) is never read, so it
cannot reach the database, logs, API responses or exports. See docs/ZIPNET_SOURCE_INVENTORY.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import date
from typing import Any, ClassVar

from .zipnet_normalize import (
    clean_free_text,
    clean_text,
    fingerprint,
    normalize_imei,
    normalize_registration,
    to_int,
    validate_image_paths,
)
from .zipnet_time import ParsedDate, parse_source_date


class RowRejected(ValueError):
    """A source row failed validation and must go to the dead-letter log."""


@dataclass
class NormalizedRecord:
    category: str
    source_id: str
    source_id_num: int | None
    title: str
    status: str | None = None
    name: str | None = None
    sex: str | None = None
    age_min: int | None = None
    age_max: int | None = None
    reported_date: date | None = None
    reported_date_kind: str | None = None
    source_created_date: date | None = None
    source_updated_raw: str | None = None
    state: str | None = None
    district: str | None = None
    police_station: str | None = None
    published_location: str | None = None
    case_reference: str | None = None
    vehicle_registration_raw: str | None = None
    vehicle_registration_norm: str | None = None
    vehicle_type: str | None = None
    vehicle_make: str | None = None
    imei: str | None = None
    image_urls: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # Fields compared for change detection (everything except bookkeeping).
    FINGERPRINT_EXCLUDE: ClassVar[set[str]] = {"warnings"}

    def content(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in self.FINGERPRINT_EXCLUDE}

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.content())

    @property
    def extraction_status(self) -> str:
        return "partial" if self.warnings else "ok"


class SourceAdapter:
    """One ZIPNET section. Subclasses declare endpoints and an allow-list normalizer."""

    category: ClassVar[str]
    label: ClassVar[str]
    listing_path: ClassVar[str]
    data_path: ClassVar[str]
    grid_id: ClassVar[str]
    script_path: ClassVar[str]
    id_field: ClassVar[str]
    order_field: ClassVar[str] = "CreatedOn"
    created_field: ClassVar[str] = "CreatedOn"
    # Keys that must be present in every row; a missing key signals a source schema change.
    required_keys: ClassVar[tuple[str, ...]] = ()
    has_images: ClassVar[bool] = False

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    @property
    def listing_url(self) -> str:
        return self.base_url + self.listing_path

    @property
    def data_url(self) -> str:
        return self.base_url + self.data_path

    def form_data(self, start: int, length: int, search: str = "") -> dict[str, str]:
        """The same DataTables parameters the public page sends."""
        return {
            "draw": "1",
            "start": str(start),
            "length": str(length),
            "search[value]": search,
            "search[regex]": "true" if search else "false",
            "order[0][column]": "1",
            "order[0][dir]": "desc",
            "columns[0][data]": self.id_field,
            "columns[0][name]": self.id_field,
            "columns[1][data]": self.order_field,
            "columns[1][name]": self.order_field,
            "searchFormJson": "{}",
        }

    # -- helpers for subclasses -------------------------------------------------------

    def _base(self, row: dict[str, Any]) -> tuple[str, int | None, list[str]]:
        raw_id = row.get(self.id_field)
        sid = clean_text(raw_id, 40, keep_zero=True)
        if sid is None:
            raise RowRejected(f"missing {self.id_field}")
        if not sid.isdigit():
            raise RowRejected(f"non-numeric {self.id_field}")
        return sid, int(sid), []

    @staticmethod
    def _date(row: dict[str, Any], key: str, warnings: list[str]) -> ParsedDate:
        pd = parse_source_date(row.get(key))
        if not pd.ok:
            warnings.append(f"unparsed_date:{key}")
        return pd

    def _images(self, row: dict[str, Any], warnings: list[str]) -> list[str]:
        paths, w = validate_image_paths(row.get("ImageUrls"))
        warnings.extend(w)
        return paths

    def missing_required_keys(self, row: dict[str, Any]) -> list[str]:
        return [k for k in (self.id_field, *self.required_keys) if k not in row]

    def normalize(self, row: dict[str, Any]) -> NormalizedRecord:  # pragma: no cover - abstract
        raise NotImplementedError




def _area(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": clean_text(row.get("State"), 100),
        "district": clean_text(row.get("District"), 150),
        "police_station": clean_text(row.get("PoliceStation"), 150),
    }


def _join_ref(*parts: tuple[str, Any]) -> str | None:
    bits = [f"{label} {clean_text(v, 60)}" for label, v in parts if clean_text(v, 60)]
    return " · ".join(bits) or None


def _description(row: dict[str, Any], keys: tuple[str, ...]) -> dict[str, str]:
    out = {}
    for k in keys:
        v = clean_free_text(row.get(k), 120)
        if v:
            out[k] = v
    return out


_PHYSICAL = (
    "Height",
    "Build",
    "Complexion",
    "Face",
    "Eyes",
    "Hair",
    "DressUpper",
    "DressUpperColor",
    "DressLower",
    "DressLowerColor",
)

_BOOL_TRUE = {"true", "yes", "y", "1"}
_BOOL_FALSE = {"false", "no", "n", "0"}


def _matched_status(value: Any) -> str | None:
    if isinstance(value, bool):
        return "Matched" if value else "Not matched"
    s = clean_text(value, 40, keep_zero=True)
    if s is None:
        return None
    if s.lower() in _BOOL_TRUE:
        return "Matched"
    if s.lower() in _BOOL_FALSE:
        return "Not matched"
    return s


class MissingPersonsAdapter(SourceAdapter):
    category = "missing_persons"
    label = "Missing Persons"
    listing_path = "/Victims/MissingPersons"
    data_path = "/Victims/GetMissingPersonsData/"
    grid_id = "missingPersonGrid"
    script_path = "/Scripts/app/MissingPersons.js"
    id_field = "MissingPersonId"
    required_keys = ("State", "District", "PoliceStation", "TracingStatus", "CreatedOn", "ImageUrls")
    has_images = True

    def normalize(self, row: dict[str, Any]) -> NormalizedRecord:
        sid, sid_num, w = self._base(row)
        created = self._date(row, "CreatedOn", w)
        missing_since = self._date(row, "DateFrom", w)
        dd_date = self._date(row, "DD_Date", w)
        reporting = self._date(row, "ReportingDate", w)
        tracing = self._date(row, "TracingDate", w)
        name = clean_text(row.get("Name"), 200)
        birth_year = to_int(row.get("BirthYear"), 1900, 2100)
        attrs: dict[str, Any] = {
            "fir_no": clean_text(row.get("FIRNo"), 60),
            "dd_no": clean_text(row.get("DDNo"), 60),
            "dd_date": dd_date.date,
            "serial_number": clean_text(row.get("SerialNumber"), 60),
            "reporting_date": reporting.date,
            "missing_since": missing_since.date,
            "tracing_date": tracing.date,
            "birth_year_reported": birth_year,
            "description": _description(row, _PHYSICAL),
        }
        return NormalizedRecord(
            category=self.category,
            source_id=sid,
            source_id_num=sid_num,
            title=name or "Name not published",
            status=clean_text(row.get("TracingStatus"), 80),
            name=name,
            sex=clean_text(row.get("Sex"), 20),
            reported_date=missing_since.date or reporting.date,
            reported_date_kind="missing_since" if missing_since.date else ("reported" if reporting.date else None),
            source_created_date=created.date,
            source_updated_raw=clean_text(row.get("UpdatedOn"), 40),
            published_location=clean_free_text(row.get("MissingFrom"), 300),
            case_reference=clean_text(row.get("SerialNumber"), 60) or _join_ref(("FIR", row.get("FIRNo"))),
            image_urls=self._images(row, w),
            attributes=attrs,
            warnings=w,
            **_area(row),
        )


class _UnidentifiedBase(SourceAdapter):
    serial_field: str = ""
    has_images = True

    def _common(self, row: dict[str, Any]) -> tuple[str, int | None, list[str], dict[str, Any], Any, Any]:
        sid, sid_num, w = self._base(row)
        created = self._date(row, "CreatedOn", w)
        found = self._date(row, "FoundDateTime", w)
        dd_date = self._date(row, "DD_Date", w)
        matching = self._date(row, "MatchingDate", w)
        attrs: dict[str, Any] = {
            "dd_no": clean_text(row.get("DDNo"), 60),
            "dd_date": dd_date.date,
            "serial_number": clean_text(row.get(self.serial_field), 60),
            "found_at_utc": found.at_utc,
            "matched_raw": clean_text(row.get("Matched"), 40, keep_zero=True),
            "matching_date": matching.date,
            "description": _description(row, _PHYSICAL),
        }
        return sid, sid_num, w, attrs, created, found


class UnidentifiedDeadBodiesAdapter(_UnidentifiedBase):
    category = "unidentified_dead_bodies"
    label = "Un-Identified Dead Bodies"
    listing_path = "/Victims/UnIdentifiedDeadBodies"
    data_path = "/Victims/GetUnIdentifiedDeadBodiesData/"
    grid_id = "unidentifiedDeadBodiesGrid"
    script_path = "/Scripts/app/UnIdentifiedDeadBodies.js"
    id_field = "UnIdentifiedDeadBodyId"
    serial_field = "UIDBSerialNumber"
    required_keys = ("State", "District", "PoliceStation", "UIDBSerialNumber", "CreatedOn", "ImageUrls")

    def normalize(self, row: dict[str, Any]) -> NormalizedRecord:
        sid, sid_num, w, attrs, created, found = self._common(row)
        attrs["fir_no"] = clean_text(row.get("FIRNo"), 60)
        serial = attrs["serial_number"]
        return NormalizedRecord(
            category=self.category,
            source_id=sid,
            source_id_num=sid_num,
            title=f"Unidentified body · {serial}" if serial else "Unidentified body",
            status=_matched_status(row.get("Matched")),
            sex=clean_text(row.get("Sex"), 20),
            age_min=to_int(row.get("AgeFrom"), 0, 120),
            age_max=to_int(row.get("AgeTo"), 0, 120),
            reported_date=found.date,
            reported_date_kind="found" if found.date else None,
            source_created_date=created.date,
            source_updated_raw=clean_text(row.get("UpdatedOn"), 40),
            published_location=clean_free_text(row.get("FoundPlace"), 300),
            case_reference=serial or _join_ref(("DD", row.get("DDNo"))),
            image_urls=self._images(row, w),
            attributes=attrs,
            warnings=w,
            **_area(row),
        )


class UnidentifiedPersonsAdapter(_UnidentifiedBase):
    category = "unidentified_persons"
    label = "Un-Identified Persons Found"
    listing_path = "/Victims/UnIdentifiedPersons"
    data_path = "/Victims/GetUnIdentifiedPersonsData/"
    grid_id = "unidentifiedPersonsGrid"
    script_path = "/Scripts/app/UnIdentifiedPersons.js"
    id_field = "UnIdentifiedPersonId"
    serial_field = "UIPFSerialNumber"
    required_keys = ("State", "District", "PoliceStation", "UIPFSerialNumber", "CreatedOn", "ImageUrls")

    def normalize(self, row: dict[str, Any]) -> NormalizedRecord:
        sid, sid_num, w, attrs, created, found = self._common(row)
        name = clean_text(row.get("Name"), 200)
        serial = attrs["serial_number"]
        return NormalizedRecord(
            category=self.category,
            source_id=sid,
            source_id_num=sid_num,
            title=name or (f"Unidentified person · {serial}" if serial else "Unidentified person"),
            status=clean_text(row.get("FoundStatus"), 80) or _matched_status(row.get("Matched")),
            name=name,
            sex=clean_text(row.get("Sex"), 20),
            age_min=to_int(row.get("AgeFrom"), 0, 120),
            age_max=to_int(row.get("AgeTo"), 0, 120),
            reported_date=found.date,
            reported_date_kind="found" if found.date else None,
            source_created_date=created.date,
            source_updated_raw=clean_text(row.get("UpdatedOn"), 40),
            published_location=clean_free_text(row.get("FoundPlace"), 300),
            case_reference=serial or _join_ref(("DD", row.get("DDNo"))),
            image_urls=self._images(row, w),
            attributes=attrs,
            warnings=w,
            **_area(row),
        )


class _VehicleBase(SourceAdapter):
    status_field = "Status"

    def _vehicle(self, row: dict[str, Any], w: list[str]) -> dict[str, Any]:
        raw, norm, valid = normalize_registration(row.get("VehicleRegistrationNo"))
        if raw and not valid:
            w.append("registration_nonstandard_format")
        return {
            "vehicle_registration_raw": raw,
            "vehicle_registration_norm": norm,
            "vehicle_type": clean_text(row.get("VehicleType"), 80),
            "vehicle_make": clean_text(row.get("VehicleMake"), 150),
            "_valid": valid,
        }


class StolenVehiclesAdapter(_VehicleBase):
    category = "stolen_vehicles"
    label = "Stolen Vehicles"
    listing_path = "/VehiclesMobiles/MissingVehicles"
    data_path = "/VehiclesMobiles/GetMissingVehiclesData/"
    grid_id = "missingVehiclesGrid"
    script_path = "/Scripts/app/MissingVehicles.js"
    id_field = "MissingVehiclesId"
    order_field = "FIRDate"
    required_keys = ("State", "District", "PoliceStation", "VehicleRegistrationNo", "MissingStatus", "CreatedOn")

    def normalize(self, row: dict[str, Any]) -> NormalizedRecord:
        sid, sid_num, w = self._base(row)
        created = self._date(row, "CreatedOn", w)
        fir = self._date(row, "FIRDate", w)
        stolen = self._date(row, "StolenDate", w)
        report = self._date(row, "ReportDateTime", w)
        v = self._vehicle(row, w)
        valid = v.pop("_valid")
        attrs = {
            "fir_no": clean_text(row.get("FIRNo"), 60),
            "fir_date": fir.date,
            "report_at_utc": report.at_utc,
            "stolen_date": stolen.date,
            "vehicle_color": clean_text(row.get("VehicleColor"), 60),
            "vehicle_model": clean_text(row.get("VehicleModel"), 80),
            "vehicle_year": to_int(row.get("Year"), 1900, 2100),
            "registration_valid_format": valid,
        }
        reg = v["vehicle_registration_raw"]
        return NormalizedRecord(
            category=self.category,
            source_id=sid,
            source_id_num=sid_num,
            title=" · ".join(x for x in (v["vehicle_make"], reg) if x) or "Vehicle (details not published)",
            status=clean_text(row.get("MissingStatus"), 80),
            reported_date=stolen.date or fir.date,
            reported_date_kind="stolen" if stolen.date else ("fir" if fir.date else None),
            source_created_date=created.date,
            source_updated_raw=clean_text(row.get("UpdatedOn"), 40),
            published_location=clean_free_text(row.get("StolenFrom"), 300),
            case_reference=_join_ref(("FIR", row.get("FIRNo"))),
            attributes=attrs,
            warnings=w,
            **v,
            **_area(row),
        )


class UnclaimedSeizedVehiclesAdapter(_VehicleBase):
    category = "unclaimed_seized_vehicles"
    label = "Un-claimed / Seized Vehicles"
    listing_path = "/VehiclesMobiles/UnclaimedSeizedVehicles"
    data_path = "/VehiclesMobiles/GetUnclaimedSeizedVehiclesData/"
    grid_id = "unclaimedSeizedVehiclesGrid"
    script_path = "/Scripts/app/UnclaimedSeizedVehicles.js"
    id_field = "UnclaimedSeizedVehiclesId"
    required_keys = ("State", "District", "PoliceStation", "VehicleRegistrationNo", "Status", "CreatedOn")

    def normalize(self, row: dict[str, Any]) -> NormalizedRecord:
        sid, sid_num, w = self._base(row)
        created = self._date(row, "CreatedOn", w)
        dd = self._date(row, "DD_Date", w)
        reporting = self._date(row, "ReportingDate", w)
        v = self._vehicle(row, w)
        valid = v.pop("_valid")
        attrs = {
            "dd_no": clean_text(row.get("DDNo"), 60),
            "dd_date": dd.date,
            "reporting_date": reporting.date,
            "vehicle_color": clean_text(row.get("VehicleColor"), 60),
            "vehicle_model": clean_text(row.get("VehicleModel"), 80),
            "vehicle_year": to_int(row.get("Year"), 1900, 2100),
            "registration_valid_format": valid,
        }
        reg = v["vehicle_registration_raw"]
        return NormalizedRecord(
            category=self.category,
            source_id=sid,
            source_id_num=sid_num,
            title=" · ".join(x for x in (v["vehicle_make"] or v["vehicle_type"], reg) if x)
            or "Vehicle (details not published)",
            status=clean_text(row.get("Status"), 80),
            reported_date=dd.date or reporting.date,
            reported_date_kind="dd_entry" if dd.date else ("reported" if reporting.date else None),
            source_created_date=created.date,
            source_updated_raw=clean_text(row.get("UpdatedOn"), 40),
            case_reference=_join_ref(("DD", row.get("DDNo"))),
            attributes=attrs,
            warnings=w,
            **v,
            **_area(row),
        )


class MissingMobilesAdapter(SourceAdapter):
    category = "missing_mobiles"
    label = "Missing Mobiles"
    listing_path = "/VehiclesMobiles/MissingMobiles"
    data_path = "/VehiclesMobiles/GetMissingMobilesData/"
    grid_id = "missingMobilesGrid"
    script_path = "/Scripts/app/MissingMobiles.js"
    id_field = "MissingMobileId"
    required_keys = ("State", "District", "PoliceStation", "IMEINo", "Status", "CreatedOn")

    def normalize(self, row: dict[str, Any]) -> NormalizedRecord:
        sid, sid_num, w = self._base(row)
        created = self._date(row, "CreatedOn", w)
        fir = self._date(row, "FIRDate", w)
        stolen = self._date(row, "StolenDate", w)
        reporting = self._date(row, "ReportingDate", w)
        imei, imei_ok = normalize_imei(row.get("IMEINo"))
        imei2, imei2_ok = normalize_imei(row.get("IMEINoOther"))
        if imei and not imei_ok:
            w.append("imei_failed_checksum_or_length")
        make = clean_text(row.get("MobileMake"), 100)
        attrs = {
            "fir_no": clean_text(row.get("FIRNo"), 60),
            "fir_date": fir.date,
            "stolen_date": stolen.date,
            "reporting_date": reporting.date,
            "imei_other": imei2,
            "imei_valid": imei_ok,
            "imei_other_valid": imei2_ok if imei2 else None,
            "mobile_type": clean_text(row.get("MobileType"), 60),
        }
        return NormalizedRecord(
            category=self.category,
            source_id=sid,
            source_id_num=sid_num,
            title=" · ".join(x for x in (make, f"IMEI {imei}" if imei else None) if x) or "Mobile phone",
            status=clean_text(row.get("Status"), 80),
            reported_date=stolen.date or fir.date,
            reported_date_kind="stolen" if stolen.date else ("fir" if fir.date else None),
            source_created_date=created.date,
            source_updated_raw=clean_text(row.get("UpdatedOn"), 40),
            case_reference=_join_ref(("FIR", row.get("FIRNo"))),
            vehicle_make=None,
            imei=imei,
            attributes={**attrs, "mobile_make": make},
            warnings=w,
            **_area(row),
        )


ADAPTER_CLASSES: dict[str, type[SourceAdapter]] = {
    a.category: a
    for a in (
        MissingPersonsAdapter,
        UnidentifiedDeadBodiesAdapter,
        UnidentifiedPersonsAdapter,
        StolenVehiclesAdapter,
        UnclaimedSeizedVehiclesAdapter,
        MissingMobilesAdapter,
    )
}
ADAPTERS = ADAPTER_CLASSES  # public alias


def get_adapter(category: str, base_url: str) -> SourceAdapter:
    try:
        return ADAPTER_CLASSES[category](base_url)
    except KeyError as e:
        raise ValueError(f"unknown category: {category}") from e
