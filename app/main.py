"""Employee attendance and analytics API.

Run with ``uvicorn app.main:app --port 8000``. MongoDB settings are read from
MONGO_URI and MONGO_DB (a local .env file is loaded for convenience).
"""
import json
import os
import re
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Annotated, Literal

from bson import Decimal128, json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError
from pymongo.collection import ReturnDocument

load_dotenv()

client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc
PRESENCE_STATUSES = ("PRESENT", "WFH", "ON_DUTY")
ALL_STATUSES = ("PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE")
EPOCH_MIN = 100_000_000_000
EPOCH_MAX = 4_102_444_800_000
EpochMillis = Annotated[StrictInt, Field(ge=EPOCH_MIN, le=EPOCH_MAX)]

app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0")


@app.on_event("startup")
def create_indexes() -> None:
    db.employees.create_index([("emp_code", ASCENDING)], unique=True, name="uq_employee_code")
    db.employees.create_index(
        [("department", ASCENDING), ("emp_code", ASCENDING)],
        name="ix_employee_department_code",
    )
    db.employees.create_index(
        [("department", ASCENDING), ("joined_on", ASCENDING)],
        name="ix_employee_department_joined",
    )
    db.employees.create_index([("joined_on", ASCENDING)], name="ix_employee_joined")

    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)],
        unique=True,
        name="uq_attendance_employee_date",
    )
    db.attendance_logs.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)],
        name="ix_attendance_date_code",
    )
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("punch_in", DESCENDING)],
        name="ix_attendance_employee_punch_in",
    )


# --------------------------------------------------------------------------- #
# Validation and representation helpers
# --------------------------------------------------------------------------- #
def require_date(value: str, label: str = "date") -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        raise HTTPException(422, f"{label} must be a valid YYYY-MM-DD date")
    if parsed.isoformat() != value:
        raise HTTPException(422, f"{label} must be a valid YYYY-MM-DD date")
    return parsed


def require_month(value: str) -> tuple[date, date]:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        raise HTTPException(422, "month must be in YYYY-MM format")
    year, month = map(int, value.split("-"))
    try:
        start = date(year, month, 1)
        end = date(year + (month == 12), 1 if month == 12 else month + 1, 1) - timedelta(days=1)
    except ValueError:
        raise HTTPException(422, "month must be a valid calendar month")
    return start, end


def require_shift(value: str) -> str:
    if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", value):
        raise HTTPException(422, "shift times must be in HH:MM format")
    return value


def truncate_second(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).replace(microsecond=0)


def from_epoch_millis(value: int | None) -> datetime:
    if value is None:
        return datetime.now(UTC).replace(microsecond=0)
    return datetime.fromtimestamp(value / 1000, tz=UTC).replace(microsecond=0)


def to_epoch_millis(value: datetime | None) -> int | None:
    if value is None:
        return None
    value = truncate_second(value)
    delta = value - datetime(1970, 1, 1, tzinfo=UTC)
    return delta.days * 86_400_000 + delta.seconds * 1_000


def working_days(start: date, end: date) -> int:
    if end < start:
        return 0
    complete_weeks, extra_days = divmod((end - start).days + 1, 7)
    count = complete_weeks * 5
    for offset in range(extra_days):
        if (start.weekday() + offset) % 7 < 5:
            count += 1
    return count


def round_half_up(value: float | Decimal, places: int) -> float:
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def _time_parts(value: str) -> tuple[int, int]:
    hour, minute = map(int, value.split(":"))
    return hour, minute


def attendance_date(punch_in: datetime, employee: dict) -> date:
    local = truncate_second(punch_in).astimezone(IST)
    shift_start = require_shift(employee["shift_start"])
    shift_end = require_shift(employee["shift_end"])
    if shift_end <= shift_start:
        end_hour, end_minute = _time_parts(shift_end)
        if (local.hour, local.minute) < (end_hour, end_minute):
            return local.date() - timedelta(days=1)
    return local.date()


def compute_late_minutes(punch_in: datetime, shift_start: str, shift_end: str) -> int:
    local = truncate_second(punch_in).astimezone(IST)
    attendance_day = attendance_date(
        punch_in, {"shift_start": shift_start, "shift_end": shift_end}
    )
    start_hour, start_minute = _time_parts(shift_start)
    shift_start_at = datetime.combine(
        attendance_day, time(start_hour, start_minute), tzinfo=IST
    )
    elapsed = int((local - shift_start_at).total_seconds())
    return elapsed // 60 if elapsed > 600 else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    seconds = int((truncate_second(punch_out) - truncate_second(punch_in)).total_seconds())
    return round_half_up(Decimal(seconds) / Decimal(3600), 2)


def compute_overtime(
    punch_out: datetime,
    shift_end: str,
    shift_start: str,
    attendance_day: date,
) -> int:
    end_hour, end_minute = _time_parts(shift_end)
    end_day = attendance_day
    if shift_end <= shift_start:
        end_day += timedelta(days=1)
    shift_end_at = datetime.combine(end_day, time(end_hour, end_minute), tzinfo=IST)
    elapsed = int((truncate_second(punch_out).astimezone(IST) - shift_end_at).total_seconds())
    return elapsed // 60 if elapsed >= 1_800 else 0


def serialize_attendance(doc: dict) -> dict:
    result = {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": to_epoch_millis(doc.get("punch_in")),
        "punch_out": to_epoch_millis(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": doc.get("half_day", False),
        "history": [],
    }
    for entry in doc.get("history", []):
        changes = {}
        for name, change in entry.get("changes", {}).items():
            if name in ("punch_in", "punch_out"):
                changes[name] = {
                    "from": to_epoch_millis(change.get("from")),
                    "to": to_epoch_millis(change.get("to")),
                }
            else:
                changes[name] = change
        result["history"].append(
            {
                "at": to_epoch_millis(entry["at"]),
                "by": entry["by"],
                "reason": entry["reason"],
                "changes": changes,
            }
        )
    return result


def date_range_query(start: date, end: date) -> dict:
    return {"$gte": start.isoformat(), "$lte": end.isoformat()}


def _mongo_round_half_up(expression: object, places: int) -> dict:
    scale = Decimal128(str(10**places))
    half = Decimal128("0.5")
    return {
        "$toDouble": {
            "$divide": [
                {
                    "$floor": {
                        "$add": [
                            {
                                "$multiply": [
                                    {"$toDecimal": {"$toString": expression}},
                                    scale,
                                ]
                            },
                            half,
                        ]
                    }
                },
                scale,
            ]
        }
    }


def _presence_day_value() -> dict:
    return {
        "$cond": [
            {
                "$and": [
                    {"$in": ["$status", list(PRESENCE_STATUSES)]},
                    {"$in": [{"$dayOfWeek": "$_day"}, [2, 3, 4, 5, 6]]},
                ]
            },
            {"$cond": [{"$ifNull": ["$half_day", False]}, 0.5, 1]},
            0,
        ]
    }


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class EmployeeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field(default="18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def validate_joined_on(cls, value: str) -> str:
        require_date(value, "joined_on")
        return value

    @model_validator(mode="after")
    def shifts_must_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str
    punched_at: EpochMillis | None = None
    status: Literal["PRESENT", "WFH", "ON_DUTY"] = "PRESENT"


class PunchOutIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str
    punched_at: EpochMillis | None = None


class RegularizeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE"] | None = None
    punch_in: EpochMillis | None = None
    punch_out: EpochMillis | None = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def reject_explicit_nulls(self):
        for name in ("status", "punch_in", "punch_out"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"{name} may not be null")
        return self


# --------------------------------------------------------------------------- #
# Employee and attendance endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    try:
        db.command("ping")
    except PyMongoError:
        raise HTTPException(503, "database is unavailable")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn) -> dict:
    if body.shift_start == body.shift_end:
        raise HTTPException(422, "shift_start must differ from shift_end")
    doc = body.model_dump()
    doc["created_at"] = datetime.now(UTC).replace(microsecond=0)
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    doc.pop("_id", None)
    doc["created_at"] = to_epoch_millis(doc["created_at"])
    return doc


@app.get("/employees")
def list_employees(
    department: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict:
    query = {"department": department} if department is not None else {}
    total = db.employees.count_documents(query)
    cursor = (
        db.employees.find(query, {"_id": 0})
        .sort("emp_code", ASCENDING)
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    items = list(cursor)
    for item in items:
        item["created_at"] = to_epoch_millis(item["created_at"])
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn) -> dict:
    employee = db.employees.find_one({"emp_code": body.emp_code})
    if employee is None:
        raise HTTPException(404, "employee not found")

    timestamp = from_epoch_millis(body.punched_at)
    day = attendance_date(timestamp, employee).isoformat()
    doc = {
        "emp_code": body.emp_code,
        "date": day,
        "status": body.status,
        "punch_in": timestamp,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": compute_late_minutes(
            timestamp, employee["shift_start"], employee["shift_end"]
        ),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        result = db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    doc["_id"] = result.inserted_id
    return serialize_attendance(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn) -> dict:
    employee = db.employees.find_one({"emp_code": body.emp_code})
    if employee is None:
        raise HTTPException(404, "employee not found")

    timestamp = from_epoch_millis(body.punched_at)
    record = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": timestamp}},
        sort=[("punch_in", DESCENDING)],
    )
    if record is None:
        raise HTTPException(404, "no punch-in found")
    if record.get("punch_out") is not None:
        raise HTTPException(409, "record is already punched out")
    punch_in_at = truncate_second(record["punch_in"])
    seconds = int((timestamp - punch_in_at).total_seconds())
    if seconds <= 0 or seconds > 86_400:
        raise HTTPException(422, "punch_out must be after punch_in and within 24 hours")

    record_day = require_date(record["date"])
    work_hours = compute_work_hours(punch_in_at, timestamp)
    overtime = compute_overtime(
        timestamp,
        employee["shift_end"],
        employee["shift_start"],
        record_day,
    )
    update = db.attendance_logs.find_one_and_update(
        {
            "_id": record["_id"],
            "punch_out": None,
        },
        {
            "$set": {
                "punch_out": timestamp,
                "work_hours": work_hours,
                "overtime_minutes": overtime,
                "half_day": work_hours < 4.5,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if update is None:
        raise HTTPException(409, "record is already punched out")
    return serialize_attendance(update)


@app.get("/attendance")
def list_attendance(
    emp_code: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    status: Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"] | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict:
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from must be on or before date_to")
    query: dict = {}
    if emp_code is not None:
        query["emp_code"] = emp_code
    if date_from or date_to:
        query["date"] = {}
        if date_from:
            query["date"]["$gte"] = date_from.isoformat()
        if date_to:
            query["date"]["$lte"] = date_to.isoformat()
    if status is not None:
        query["status"] = status

    total = db.attendance_logs.count_documents(query)
    attendance_hint = (
        "uq_attendance_employee_date"
        if emp_code is not None
        else "ix_attendance_date_code"
    )
    docs = db.attendance_logs.find(query).hint(attendance_hint).sort(
        [("date", DESCENDING), ("emp_code", ASCENDING)]
    ).skip((page - 1) * page_size).limit(page_size)
    return {
        "items": [serialize_attendance(doc) for doc in docs],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def _patch_snapshot_filter(record: dict) -> dict:
    conditions = [{"_id": record["_id"]}]
    for field in ("status", "punch_in", "punch_out", "history"):
        if field in record:
            conditions.append({field: record[field]})
        else:
            conditions.append({field: {"$exists": False}})
    return {"$and": conditions}


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(
    emp_code: str, date: date, body: RegularizeIn
) -> dict:
    day = date.isoformat()
    employee = db.employees.find_one({"emp_code": emp_code})
    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": day})
    if employee is None or record is None:
        raise HTTPException(404, "employee or attendance record not found")

    changes_requested = body.model_fields_set
    old_status = record["status"]
    old_in = record.get("punch_in")
    old_out = record.get("punch_out")
    status = body.status if "status" in changes_requested else old_status
    if status in ("ABSENT", "LEAVE"):
        if "punch_in" in changes_requested or "punch_out" in changes_requested:
            raise HTTPException(422, "ABSENT and LEAVE cannot include punch times")
        new_in = new_out = None
    else:
        new_in = from_epoch_millis(body.punch_in) if "punch_in" in changes_requested else old_in
        new_out = from_epoch_millis(body.punch_out) if "punch_out" in changes_requested else old_out
        if new_in is None:
            raise HTTPException(422, "presence statuses require punch_in")
        if attendance_date(new_in, employee).isoformat() != day:
            raise HTTPException(422, "punch_in must belong to the attendance date")
        if new_out is not None:
            duration = int((new_out - new_in).total_seconds())
            if duration <= 0 or duration > 86_400:
                raise HTTPException(422, "punch_out must be after punch_in and within 24 hours")

    late = 0
    hours = None
    overtime = 0
    half_day = False
    if status in PRESENCE_STATUSES:
        late = compute_late_minutes(new_in, employee["shift_start"], employee["shift_end"])
        if new_out is not None:
            hours = compute_work_hours(new_in, new_out)
            overtime = compute_overtime(
                new_out,
                employee["shift_end"],
                employee["shift_start"],
                date,
            )
            half_day = hours < 4.5

    final = {
        "status": status,
        "punch_in": new_in,
        "punch_out": new_out,
        "work_hours": hours,
        "late_minutes": late,
        "overtime_minutes": overtime,
        "half_day": half_day,
    }
    before = {
        "status": old_status,
        "punch_in": old_in,
        "punch_out": old_out,
        "work_hours": record.get("work_hours"),
        "late_minutes": record.get("late_minutes", 0),
        "overtime_minutes": record.get("overtime_minutes", 0),
        "half_day": record.get("half_day", False),
    }
    changed = {}
    for name, value in final.items():
        prior = before[name]
        if isinstance(prior, datetime) or isinstance(value, datetime):
            prior_compare = truncate_second(prior) if prior is not None else None
            value_compare = truncate_second(value) if value is not None else None
        else:
            prior_compare, value_compare = prior, value
        if prior_compare != value_compare:
            changed[name] = {"from": prior, "to": value}
    if not changed:
        raise HTTPException(422, "regularization does not change the record")

    audit = {
        "at": datetime.now(UTC).replace(microsecond=0),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changed,
    }
    updated = db.attendance_logs.find_one_and_update(
        _patch_snapshot_filter(record),
        {"$set": final, "$push": {"history": audit}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "attendance record changed concurrently")
    return serialize_attendance(updated)


# --------------------------------------------------------------------------- #
# Analytics pipeline builders
# --------------------------------------------------------------------------- #
def employee_monthly_pipeline(emp_code: str, start: date, end: date, joined_on: date) -> list[dict]:
    return [
        {"$match": {"emp_code": emp_code, "date": date_range_query(start, end)}},
        {
            "$set": {
                "_day": {"$dateFromString": {"dateString": "$date"}},
                "_late": {"$ifNull": ["$late_minutes", 0]},
                "_overtime": {"$ifNull": ["$overtime_minutes", 0]},
            }
        },
        {
            "$group": {
                "_id": None,
                "present_days": {"$sum": _presence_day_value()},
                "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                "late_count": {"$sum": {"$cond": [{"$gt": ["$_late", 0]}, 1, 0]}},
                "total_late_minutes": {"$sum": "$_late"},
                "total_overtime_minutes": {"$sum": "$_overtime"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "present_days": 1,
                "leave_days": 1,
                "late_count": 1,
                "total_late_minutes": 1,
                "total_overtime_minutes": 1,
            }
        },
    ]


def department_summary_pipeline(start: date, end: date, department: str | None) -> list[dict]:
    match: dict = {"joined_on": {"$lte": end.isoformat()}}
    if department is not None:
        match["department"] = department
    return [
        {"$match": match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"code": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$code"]},
                                    {"$gte": ["$date", start.isoformat()]},
                                    {"$lte": ["$date", end.isoformat()]},
                                ]
                            }
                        }
                    },
                    {
                        "$set": {
                            "_day": {"$dateFromString": {"dateString": "$date"}},
                            "_late": {"$ifNull": ["$late_minutes", 0]},
                            "_overtime": {"$ifNull": ["$overtime_minutes", 0]},
                        }
                    },
                ],
                "as": "_logs",
            }
        },
        {"$unwind": {"path": "$_logs", "preserveNullAndEmptyArrays": True}},
        {
            "$group": {
                "_id": {"department": "$department", "emp_code": "$emp_code"},
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$ne": [
                                    {"$ifNull": ["$_logs.date", None]},
                                    None,
                                ]
                            },
                            {
                                "$let": {
                                    "vars": {
                                        "status": "$_logs.status",
                                        "half_day": {"$ifNull": ["$_logs.half_day", False]},
                                        "weekday": {"$dayOfWeek": "$_logs._day"},
                                    },
                                    "in": {
                                        "$cond": [
                                            {
                                                "$and": [
                                                    {"$in": ["$$status", list(PRESENCE_STATUSES)]},
                                                    {"$in": ["$$weekday", [2, 3, 4, 5, 6]]},
                                                ]
                                            },
                                            {"$cond": ["$$half_day", 0.5, 1]},
                                            0,
                                        ]
                                    },
                                }
                            },
                            0,
                        ]
                    }
                },
                "hours_sum": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$_logs.status", list(PRESENCE_STATUSES)]},
                                    {
                                        "$ne": [
                                            {"$ifNull": ["$_logs.work_hours", None]},
                                            None,
                                        ]
                                    },
                                ]
                            },
                            "$_logs.work_hours",
                            0,
                        ]
                    }
                },
                "hours_count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$_logs.status", list(PRESENCE_STATUSES)]},
                                    {
                                        "$ne": [
                                            {"$ifNull": ["$_logs.work_hours", None]},
                                            None,
                                        ]
                                    },
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$_logs._late", 0]}, 1, 0]}
                },
                "total_late_minutes": {"$sum": {"$ifNull": ["$_logs._late", 0]}},
                "leave_count": {
                    "$sum": {"$cond": [{"$eq": ["$_logs.status", "LEAVE"]}, 1, 0]}
                },
                "on_duty_count": {
                    "$sum": {"$cond": [{"$eq": ["$_logs.status", "ON_DUTY"]}, 1, 0]}
                },
            }
        },
        {
            "$group": {
                "_id": "$_id.department",
                "headcount": {"$sum": 1},
                "present_days": {"$sum": "$present_days"},
                "hours_sum": {"$sum": "$hours_sum"},
                "hours_count": {"$sum": "$hours_count"},
                "late_count": {"$sum": "$late_count"},
                "total_late_minutes": {"$sum": "$total_late_minutes"},
                "leave_count": {"$sum": "$leave_count"},
                "on_duty_count": {"$sum": "$on_duty_count"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": 1,
                "present_days": 1,
                "avg_work_hours": {
                    "$cond": [
                        {"$gt": ["$hours_count", 0]},
                        _mongo_round_half_up(
                            {"$divide": ["$hours_sum", "$hours_count"]}, 2
                        ),
                        None,
                    ]
                },
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
        {"$sort": {"department": 1}},
    ]


def late_leaderboard_pipeline(
    start: date, end: date, limit: int, department: str | None
) -> list[dict]:
    pipeline = [
        {"$match": {"date": date_range_query(start, end)}},
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "late_count": {
                    "$sum": {
                        "$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]
                    }
                },
            }
        },
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "_employee",
            }
        },
        {"$unwind": "$_employee"},
    ]
    if department is not None:
        pipeline.append({"$match": {"_employee.department": department}})
    pipeline.extend(
        [
            {
                "$setWindowFields": {
                    "sortBy": {"total_late_minutes": -1},
                    "output": {"rank": {"$rank": {}}},
                }
            },
            {"$match": {"rank": {"$lte": limit}}},
            {"$sort": {"total_late_minutes": -1, "_id": 1}},
            {
                "$project": {
                    "_id": 0,
                    "rank": 1,
                    "emp_code": "$_id",
                    "name": "$_employee.name",
                    "department": "$_employee.department",
                    "total_late_minutes": 1,
                    "late_count": 1,
                }
            },
        ]
    )
    return pipeline


def department_trend_pipeline(
    department: str, start: date, end: date
) -> list[dict]:
    start_dt = datetime.combine(start, time.min, tzinfo=UTC)
    end_exclusive = datetime.combine(end + timedelta(days=1), time.min, tzinfo=UTC)
    return [
        {"$match": {"date": date_range_query(start, end)}},
        {
            "$lookup": {
                "from": "employees",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "as": "_employee",
            }
        },
        {"$unwind": "$_employee"},
        {"$match": {"_employee.department": department}},
        {
            "$set": {
                "_day": {"$dateFromString": {"dateString": "$date"}},
                "_present": {
                    "$cond": [
                        {"$in": ["$status", list(PRESENCE_STATUSES)]},
                        {"$cond": [{"$ifNull": ["$half_day", False]}, 0.5, 1]},
                        0,
                    ]
                },
                "_late": {
                    "$cond": [
                        {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]},
                        1,
                        0,
                    ]
                },
            }
        },
        {
            "$group": {
                "_id": "$_day",
                "present_count": {"$sum": "$_present"},
                "late_count": {"$sum": "$_late"},
            }
        },
        {
            "$unionWith": {
                "pipeline": [
                    {
                        "$documents": [
                            {
                                "_id": start_dt,
                                "present_count": 0,
                                "late_count": 0,
                            }
                        ]
                    }
                ]
            }
        },
        {
            "$group": {
                "_id": "$_id",
                "present_count": {"$sum": "$present_count"},
                "late_count": {"$sum": "$late_count"},
            }
        },
        {
            "$densify": {
                "field": "_id",
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [start_dt, end_exclusive],
                },
            }
        },
        {
            "$set": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$_id"}},
                "is_working_day": {
                    "$in": [{"$dayOfWeek": "$_id"}, [2, 3, 4, 5, 6]]
                },
                "present_count": {"$ifNull": ["$present_count", 0]},
                "late_count": {"$ifNull": ["$late_count", 0]},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "let": {"day": "$date"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$department", department]},
                                    {"$lte": ["$joined_on", "$$day"]},
                                ]
                            }
                        }
                    },
                    {"$count": "count"},
                ],
                "as": "_headcount",
            }
        },
        {
            "$set": {
                "headcount": {
                    "$ifNull": [{"$arrayElemAt": ["$_headcount.count", 0]}, 0]
                }
            }
        },
        {
            "$set": {
                "attendance_rate": {
                    "$cond": [
                        {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
                        _mongo_round_half_up(
                            {
                                "$multiply": [
                                    {"$divide": ["$present_count", "$headcount"]},
                                    100,
                                ]
                            },
                            4,
                        ),
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"_id": 1},
                "output": {
                    "_moving": {
                        "$avg": "$attendance_rate",
                        "window": {"documents": [-6, 0]},
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
                "attendance_rate": 1,
                "moving_avg_7d": {
                    "$cond": [
                        {"$eq": ["$_moving", None]},
                        None,
                        _mongo_round_half_up("$_moving", 4),
                    ]
                },
            }
        },
        {"$sort": {"date": 1}},
    ]


# --------------------------------------------------------------------------- #
# Analytics endpoints
# --------------------------------------------------------------------------- #
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str) -> dict:
    start, end = require_month(month)
    employee = db.employees.find_one({"emp_code": emp_code}, {"joined_on": 1, "_id": 0})
    if employee is None:
        raise HTTPException(404, "employee not found")
    joined_on = require_date(employee["joined_on"], "joined_on")
    first_counted = max(start, joined_on)
    days = working_days(first_counted, end)
    totals = list(
        db.attendance_logs.aggregate(
            employee_monthly_pipeline(emp_code, start, end, joined_on),
            hint="uq_attendance_employee_date",
        )
    )
    summary = totals[0] if totals else {}
    present_days = float(summary.get("present_days", 0))
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": days,
        "present_days": round_half_up(present_days, 2),
        "leave_days": summary.get("leave_days", 0),
        "late_count": summary.get("late_count", 0),
        "total_late_minutes": summary.get("total_late_minutes", 0),
        "total_overtime_minutes": summary.get("total_overtime_minutes", 0),
        "attendance_pct": (
            round_half_up(Decimal(str(present_days)) / Decimal(days) * 100, 4)
            if days
            else None
        ),
    }


@app.get("/analytics/departments/summary")
def department_summary(
    month: str, department: str | None = None
) -> dict:
    start, end = require_month(month)
    pipeline = department_summary_pipeline(start, end, department)
    items = list(db.employees.aggregate(pipeline, hint="ix_employee_joined"))
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str,
    limit: int = Query(10, ge=1, le=50),
    department: str | None = None,
) -> dict:
    start, end = require_month(month)
    pipeline = late_leaderboard_pipeline(start, end, limit, department)
    items = list(
        db.attendance_logs.aggregate(
            pipeline,
            hint="ix_attendance_date_code",
            allowDiskUse=True,
        )
    )
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_date: Annotated[date, Query(alias="from")],
    to_date: Annotated[date, Query(alias="to")],
) -> dict:
    if to_date < from_date:
        raise HTTPException(422, "to must be on or after from")
    if (to_date - from_date).days + 1 > 92:
        raise HTTPException(422, "date range may not exceed 92 days")
    if db.employees.find_one({"department": department}, {"_id": 1}) is None:
        raise HTTPException(404, "department not found")
    pipeline = department_trend_pipeline(department, from_date, to_date)
    items = list(
        db.attendance_logs.aggregate(
            pipeline,
            hint="ix_attendance_date_code",
            allowDiskUse=True,
        )
    )
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# Explain endpoint
# --------------------------------------------------------------------------- #
def _execution_explain(collection: str, operation: dict) -> dict:
    return db.command(
        "explain",
        operation,
        verbosity="executionStats",
    )


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: Literal[
        "attendance_list",
        "employee_monthly",
        "department_summary",
        "late_leaderboard",
        "department_trend",
    ],
    emp_code: str | None = None,
    month: str | None = None,
    department: str | None = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: date | None = None,
    date_to: date | None = None,
    status: Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"] | None = None,
    from_date: Annotated[date | None, Query(alias="from")] = None,
    to_date: Annotated[date | None, Query(alias="to")] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict:
    if endpoint == "attendance_list":
        if date_from and date_to and date_from > date_to:
            raise HTTPException(422, "date_from must be on or before date_to")
        query: dict = {}
        if emp_code is not None:
            query["emp_code"] = emp_code
        if date_from or date_to:
            query["date"] = {}
            if date_from:
                query["date"]["$gte"] = date_from.isoformat()
            if date_to:
                query["date"]["$lte"] = date_to.isoformat()
        if status is not None:
            query["status"] = status
        operation = {
            "find": "attendance_logs",
            "filter": query,
            "sort": {"date": -1, "emp_code": 1},
            "skip": (page - 1) * page_size,
            "limit": page_size,
            "hint": (
                "uq_attendance_employee_date"
                if emp_code is not None
                else "ix_attendance_date_code"
            ),
        }
        collection = "attendance_logs"
    elif endpoint == "employee_monthly":
        if not emp_code or not month:
            raise HTTPException(422, "emp_code and month are required")
        start, end = require_month(month)
        operation = {
            "aggregate": "attendance_logs",
            "pipeline": employee_monthly_pipeline(emp_code, start, end, start),
            "cursor": {},
            "hint": "uq_attendance_employee_date",
        }
        collection = "attendance_logs"
    elif endpoint == "department_summary":
        if not month:
            raise HTTPException(422, "month is required")
        start, end = require_month(month)
        operation = {
            "aggregate": "employees",
            "pipeline": department_summary_pipeline(start, end, department),
            "cursor": {},
            "hint": "ix_employee_joined",
        }
        collection = "employees"
    elif endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(422, "month is required")
        start, end = require_month(month)
        operation = {
            "aggregate": "attendance_logs",
            "pipeline": late_leaderboard_pipeline(start, end, limit, department),
            "cursor": {},
            "hint": "ix_attendance_date_code",
            "allowDiskUse": True,
        }
        collection = "attendance_logs"
    else:
        if not department or from_date is None or to_date is None:
            raise HTTPException(422, "department, from and to are required")
        if to_date < from_date or (to_date - from_date).days + 1 > 92:
            raise HTTPException(422, "invalid trend date range")
        operation = {
            "aggregate": "attendance_logs",
            "pipeline": department_trend_pipeline(department, from_date, to_date),
            "cursor": {},
            "hint": "ix_attendance_date_code",
            "allowDiskUse": True,
        }
        collection = "attendance_logs"

    explain = _execution_explain(collection, operation)
    return {
        "endpoint": endpoint,
        "collection": collection,
        "explain": json.loads(json_util.dumps(explain)),
    }
