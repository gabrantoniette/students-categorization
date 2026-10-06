from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

from src.cleaning import clean_values, normalize_headers

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SOURCE_FILE = SRC_DIR / "docs" / "students_raw.csv"
NEW_SOURCE_FILE = SRC_DIR / "docs" / "students_new.csv"
CONTRACT_FILE = SRC_DIR / "contract.json"

BRONZE_FILE = SRC_DIR / "bronze" / "students.parquet"
SILVER_FILE = SRC_DIR / "silver" / "students.parquet"
REJECTED_FILE = SRC_DIR / "silver" / "students_rejected.parquet"
GOLD_DIR = SRC_DIR / "gold"

# New students go through the same layers in files of their own, so the training set never changes
NEW_BRONZE_FILE = SRC_DIR / "bronze" / "new_students.parquet"
NEW_SILVER_FILE = SRC_DIR / "silver" / "new_students.parquet"
NEW_GOLD_FILE = GOLD_DIR / "new_students.json"
# Every new student who does not go in, with the reason
QUARANTINE_FILE = SRC_DIR / "quarantine" / "new_students.csv"

LINEAGE = ["_source_file", "_source_line", "_ingested_at"]
DTYPES = {"int": "int64", "text": "str", "category": "str", "datetime": "datetime64[us, UTC]"}


class SchemaError(Exception):
    """Raised when the source file does not have the columns defined in the contract."""


def load_contract(path=CONTRACT_FILE):
    return json.loads(path.read_text(encoding="utf-8"))


def write_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def isin(df, other, keys):
    """Which rows of df have the same keys as a row of other."""
    if other is None or other.empty:
        return pd.Series(False, index=df.index)
    found = pd.MultiIndex.from_frame(df[keys].astype(str)).isin(pd.MultiIndex.from_frame(other[keys].astype(str)))
    return pd.Series(found, index=df.index)


def one_hot(series, allowed):
    return pd.DataFrame({value: (series == value).astype(int) for value in allowed})


def is_empty(df, columns):
    return ~(df[columns].apply(lambda col: col.str.strip()) != "").any(axis=1)


# --- bronze: ingest the source as is, only validating the schema -------------

def read_source(path, contract, columns):
    fmt = contract["format"]
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=fmt["encoding"], sep=fmt["delimiter"])
    df.columns = [normalize_headers(col) for col in df.columns]

    expected, found = set(columns), set(df.columns)
    missing, extra = sorted(expected - found), sorted(found - expected)
    if missing or (contract.get("strict_columns") and extra):
        raise SchemaError(f"missing columns {missing}, unexpected columns {extra}")

    df["_source_file"] = path.name
    df["_source_line"] = df.index + 2  # line 1 is the header
    df["_ingested_at"] = datetime.now(timezone.utc).replace(microsecond=0)
    return df


def to_bronze(contract):
    df = read_source(SOURCE_FILE, contract, contract["columns"])
    write_parquet(df, BRONZE_FILE)
    return df


# --- silver: apply the contract to every value -------------------------------

def clean(bronze, specs):
    """The clean values, one error per field that failed, and which rows passed."""
    values, errors = {}, {}
    for col, spec in specs.items():
        results = [clean_values(raw, spec) for raw in bronze[col]]
        values[col] = [value for value, _ in results]
        errors[col] = [error for _, error in results]
    values = pd.DataFrame(values, index=bronze.index)
    errors = pd.DataFrame(errors, index=bronze.index)
    return values, errors, errors.isna().all(axis=1)


def to_silver(contract):
    bronze = pd.read_parquet(BRONZE_FILE)
    specs = contract["columns"]
    columns = list(specs)

    if contract.get("drop_empty_rows"):
        bronze = bronze[~is_empty(bronze, columns)]

    values, errors, is_valid = clean(bronze, specs)

    silver = values[is_valid].astype({col: DTYPES[spec["type"]] for col, spec in specs.items()})
    silver = silver.join(bronze[LINEAGE])
    if contract.get("deduplicate"):
        silver = silver.drop_duplicates(subset=columns)
    # Rejected rows keep the raw values and get one error column per field
    rejected = bronze[~is_valid].join(errors[~is_valid].add_suffix("_error"))

    write_parquet(silver, SILVER_FILE)
    write_parquet(rejected, REJECTED_FILE)
    return silver, rejected


# --- gold: encode the students as the model's input and output vectors -------

def encode_inputs(silver, specs):
    age = specs["age"]
    # Order: [normalized_age, blue, red, green, São Paulo, Rio, Curitiba]
    # The age is scaled with the contract bounds, not the data min/max, so new students
    # are scaled the same way at prediction time
    return pd.concat([
        ((silver["age"] - age["min"]) / (age["max"] - age["min"])).round(4),
        one_hot(silver["color"], specs["color"]["allowed"]),
        one_hot(silver["location"], specs["location"]["allowed"]),
    ], axis=1)


def to_gold(contract):
    silver = pd.read_parquet(SILVER_FILE)
    specs = contract["columns"]

    xs = encode_inputs(silver, specs)
    # Order: [premium, medium, basic]
    ys = one_hot(silver["category"], specs["category"]["allowed"])

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    xs.to_json(GOLD_DIR / "xs.json", orient="values")
    ys.to_json(GOLD_DIR / "ys.json", orient="values")
    return xs, ys


# --- upsert: add only the new students who signed up after the last one in ----

def new_student_specs(contract):
    """The contract's columns without the label, plus the ones only new students have."""
    specs = {col: spec for col, spec in contract["columns"].items() if col != contract["label"]}
    return {**specs, **contract["new_students"]["columns"]}


def upsert(contract):
    specs = new_student_specs(contract)
    columns = list(specs)
    timestamp = contract["new_students"]["timestamp"]
    # A line of the file: what it says and where it came from
    line = columns + ["_source_file", "_source_line"]

    # bronze: every line is kept once, as it arrived the first time
    arrived = read_source(NEW_SOURCE_FILE, contract, columns)
    bronze = pd.read_parquet(NEW_BRONZE_FILE) if NEW_BRONZE_FILE.exists() else None
    landed = arrived[~isin(arrived, bronze, line)]
    write_parquet(landed if bronze is None else pd.concat([bronze, landed], ignore_index=True), NEW_BRONZE_FILE)

    # silver: apply the contract, then compare with the students already in
    empty = is_empty(arrived, columns)
    values, errors, is_valid = clean(arrived, specs)
    valid = values[is_valid].astype({col: DTYPES[spec["type"]] for col, spec in specs.items()})
    valid = valid.join(arrived[LINEAGE])

    current = pd.read_parquet(NEW_SILVER_FILE) if NEW_SILVER_FILE.exists() else None
    watermark = current[timestamp].max() if current is not None and len(current) else None

    # A line already in was added on an earlier run. The same student on another line was
    # sent again, and a student who signed up before the last one in arrived late
    in_silver = isin(valid, current, columns)
    already_in = isin(valid, current, line)
    sent_again = in_silver & ~already_in
    late = ~in_silver & (valid[timestamp] <= watermark) if watermark is not None else pd.Series(False, index=valid.index)

    fresh = valid[~in_silver & ~late]
    repeated = fresh.duplicated(subset=columns)
    added = fresh[~repeated].copy()
    first_id = 1 if watermark is None else int(current["id"].max()) + 1
    added.insert(0, "id", range(first_id, first_id + len(added)))
    silver = added if watermark is None else pd.concat([current, added], ignore_index=True)
    write_parquet(silver, NEW_SILVER_FILE)

    # quarantine: every line that did not go in, once, with the values it arrived with
    reason = pd.Series(None, index=arrived.index, dtype="object")
    reason[~is_valid] = "invalid"
    reason[empty] = "empty"
    reason[late[late].index] = "late"
    reason[sent_again[sent_again].index] = "duplicate"
    reason[repeated[repeated].index] = "duplicate"
    reason = reason.dropna()
    problems = errors.loc[reason.index].add_suffix("_error")
    problems.loc[reason != "invalid"] = None
    rows = pd.concat([reason.rename("reason"), arrived.loc[reason.index, columns], problems,
                      arrived.loc[reason.index, LINEAGE]], axis=1)
    quarantined = pd.read_csv(QUARANTINE_FILE, dtype=str, keep_default_na=False) if QUARANTINE_FILE.exists() else None
    recorded = rows[~isin(rows, quarantined, line)]
    QUARANTINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    recorded.to_csv(QUARANTINE_FILE, mode="a", header=quarantined is None, index=False)

    # gold: every new student in, encoded like the training set, ready for the model
    vectors = json.loads(encode_inputs(silver, specs).to_json(orient="values"))
    students = [
        {"id": int(id_), "name": name, timestamp: when.isoformat(), "xs": xs}
        for id_, name, when, xs in zip(silver["id"], silver["name"], silver[timestamp], vectors)
    ]
    NEW_GOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    NEW_GOLD_FILE.write_text(json.dumps(students, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "arrived": len(arrived),
        "landed": len(landed),
        "added": len(added),
        "already_in": int(already_in.sum()),
        "watermark": watermark,
        "recorded": recorded["reason"].value_counts().to_dict(),
        "total": len(silver),
    }
