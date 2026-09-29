from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

from .cleaning import clean_values, normalize_headers

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SOURCE_FILE = SRC_DIR / "docs" / "students_raw.csv"
CONTRACT_FILE = SRC_DIR / "contract.json"

BRONZE_FILE = SRC_DIR / "bronze" / "students.parquet"
SILVER_FILE = SRC_DIR / "silver" / "students.parquet"
REJECTED_FILE = SRC_DIR / "silver" / "students_rejected.parquet"
GOLD_DIR = SRC_DIR / "gold"

LINEAGE = ["_source_file", "_source_line", "_ingested_at"]
DTYPES = {"int": "int64", "text": "str", "category": "str"}


class SchemaError(Exception):
    """Raised when the source file does not have the columns defined in the contract."""


def load_contract(path=CONTRACT_FILE):
    return json.loads(path.read_text(encoding="utf-8"))


def write_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def one_hot(series, allowed):
    return pd.DataFrame({value: (series == value).astype(int) for value in allowed})


# --- bronze: ingest the source as is, only validating the schema -------------

def to_bronze(contract):
    fmt = contract["format"]
    df = pd.read_csv(SOURCE_FILE, dtype=str, keep_default_na=False, encoding=fmt["encoding"], sep=fmt["delimiter"])
    df.columns = [normalize_headers(col) for col in df.columns]

    expected, found = set(contract["columns"]), set(df.columns)
    missing, extra = sorted(expected - found), sorted(found - expected)
    if missing or (contract.get("strict_columns") and extra):
        raise SchemaError(f"missing columns {missing}, unexpected columns {extra}")

    df["_source_file"] = SOURCE_FILE.name
    df["_source_line"] = df.index + 2  # line 1 is the header
    df["_ingested_at"] = datetime.now(timezone.utc).replace(microsecond=0)
    write_parquet(df, BRONZE_FILE)
    return df


# --- silver: apply the contract to every value -------------------------------

def to_silver(contract):
    bronze = pd.read_parquet(BRONZE_FILE)
    specs = contract["columns"]
    columns = list(specs)

    if contract.get("drop_empty_rows"):
        bronze = bronze[(bronze[columns].apply(lambda col: col.str.strip()) != "").any(axis=1)]

    values, errors = {}, {}
    for col, spec in specs.items():
        results = [clean_values(raw, spec) for raw in bronze[col]]
        values[col] = [value for value, _ in results]
        errors[col] = [error for _, error in results]
    values = pd.DataFrame(values, index=bronze.index)
    errors = pd.DataFrame(errors, index=bronze.index)
    is_valid = errors.isna().all(axis=1)

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

def to_gold(contract):
    silver = pd.read_parquet(SILVER_FILE)
    specs = contract["columns"]
    age = specs["age"]

    # Order: [normalized_age, blue, red, green, São Paulo, Rio, Curitiba]
    # The age is scaled with the contract bounds, not the data min/max, so new students
    # are scaled the same way at prediction time
    xs = pd.concat([
        ((silver["age"] - age["min"]) / (age["max"] - age["min"])).round(4),
        one_hot(silver["color"], specs["color"]["allowed"]),
        one_hot(silver["location"], specs["location"]["allowed"]),
    ], axis=1)
    # Order: [premium, medium, basic]
    ys = one_hot(silver["category"], specs["category"]["allowed"])

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    xs.to_json(GOLD_DIR / "xs.json", orient="values")
    ys.to_json(GOLD_DIR / "ys.json", orient="values")
    return xs, ys