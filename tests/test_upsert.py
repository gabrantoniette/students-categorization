"""The upsert of new students: only who signed up after the last student already
in goes in, and every row that does not go in is recorded with the reason.

Run from the project root: python -m unittest discover -s tests
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src import process
from src.cleaning import clean_values

HEADER = "name,age,color,location,created_at\n"


class UpsertTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.source = self.root / "docs" / "students_new.csv"
        self.source.parent.mkdir()

        # Every file the upsert reads or writes, and the training set's files,
        # point inside the temporary folder
        paths = {
            "NEW_SOURCE_FILE": self.source,
            "NEW_BRONZE_FILE": self.root / "bronze" / "new_students.parquet",
            "NEW_SILVER_FILE": self.root / "silver" / "new_students.parquet",
            "NEW_GOLD_FILE": self.root / "gold" / "new_students.json",
            "QUARANTINE_FILE": self.root / "quarantine" / "new_students.csv",
            "BRONZE_FILE": self.root / "bronze" / "students.parquet",
            "SILVER_FILE": self.root / "silver" / "students.parquet",
            "REJECTED_FILE": self.root / "silver" / "students_rejected.parquet",
            "GOLD_DIR": self.root / "gold",
        }
        patcher = mock.patch.multiple(process, **paths)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.contract = process.load_contract()

    def write(self, *rows):
        self.source.write_text(HEADER + "".join(row + "\n" for row in rows), encoding="utf-8")

    def append(self, *rows):
        with self.source.open("a", encoding="utf-8") as file:
            file.write("".join(row + "\n" for row in rows))

    def silver(self):
        return pd.read_parquet(process.NEW_SILVER_FILE)

    def quarantine(self):
        return pd.read_csv(process.QUARANTINE_FILE, dtype=str, keep_default_na=False)

    def test_the_first_run_adds_every_valid_student(self):
        self.write(
            "mariana souza,24,verde,sp,2026-10-05T09:12:00-03:00",
            "Rafael Lima,41 anos,blue,rio de janeiro - rj,2026-10-05T10:47:00-03:00",
        )

        summary = process.upsert(self.contract)

        silver = self.silver()
        self.assertEqual(summary["added"], 2)
        self.assertEqual(list(silver["id"]), [1, 2])
        self.assertEqual(list(silver["name"]), ["Mariana Souza", "Rafael Lima"])
        self.assertEqual(list(silver["location"]), ["São Paulo", "Rio"])
        self.assertEqual(silver.loc[0, "created_at"], pd.Timestamp("2026-10-05T12:12:00Z"))

    def test_running_again_on_the_same_file_adds_nothing(self):
        self.write(
            "mariana souza,24,verde,sp,2026-10-05T09:12:00-03:00",
            "Lucas Pereira,27,roxo,Rio,2026-10-06T08:40:00-03:00",
        )
        process.upsert(self.contract)

        summary = process.upsert(self.contract)

        self.assertEqual(summary["added"], 0)
        self.assertEqual(len(self.silver()), 1)
        self.assertEqual(len(self.quarantine()), 1)

    def test_only_students_who_signed_up_after_the_last_one_go_in(self):
        self.write("mariana souza,24,verde,sp,2026-10-05T09:12:00-03:00")
        process.upsert(self.contract)
        self.append(
            "Beatriz Rocha,33,azul,Curitiba,2026-10-06T14:05:00-03:00",
            "Thiago Alves,58,red,são paulo,2026-10-04T18:30:00-03:00",
        )

        summary = process.upsert(self.contract)

        self.assertEqual(summary["added"], 1)
        self.assertEqual(list(self.silver()["name"]), ["Mariana Souza", "Beatriz Rocha"])
        quarantine = self.quarantine()
        self.assertEqual(list(quarantine["name"]), ["Thiago Alves"])
        self.assertEqual(list(quarantine["reason"]), ["late"])

    def test_a_student_sent_again_on_another_line_is_recorded_as_a_duplicate(self):
        self.write("mariana souza,24,verde,sp,2026-10-05T09:12:00-03:00")
        process.upsert(self.contract)
        self.append("Mariana Souza,24,green,São Paulo,2026-10-05T09:12:00-03:00")

        summary = process.upsert(self.contract)

        quarantine = self.quarantine()
        self.assertEqual(summary["added"], 0)
        self.assertEqual(list(quarantine["reason"]), ["duplicate"])
        self.assertEqual(list(quarantine["_source_line"]), ["3"])

    def test_bronze_keeps_each_row_once_from_the_first_time_it_arrived(self):
        self.write("mariana souza,24,verde,sp,2026-10-05T09:12:00-03:00")
        process.upsert(self.contract)
        first = pd.read_parquet(process.NEW_BRONZE_FILE)["_ingested_at"].iloc[0]
        self.append("Beatriz Rocha,33,azul,Curitiba,2026-10-06T14:05:00-03:00")

        with mock.patch.object(process, "datetime") as clock:
            clock.now.return_value = first + pd.Timedelta(hours=1)
            process.upsert(self.contract)

        bronze = pd.read_parquet(process.NEW_BRONZE_FILE)
        self.assertEqual(list(bronze["name"]), ["mariana souza", "Beatriz Rocha"])
        self.assertEqual(bronze["_ingested_at"].iloc[0], first)

    def test_every_row_that_does_not_go_in_is_recorded_with_the_reason(self):
        self.write(
            "Camila Ferreira,19,vermelho,cwb,2026-10-06T08:02:00-03:00",
            "Lucas Pereira,27,roxo,Rio,2026-10-06T08:40:00-03:00",
            "Juliana Costa,45,green,Rio,06/10/2026 09:15",
            ",,,,",
            "Camila Ferreira,19,vermelho,cwb,2026-10-06T08:02:00-03:00",
        )

        summary = process.upsert(self.contract)

        quarantine = self.quarantine()
        self.assertEqual(summary["added"], 1)
        self.assertEqual(list(quarantine["reason"]), ["invalid", "invalid", "empty", "duplicate"])
        self.assertEqual(list(quarantine["_source_line"]), ["3", "4", "5", "6"])
        self.assertIn("roxo", quarantine.loc[0, "color_error"])
        self.assertIn("cannot parse", quarantine.loc[1, "created_at_error"])

    def test_new_students_are_encoded_like_the_training_set(self):
        # Line 3 of students_raw.csv, as the README shows it in the gold layer
        self.write("henrique cardoso,53,verde,Curitiba,2026-10-05T09:12:00Z")

        process.upsert(self.contract)

        gold = json.loads(process.NEW_GOLD_FILE.read_text(encoding="utf-8"))
        self.assertEqual(gold, [{
            "id": 1,
            "name": "Henrique Cardoso",
            "created_at": "2026-10-05T09:12:00+00:00",
            "xs": [0.7447, 0, 0, 1, 0, 0, 1],
        }])

    def test_the_training_set_is_left_alone(self):
        self.write("mariana souza,24,verde,sp,2026-10-05T09:12:00-03:00")

        process.upsert(self.contract)

        for path in (process.BRONZE_FILE, process.SILVER_FILE, process.REJECTED_FILE,
                     process.GOLD_DIR / "xs.json", process.GOLD_DIR / "ys.json"):
            self.assertFalse(path.exists(), path)


class DatetimeTest(unittest.TestCase):
    def test_a_timestamp_with_an_offset_is_stored_in_utc(self):
        value, error = clean_values("2026-10-05T09:12:00-03:00", {"type": "datetime", "required": True})

        self.assertIsNone(error)
        self.assertEqual(value.isoformat(), "2026-10-05T12:12:00+00:00")

    def test_a_timestamp_without_an_offset_is_read_as_utc(self):
        value, error = clean_values("2026-10-05 09:12", {"type": "datetime", "required": True})

        self.assertIsNone(error)
        self.assertEqual(value.isoformat(), "2026-10-05T09:12:00+00:00")

    def test_a_date_in_another_format_is_rejected(self):
        value, error = clean_values("06/10/2026 09:15", {"type": "datetime", "required": True})

        self.assertIsNone(value)
        self.assertIn("cannot parse", error)


if __name__ == "__main__":
    unittest.main()
