"""Extract the columns of PatEx application_data that the feasibility check needs (the patent feasibility plan).

DATASET: USPTO Patent Examination Research Dataset (PatEx), release folder "2022" (PEDS data of June 2023),
file application_data.csv.zip, downloaded on 2026-10-04 into data/raw/patex/.
Keeps every nonprovisional utility application with a filing date (all years: the pending inventory needs the
applications filed before 2001 and after 2019 too).
Usage: extract.py -> data/processed/patex_utility.parquet
"""

from __future__ import annotations

import logging
import zipfile
from typing import Final

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from maturity import data

log = logging.getLogger(__name__)
RAW: Final = data.DATA_DIR / "raw" / "patex" / "application_data.csv.zip"
OUT: Final = data.DATA_DIR / "processed" / "patex_utility.parquet"
COLUMNS: Final = [
    "application_number",
    "filing_date",
    "application_invention_type",
    "examiner_art_unit",
    "appl_status_desc",
    "appl_status_date",
    "earliest_pgpub_date",
    "patent_issue_date",
]


def main() -> None:
    types: dict[str, int] = {}
    parts = []
    with zipfile.ZipFile(RAW) as z, z.open("application_data.csv") as f:
        reader = pacsv.open_csv(
            f,
            read_options=pacsv.ReadOptions(block_size=1 << 27),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                include_columns=COLUMNS, column_types={c: pa.string() for c in COLUMNS}
            ),
        )
        for batch in reader:
            t = pa.Table.from_batches([batch])
            for value, count in zip(
                *[
                    x.to_pylist()
                    for x in pc.value_counts(t["application_invention_type"]).flatten()
                ],
                strict=True,
            ):
                types[str(value)] = types.get(str(value), 0) + count
            kind = pc.utf8_lower(pc.fill_null(t["application_invention_type"], ""))
            keep = pc.and_(pc.is_valid(t["filing_date"]), pc.match_substring(kind, "utility"))
            parts.append(t.filter(keep))
    out = pa.concat_tables(parts)
    pq.write_table(out, OUT)
    log.info("application types in the file: %s", types)
    log.info("kept %d utility applications with a filing date", out.num_rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
