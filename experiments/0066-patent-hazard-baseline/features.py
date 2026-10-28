"""Application fields fixed at filing for model M3 (the patent hazard specification).

DATASET: USPTO PatEx, release folder "2022" (PEDS data of June 2023), application_data.csv.zip in data/raw/patex/ (the
file of experiment 0020). Keeps the utility applications with a filing date, as experiments/0020/extract.py.
Usage: uv run python features.py -> artifacts/features.parquet
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from maturity import data

log = logging.getLogger(__name__)
RAW: Final = data.DATA_DIR / "raw" / "patex" / "application_data.csv.zip"
OUT: Final = Path(__file__).resolve().parent / "artifacts" / "features.parquet"
COLUMNS: Final = [
    "application_number",
    "filing_date",
    "application_invention_type",
    "uspc_class",
    "small_entity_indicator",
    "aia_first_to_file",
]


def main() -> None:
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
            kind = pc.utf8_lower(pc.fill_null(t["application_invention_type"], ""))
            keep = pc.and_(pc.is_valid(t["filing_date"]), pc.match_substring(kind, "utility"))
            parts.append(t.filter(keep).select(["application_number", *COLUMNS[3:]]))
    out = pa.concat_tables(parts)
    pq.write_table(out, OUT)
    log.info("kept %d utility applications", out.num_rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
