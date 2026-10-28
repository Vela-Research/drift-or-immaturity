"""CONFIRM2 step 0.2: download the raw files, record URL, date, size and sha256 (preregistration/CONFIRM2_SPEC.md).

DATASET: A = SBA 7(a) FOIA (data.sba.gov, as of 2026-06-30); B = FJC IDB civil cumulative file + codebook and the FJC
Biographical Directory (federal judicial service); covariates = BLS LAUS state unemployment, seasonally adjusted (the
FRED copies <ST>UR) and the FHFA all-transactions state HPI. All public domain.
Raw files go to RAW (outside the repo) and are deleted by build.py once the parquet files are checked.
Usage: uv run python download.py -> artifacts/manifest.json
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Final

HERE: Final = Path(__file__).resolve().parent
RAW: Final = Path(os.environ.get("CONFIRM2_RAW", HERE / "artifacts" / "raw"))
SBA: Final = "https://data.sba.gov/sites/default/files/uploaded_resources/"
FJC: Final = "https://www.fjc.gov/sites/default/files/"
STATES: Final = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "DC",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]
FILES: Final = {
    **{
        f"sba/{n}": SBA + n
        for n in (
            "FOIA_7a_FY1991_FY1999_asof_260630.csv",
            "FOIA_7a_FY2000_FY2009_asof_260630.csv",
            "FOIA_7a_FY2010_FY2019_asof_260630.csv",
            "FOIA_7a_FY2020_Present_asof_260630.csv",
            "7a_504_foia_data_dictionary.xlsx",
        )
    },
    "fjc/cv88on.zip": FJC + "idb/textfiles/cv88on.zip",
    "fjc/civil_codebook.pdf": FJC
    + "idb/codebooks/Civil%20Codebook%201988%20Forward%2010252023.pdf",
    "fjc/federal-judicial-service.csv": FJC + "history/federal-judicial-service.csv",
    "cov/fhfa_state.csv": "https://www.fhfa.gov/hpi/download/quarterly_datasets/hpi_at_state.csv",
    **{
        f"cov/unrate_{s}.csv": f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}UR"
        for s in STATES
    },
}


LAUS_SOURCE: Final = (
    "BLS LAUS state unemployment rate, seasonally adjusted (series LASST<fips>0000000000003), as copied by FRED "
    "(<ST>UR); FRED copy accepted before any method run"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest_path = HERE / "artifacts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, url in FILES.items():
        path = RAW / name
        if name in manifest and path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # curl, not urllib: urllib hangs on these hosts (no IPv4 fallback)
        headers = subprocess.run(
            ["curl", "-sSfL", "--retry", "3", "-m", "1800", "-D", "-", "-o", str(path), url],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        modified = next(
            (
                h.split(":", 1)[1].strip()
                for h in headers.splitlines()
                if h.lower().startswith("last-modified:")
            ),
            None,
        )
        manifest[name] = {
            "url": url,
            "downloaded_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_modified": modified,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if name.startswith("cov/unrate_"):
            manifest[name]["source"] = LAUS_SOURCE
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(name, manifest[name]["bytes"])


if __name__ == "__main__":
    main()
