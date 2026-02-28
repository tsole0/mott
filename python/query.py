"""
query.py
========
Query the Materials Project for transition-metal compounds and save the raw
results to data/raw_materials.json.

Fetches all TM-containing materials within the stability window. Hubbard U
values are NOT available from this endpoint — they are hardcoded from the
standard MP GGA+U parameterization in features.py instead.

Usage:
    python python/query.py
    python python/query.py --api-key YOUR_KEY
    python python/query.py --output path/to/out.json
    python python/query.py --chemsys "Fe-O,Ni-O"
    python python/query.py --elements "Fe,Ni"
    python python/query.py --max 500

    # Or set your key as an env variable:
    export MP_API_KEY=YOUR_KEY
    python python/query.py
"""

import os
import sys
import json
import argparse
from pathlib import Path

from mp_api.client import MPRester

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
DEFAULT_OUT  = DATA_DIR / "raw_materials.json"

FIELDS = [
    # Identity
    "material_id",
    "formula_pretty",
    "chemsys",
    "elements",
    # Electronic
    "band_gap",
    "is_metal",
    "is_magnetic",
    "ordering",
    "total_magnetization",
    "num_magnetic_sites",
    # Structure & composition
    "structure",
    "nsites",
    "volume",
    "density",
    "nelements",
    "symmetry",
    # Stability
    "energy_above_hull",
    "formation_energy_per_atom",
    # Band-edge scalars
    "cbm",
    "vbm",
    "efermi",
]


# Default element filter: TM elements relevant to Mott physics.
# Used when --elements is passed on the CLI. NOT used as the default because
# mp_api interprets elements as AND (material must contain ALL listed elements),
# so passing the full TM list returns nothing. Default query uses stability
# filter only; features.py filters to TM sites during processing.
TM_ELEMENTS = [
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Nb", "Mo", "Ru", "Rh", "Ta", "W", "Os", "Ir",
]


def query(
    api_key: str,
    chemsys: list[str] | None = None,
    elements: list[str] | None = None,
    max_results: int | None = None,
) -> list[dict]:
    kwargs: dict = {
        "fields": FIELDS,
        "energy_above_hull": (0.0, 0.5),
        "deprecated": False,
    }
    if chemsys:
        kwargs["chemsys"] = chemsys
    elif elements:
        # Caller passed a specific element list — use it (AND logic, so keep list short)
        kwargs["elements"] = elements

    with MPRester(api_key) as mpr:
        print("Querying Materials Project (energy_above_hull ≤ 0.5 eV)...")
        docs = mpr.materials.summary.search(**kwargs)

    print(f"  Retrieved {len(docs)} materials.")

    if max_results:
        docs = docs[:max_results]

    print(f"  Serializing {len(docs)} materials...")

    data = []
    for r in docs:
        d = r.model_dump()
        if r.structure is not None:
            d["structure"] = r.structure.as_dict()
        data.append(d)

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query Materials Project (GGA+U materials) and save to JSON."
    )
    parser.add_argument("--api-key",  metavar="KEY",
                        help="MP API key (or set MP_API_KEY env var)")
    parser.add_argument("--output",   metavar="FILE", default=str(DEFAULT_OUT),
                        help=f"Output JSON file (default: {DEFAULT_OUT})")
    parser.add_argument("--chemsys",  metavar="SYSTEMS",
                        help="Comma-separated chemical systems, e.g. 'Fe-O,Ni-O'")
    parser.add_argument("--elements", metavar="ELEMS",
                        help="Comma-separated elements to filter by, e.g. 'Fe,Ni'")
    parser.add_argument("--max",      metavar="N", type=int,
                        help="Max number of results to keep")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        print("Error: no API key. Pass --api-key or set MP_API_KEY.")
        print("Get your key at: https://materialsproject.org/dashboard")
        sys.exit(1)

    chemsys  = args.chemsys.split(",")  if args.chemsys  else None
    elements = args.elements.split(",") if args.elements else None

    data = query(api_key, chemsys=chemsys, elements=elements, max_results=args.max)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f)

    print(f"Saved {len(data)} materials to {out}")


if __name__ == "__main__":
    main()
