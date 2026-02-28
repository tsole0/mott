"""
query_mp.py
===========
Queries the Materials Project API and saves raw results to a JSON file.

Usage:
    python query_mp.py --api-key YOUR_KEY
    python query_mp.py --api-key YOUR_KEY --output results.json
    python query_mp.py --api-key YOUR_KEY --chemsys "Fe-O,Ni-O"
    python query_mp.py --api-key YOUR_KEY --max 500

    # Or set your key as an env variable:
    export MP_API_KEY=YOUR_KEY
    python query_mp.py
"""

import os
import sys
import argparse
from pathlib import Path
from mp_api.client import MPRester
from monty.serialization import dumpfn
from monty.json import jsanitize

FIELDS = [
    # Identity
    "material_id",
    "formula_pretty",
    "chemsys",
    "elements",
    # Electronic - core signals
    "band_gap",
    "is_metal",
    "is_magnetic",
    "total_magnetization",
    "num_magnetic_sites",
    # Structure
    "structure",
    "nsites",
    "volume",
    "density",
    "symmetry",
    # Composition
    "nelements",
    # Stability
    "energy_above_hull",
    "formation_energy_per_atom",
    # Electronic structure scalars
    "cbm",
    "vbm",
    "efermi",
]

# Transition metals commonly found in Mott systems
TM_ELEMENTS = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
               "Mo", "Ru", "Rh", "Pd", "W", "Re", "Os", "Ir"]


def query(api_key: str, chemsys: list = None, elements: list = None,
          max_results: int = None) -> list:

    with MPRester(api_key) as mpr:
        kwargs = {
            "fields": FIELDS,
            "energy_above_hull": (0.0, 0.5),
            "deprecated": False,
        }

        if chemsys:
            kwargs["chemsys"] = chemsys
        elif elements:
            kwargs["elements"] = elements
        else:
            kwargs["elements"] = TM_ELEMENTS

        print(f"Querying Materials Project API...")
        docs = mpr.materials.summary.search(**kwargs)
        if max_results:
            docs = docs[:max_results]
        print(f"Retrieved {len(docs)} materials.")

        return [jsanitize(doc, with_data=True) for doc in docs]


def main():
    parser = argparse.ArgumentParser(description="Query Materials Project and save to JSON")
    parser.add_argument("--api-key",  metavar="KEY",
                        help="MP API key (or set MP_API_KEY env var)")
    parser.add_argument("--output",   metavar="FILE", default="../../data/raw_materials.json",
                        help="Output JSON file (default: ../../data/raw_materials.json)")
    parser.add_argument("--chemsys",  metavar="SYSTEMS",
                        help="Comma-separated chemical systems, e.g. 'Fe-O,Ni-O'")
    parser.add_argument("--elements", metavar="ELEMS",
                        help="Comma-separated elements to filter by, e.g. 'Fe,O'")
    parser.add_argument("--max",      metavar="N", type=int,
                        help="Max number of materials to fetch")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        print("Error: No API key. Pass --api-key or set MP_API_KEY.")
        print("Get your key at: https://materialsproject.org/dashboard")
        sys.exit(1)

    chemsys  = args.chemsys.split(",")  if args.chemsys  else None
    elements = args.elements.split(",") if args.elements else None

    data = query(api_key, chemsys=chemsys, elements=elements, max_results=args.max)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dumpfn(data, args.output, indent=2)

    print(f"Saved {len(data)} materials to {args.output}")


if __name__ == "__main__":
    main()