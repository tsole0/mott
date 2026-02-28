from mp_api.client import MPRester
import json
import os

API_KEY = os.getenv("MP_API_KEY", "your_key_here")

fields = [
    "material_id", "formula_pretty", "chemsys", "elements",
    "band_gap", "is_metal", "is_magnetic", "ordering",
    "total_magnetization", "num_magnetic_sites",
    "is_hubbard", "hubbard_u", "run_type",
    "nsites", "volume", "density", "crystal_system",
    "composition", "nelements",
    "energy_above_hull", "formation_energy_per_atom",
    "cbm", "vbm", "efermi",
    "structure",  # needed to compute bond lengths / U/W via mott module
]

with MPRester(API_KEY) as mpr:
    # Only pull materials DFT classifies as metals
    results = mpr.materials.summary.search(
        is_metal=True,
        fields=fields
    )

# Serialize structure separately so pymatgen's as_dict() is used
data = []
for r in results:
    d = r.model_dump()
    if r.structure is not None:
        d["structure"] = r.structure.as_dict()
    data.append(d)

with open("raw_materials.json", "w") as f:
    json.dump(data, f)

print(f"Pulled {len(data)} materials")