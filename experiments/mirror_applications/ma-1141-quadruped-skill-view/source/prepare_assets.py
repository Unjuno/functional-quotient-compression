"""Generate and hash the four frozen morphology variants before data collection."""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from gymnasium.envs.mujoco import ant_v5

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets"
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text())


def scaled_vec(value: str, scale: float) -> str:
    vals = [float(x) for x in value.split()]
    return " ".join(f"{x * scale:.9g}" for x in vals)


def make_morphology(spec: dict) -> Path:
    src = Path(ant_v5.__file__).parent / "assets" / "ant.xml"
    tree = ET.parse(src)
    root = tree.getroot()
    leg_scale = spec["leg_length_scale"]
    for body in root.iter("body"):
        if body.get("name") == "torso":
            continue
        if body.get("pos"):
            body.set("pos", scaled_vec(body.get("pos"), leg_scale))
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if name == "torso_geom":
            geom.set("density", f"{5.0 * spec['torso_mass_scale']:.9g}")
        elif name != "floor" and geom.get("fromto"):
            geom.set("fromto", scaled_vec(geom.get("fromto"), leg_scale))
            if geom.get("size"):
                geom.set("size", scaled_vec(geom.get("size"), leg_scale))
    for actuator in root.iter("motor"):
        actuator.set("gear", f"{150.0 * spec['actuator_scale']:.9g}")
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSET_DIR / f"ant_morph_{spec['id']}.xml"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in PROTOCOL["simulator"]["morphologies"]:
        path = make_morphology(spec)
        rows.append({"morphology_id": spec["id"], "path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    base = Path(ant_v5.__file__).parent / "assets" / "ant.xml"
    manifest_path = ROOT / "source" / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base_ant_xml_sha256"] = hashlib.sha256(base.read_bytes()).hexdigest()
    manifest["morphologies"] = rows
    manifest.setdefault("trajectory_data", "not generated")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
