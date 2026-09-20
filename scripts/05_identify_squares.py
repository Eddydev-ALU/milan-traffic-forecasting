#!/usr/bin/env python
"""Identify what a square ID actually is, in the real world.

The grid geojson has one polygon per square but no place names, so this
script computes each requested square's centroid (lat/lon) and prints a
clickable map link. Opening the link shows what is actually there (a
stadium, a station, a park, a business district), which is what turns a
generic "Square 5161" label into a real-world explanation of the traffic
pattern.

Usage
-----
    python scripts/05_identify_squares.py --squares 5161 5059 5259 4159 4556
    python scripts/05_identify_squares.py               # reads top3.json + config extras

The script is defensive about property naming because different releases of
the Milano grid geojson have used different keys (`cellId`, `CELLID`, `id`,
`square_id`, ...); it searches a short list of common candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config  # noqa: E402

CANDIDATE_ID_KEYS = ["cellId", "CELLID", "cell_id", "id", "ID", "square_id", "SQUAREID", "NO"]


def find_geojson(grid_dir: Path) -> Path:
    candidates = list(grid_dir.rglob("*.geojson")) + list(grid_dir.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No .geojson/.json found under {grid_dir}. "
            "Unzip the Milano Grid dataset into data/grid/ first."
        )
    # Prefer a file that actually looks like the grid (many features).
    best, best_n = candidates[0], -1
    for c in candidates:
        try:
            data = json.loads(c.read_text())
            n = len(data.get("features", []))
            if n > best_n:
                best, best_n = c, n
        except Exception:
            continue
    return best


def feature_square_id(props: dict) -> int | None:
    for key in CANDIDATE_ID_KEYS:
        if key in props:
            try:
                return int(props[key])
            except (TypeError, ValueError):
                continue
    return None


def polygon_centroid(geometry: dict) -> tuple[float, float]:
    """Plain-average centroid of the exterior ring -- exact enough for a ~235m square."""
    gtype = geometry["type"]
    coords = geometry["coordinates"]
    if gtype == "Polygon":
        ring = coords[0]
    elif gtype == "MultiPolygon":
        ring = coords[0][0]
    else:
        raise ValueError(f"Unsupported geometry type: {gtype}")
    lons = [pt[0] for pt in ring]
    lats = [pt[1] for pt in ring]
    return sum(lats) / len(lats), sum(lons) / len(lons)  # (lat, lon)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--grid-dir", default="data/grid")
    ap.add_argument("--squares", type=int, nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    grid_dir = Path(args.grid_dir)

    squares = args.squares
    if squares is None:
        squares = list(cfg["eda"]["extra_squares"])
        top3_path = Path(cfg.path("tables_dir")) / "top3.json"
        if top3_path.exists():
            squares = json.loads(top3_path.read_text())["top3"] + squares
        print(f"No --squares given; using top3.json + config extras: {squares}")

    geojson_path = find_geojson(grid_dir)
    print(f"Reading {geojson_path}")
    data = json.loads(geojson_path.read_text())
    features = data.get("features", [])
    if not features:
        raise SystemExit("geojson has no 'features' -- is this really the grid file?")

    sample_keys = list(features[0].get("properties", {}).keys())
    print(f"Property keys found on first feature: {sample_keys}")

    lookup: dict[int, dict] = {}
    for feat in features:
        sid = feature_square_id(feat.get("properties", {}))
        if sid is not None:
            lookup[sid] = feat

    if not lookup:
        raise SystemExit(
            "Could not find a square-id property using any of "
            f"{CANDIDATE_ID_KEYS}. Open the geojson and check the actual key, "
            "then add it to CANDIDATE_ID_KEYS in this script."
        )

    print(f"\nIndexed {len(lookup)} squares.\n")
    rows = []
    for sid in squares:
        feat = lookup.get(sid)
        if feat is None:
            print(f"square {sid}: NOT FOUND in this geojson")
            continue
        lat, lon = polygon_centroid(feat["geometry"])
        maps_url = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
        print(f"square {sid:>5}  lat={lat:.6f}  lon={lon:.6f}   {maps_url}")
        rows.append({"square_id": sid, "lat": lat, "lon": lon, "maps_url": maps_url})

    out_path = Path(cfg.path("tables_dir")) / "square_locations.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved -> {out_path}")
    print(
        "\nOpen each maps_url and note what's actually there (station, stadium, "
        "park, business district, residential area). That observation is what "
        "goes in your EDA discussion next to each square's time-series plot."
    )


if __name__ == "__main__":
    main()