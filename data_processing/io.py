"""Raw-data loading for PathSnap trips. Owner: Venkatesh."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARTITIONS = {"train": "train", "test": "student_test"}


def trip_ids(partition: str) -> list[str]:
    return sorted(p.name for p in (DATA_DIR / PARTITIONS[partition]).iterdir() if p.is_dir())


def load_trip(trip_id: str, partition: str) -> dict:
    d = DATA_DIR / PARTITIONS[partition] / trip_id
    track = pd.read_csv(d / f"{trip_id}.track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(d / f"{trip_id}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(d / f"{trip_id}.arcs", sep=r"\s+", header=None, names=["u", "v"])
    rf = d / f"{trip_id}.route"
    route = np.loadtxt(rf, dtype=np.int64, ndmin=1) if rf.exists() and rf.stat().st_size else None
    return {"id": trip_id, "track": track, "nodes": nodes, "arcs": arcs, "route": route}


def clean_track(track: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Sort by time, drop NaNs / duplicate timestamps / points far outside the road network bbox."""
    tr = track.dropna().sort_values("t").drop_duplicates("t").reset_index(drop=True)
    pad = 0.01  # ~1 km
    ok = (tr.lon.between(nodes.lon.min() - pad, nodes.lon.max() + pad)
          & tr.lat.between(nodes.lat.min() - pad, nodes.lat.max() + pad))
    return tr[ok].reset_index(drop=True)
