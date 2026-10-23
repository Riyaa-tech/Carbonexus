"""Configuration loading (TOML) and typed access."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "carbonexus.toml"

# Channel order. The paper's 13 channels = 4 gases + 5 meteorology + 3 land use + CEI.
GASES = ("no2", "ch4", "so2", "co")
MET = ("u10", "v10", "t2m", "tp", "sp")
LANDUSE = ("built", "crop", "nlights")
CHANNELS = GASES + MET + LANDUSE


@dataclass
class Config:
    raw: dict
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("CARBONEXUS_DATA", ROOT / "data")))

    def __getitem__(self, k):
        return self.raw[k]

    # ---- grid -------------------------------------------------------------
    @property
    def lats(self) -> np.ndarray:
        s = self.raw["study"]
        return s["north"] - s["res"] * (np.arange(s["height"]) + 0.5)

    @property
    def lons(self) -> np.ndarray:
        s = self.raw["study"]
        return s["west"] + s["res"] * (np.arange(s["width"]) + 0.5)

    @property
    def months(self) -> pd.DatetimeIndex:
        s = self.raw["study"]
        return pd.date_range(s["start"], s["end"], freq="MS")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(west, south, east, north)"""
        s = self.raw["study"]
        return (s["west"], s["north"] - s["res"] * s["height"],
                s["west"] + s["res"] * s["width"], s["north"])

    def cei_weights(self) -> dict[str, float]:
        c = self.raw["cei"]
        return {g: c[f"w_{g}"] for g in GASES}

    def gwp(self) -> dict[str, float]:
        c = self.raw["cei"]
        return {g: c[f"gwp_{g}"] for g in GASES}

    # ---- paths ------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


def load_config(path: str | os.PathLike | None = None, data_dir: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "rb") as f:
        raw = tomllib.load(f)
    cfg = Config(raw)
    if data_dir:
        cfg.data_dir = Path(data_dir)
    return cfg
