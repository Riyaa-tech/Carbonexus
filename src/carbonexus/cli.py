"""Command line interface: `carbonexus <command>`."""
from __future__ import annotations

import argparse
import sys

from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="carbonexus", description=__doc__)
    ap.add_argument("--config", help="path to config TOML (default config/carbonexus.toml)")
    ap.add_argument("--data-dir", help="data directory (default ./data or $CARBONEXUS_DATA)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download TROPOMI / ERA5-Land / land-use monthly composites via Earth Engine")
    sub.add_parser("fetch-firms", help="download NASA FIRMS VIIRS wildfire detections")
    r = sub.add_parser("run", help="preprocess, compute CEI, classify hotspots, forecast; write data/processed")
    r.add_argument("--bilstm", action="store_true", help="also train the paper's BiLSTM (needs torch)")
    r.add_argument("--epochs", type=int, default=200)
    r.add_argument("--edgar", help="optional EDGAR v8.0 NetCDF for cross-validation")
    r.add_argument("--carb", help="optional CARB facility CSV (latitude,longitude,emissions)")
    sub.add_parser("serve", help="start the API + dashboard").add_argument("--port", type=int, default=8000)
    d = sub.add_parser("demo-synthetic", help="write a SYNTHETIC analysis (software demo only, not real data)")
    d.add_argument("--bilstm", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.data_dir)
    try:
        return _dispatch(args, cfg)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _dispatch(args, cfg):
    if args.cmd == "fetch":
        from .data import gee
        gee.fetch_all(cfg)
    elif args.cmd == "fetch-firms":
        from .data import firms
        print(firms.fetch_all(cfg))
    elif args.cmd == "run":
        from . import pipeline
        from .data import firms, gee, inventories as inv
        raw, fires = gee.load_cube(cfg), firms.load_counts(cfg)
        inventory = {}
        if args.edgar:
            v, la, lo = inv.load_edgar_nc(args.edgar)
            inventory["edgar"] = inv.regrid_sum(v, la, lo, cfg.lats, cfg.lons, cfg["study"]["res"])
        if args.carb:
            inventory["carb"] = inv.facilities_to_grid(inv.load_facilities_csv(args.carb), cfg.lats, cfg.lons, cfg["study"]["res"])
        res = pipeline.run_analysis(raw, fires, cfg, provenance="REAL: S5P/TROPOMI + ERA5-Land + VIIRS via GEE/FIRMS",
                                    use_bilstm=args.bilstm, bilstm_epochs=args.epochs, inventory=inventory or None)
        print("wrote", pipeline.save(res, cfg))
        rep = res["summary"]["report"]
        for k in ("validation_raw", "validation_processed"):
            for p in rep[k]["problems"]:
                print("VALIDATION:", p)
    elif args.cmd == "demo-synthetic":
        from . import pipeline
        from .devtools import synthetic
        raw, fires = synthetic.make_raw(cfg)
        res = pipeline.run_analysis(raw, fires, cfg, provenance=synthetic.PROVENANCE, use_bilstm=args.bilstm, bilstm_epochs=30)
        print("wrote SYNTHETIC analysis to", pipeline.save(res, cfg, name="analysis"))
    elif args.cmd == "serve":
        import uvicorn
        from .api.app import create_app
        uvicorn.run(create_app(cfg), host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
