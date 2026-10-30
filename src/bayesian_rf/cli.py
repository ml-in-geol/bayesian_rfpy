"""Small operational CLI; scientific configuration lives in Python examples."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rfpy", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build-cps", help="Build the bundled hrftn96 source in the user cache")
    ingest = commands.add_parser("import-insight", help="Copy available LF-family waveform products")
    ingest.add_argument("--catalog", required=True)
    ingest.add_argument("--deglitched", required=True)
    ingest.add_argument("--denoised", required=True, help="Directory of extensionless Dahmen event files")
    ingest.add_argument("--output", default="insight_data")
    args = parser.parse_args(argv)
    if args.command == "build-cps":
        from .cps import build_hrftn96

        print(build_hrftn96())
    else:
        from .data import import_insight_data

        manifest = import_insight_data(args.catalog, args.deglitched, args.denoised, args.output)
        summary = {
            kind: sum(e["products"][kind]["status"] == "available" for e in manifest["events"])
            for kind in ("deglitched", "denoised")
        }
        summary["catalogued_events"] = sum(e["in_mqs_v14_lf_family"] for e in manifest["events"])
        summary["total_events"] = len(manifest["events"])
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
