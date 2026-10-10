"""Command line entry point; explicit flags override JSON config values."""
import argparse
import json
import sys

from .converter import convert, inspect, ACTION_KEY
from .io import read_json, write_json
from .verify import verify, export_npz


def main(argv=None):
    parser = argparse.ArgumentParser(description="RoboTwin -> TriViewWan 1.0 training datasets (all train, real future frames)")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "convert"):
        p = sub.add_parser(command)
        p.add_argument("--source-root", required=True)
        p.add_argument("--config", help="JSON defaults; explicit command flags take precedence")
        p.add_argument("--tasks", nargs="+", help="Converted bundle directory names")
        p.add_argument("--limit", type=int, default=0, help="Episodes PER task; 0 means all")
        p.add_argument("--horizon", type=int)
        p.add_argument("--stride", type=int)
        p.add_argument("--history-frames", type=int)
        p.add_argument("--image-source", choices=("auto", "png", "jpg"))
        p.add_argument("--action-key", action="append", dest="action_keys")
        p.add_argument("--action-dim", type=int, dest="expected_action_dim")
        if command == "convert":
            p.add_argument("--output-root", required=True)
            p.add_argument("--transform", help="Trusted local .py:function; runs user Python code")
    p = sub.add_parser("verify")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--deep", action="store_true", help="Decode every stored image as well as check SHA-256")
    p.add_argument("--report", help="Save verification result JSON outside dataset manifest")
    p = sub.add_parser("export-npz")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command in ("convert", "inspect"):
            config_file = args.pop("config")
            config = read_json(config_file) if config_file else {}
            defaults = {"horizon": 48, "stride": 8, "history_frames": 9, "image_source": "auto", "action_keys": [ACTION_KEY], "expected_action_dim": 14}
            if not isinstance(config, dict) or set(config) - set(defaults):
                raise ValueError(f"Config allowed keys: {sorted(defaults)}")
            for key, default in defaults.items():
                if args.get(key) is None:
                    args[key] = config.get(key, default)
            result = (convert if command == "convert" else inspect)(**args)
        elif command == "verify":
            report = args.pop("report")
            result = verify(**args)
            if report:
                write_json(report, result)
        else:
            height, width = args.pop("height"), args.pop("width")
            if (height is None) != (width is None):
                raise ValueError("Provide both --height and --width")
            result = export_npz(**args, image_size=(height, width) if height is not None else None)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError, AttributeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
