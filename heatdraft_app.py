"""
HeatDraft Inverse-Design App (Standalone)

Run this file directly to launch the interactive inverse-design app
without re-running the full ML pipeline.

Usage:
    python heatdraft_app.py                              # uses default bundle path
    python heatdraft_app.py path/to/app_bundle.pkl       # custom bundle path
"""
import argparse
import sys

from heatdraft import run_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HeatDraft Inverse-Design App — interactive mode using a saved model bundle."
    )
    parser.add_argument(
        "bundle",
        nargs="?",
        default="outputs/reports/app_bundle.pkl",
        help="Path to a saved app_bundle.pkl (default: outputs/reports/app_bundle.pkl)",
    )
    args = parser.parse_args()
    try:
        run_app(args.bundle)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        print("Run the full pipeline first (python heatdraft.py <data_file>) to generate the bundle.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
