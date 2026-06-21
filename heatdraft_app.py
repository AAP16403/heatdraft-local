"""
HeatDraft Inverse-Design App (Standalone)

Run this file directly to launch the inverse-design application
without re-running the full ML pipeline.

Modes:
    python heatdraft_app.py                              # launch web dashboard (default)
    python heatdraft_app.py path/to/app_bundle.pkl       # custom bundle path
    python heatdraft_app.py --cli                        # legacy text-based CLI mode
    python heatdraft_app.py --port 8080                  # custom port for web mode
"""
import argparse
import sys


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
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Use legacy text-based CLI mode instead of the web dashboard",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the web dashboard (default: 5000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open the browser (web mode only)",
    )
    args = parser.parse_args()

    if args.cli:
        # Legacy text-based CLI mode
        from heatdraft import run_app

        try:
            run_app(args.bundle)
        except FileNotFoundError as exc:
            print(f"\n[ERROR] {exc}")
            print("Run the full pipeline first (python heatdraft.py <data_file>) to generate the bundle.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
    else:
        # Web dashboard mode (default)
        try:
            from heatdraft_server import create_app
        except ImportError:
            print("\n[ERROR] Flask is required for the web dashboard.")
            print("Install it with:  pip install flask")
            print("Or use --cli for the legacy text-based mode.")
            sys.exit(1)

        try:
            application = create_app(args.bundle)

            url = f"http://{args.host}:{args.port}"
            print(f"\n{'='*60}")
            print(f"  HeatDraft Dashboard running at: {url}")
            print(f"  Press Ctrl+C to stop")
            print(f"{'='*60}\n")

            if not args.no_browser:
                import threading
                import webbrowser
                threading.Timer(1.2, lambda: webbrowser.open(url)).start()

            application.run(host=args.host, port=args.port, debug=False)
        except FileNotFoundError as exc:
            print(f"\n[ERROR] {exc}")
            print("Run the full pipeline first (python heatdraft.py <data_file>) to generate the bundle.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
