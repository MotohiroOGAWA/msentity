from __future__ import annotations

import argparse

from msentity.cli.commands._common import add_input_dataset_arguments


def setup_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "gui",
        help="Open a mass spectrum dataset in the graphical viewer.",
    )
    add_input_dataset_arguments(parser)
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a Gradio share URL.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open a web browser.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    try:
        from msentity.cli.gui.app import run_gui
    except ImportError as exc:
        raise SystemExit(
            "msentity GUI dependencies are not installed.\n\n"
            "Install them with:\n"
            '  pip install "msentity[gui]"\n'
        ) from exc

    try:
        run_gui(
            args.input_file,
            file_type=args.file_type,
            spec_id_prefix=args.spec_id_prefix,
            host=args.host,
            port=args.port,
            share=args.share,
            inbrowser=not args.no_browser,
        )
    except ImportError as exc:
        raise SystemExit(
            "msentity GUI dependencies are not installed.\n\n"
            "Install them with:\n"
            '  pip install "msentity[gui]"\n'
        ) from exc
