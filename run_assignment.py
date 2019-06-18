import argparse
from assignment import run_assignment


def get_command_line_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Get the args.",
        epilog=""" """,
    )

    parser.add_argument(
        "--project_config",
        required=True,
        help="Input project configuration.",
    )

    return parser.parse_args()


def main(args):
    run_assignment(
        args.project_config,
    )


if __name__ == "__main__":
    main(get_command_line_args())