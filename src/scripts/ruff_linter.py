import subprocess
import sys
import argparse


def run_ruff(fix=False):
    """Runs Ruff check and format."""

    check_cmd = ["ruff", "check", "."]
    format_cmd = ["ruff", "format", "."]

    if fix:
        print("Applying auto-fixes...")
        check_cmd.append("--fix")
    else:
        print("Checking code...")
        format_cmd.insert(2, "--check")  # adds --check to 'ruff format .'

    try:
        res_check = subprocess.run(check_cmd)
        res_format = subprocess.run(format_cmd)

        if res_check.returncode != 0 or res_format.returncode != 0:
            print("\nErrors found. Run the script with --fix to resolve style issues.")
            sys.exit(1)

        print("\nCode looks great!")
        sys.exit(0)

    except FileNotFoundError:
        print("Error: 'ruff' not found. Please run: pip install ruff")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AngioEye Linting Tool")
    parser.add_argument("--fix", action="store_true", help="Automatically fix issues")
    args = parser.parse_args()
    run_ruff(fix=args.fix)


if __name__ == "__main__":
    main()
