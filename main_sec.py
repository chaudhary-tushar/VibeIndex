import sys


def main():
    """
    This is the main entry point for the application.
    It checks the command-line arguments to decide whether to run the API or the CLI.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        from app import main as app_main
        # Remove the 'api' argument before passing to the app
        sys.argv.pop(1)
        app_main()
    else:
        from cli import main as cli_main
        cli_main()


if __name__ == "__main__":
    main()
