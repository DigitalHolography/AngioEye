import multiprocessing

from ui.app import ProcessApp, main

__all__ = ["ProcessApp", "main"]


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
