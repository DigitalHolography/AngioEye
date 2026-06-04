from tkinter import messagebox

import multiprocessing

from ui.app import ProcessApp, main
from workflows import dispatch_workflow

__all__ = ["ProcessApp", "dispatch_workflow", "main", "messagebox"]


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
