"""Application entrypoint for the winch + scales + blob logger GUI."""

from __future__ import annotations

import tkinter as tk

from app_ui import WinchUI


def main():
    root = tk.Tk()
    app = WinchUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.close_all(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
