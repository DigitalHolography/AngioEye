from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ViewController:
    app: object

    def set_widget(self, name: str, widget) -> None:
        setattr(self.app, name, widget)
