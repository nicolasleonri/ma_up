from dataclasses import dataclass


@dataclass

class Page:

    newspaper: str

    date: str

    edition: str

    page_number: int

    page_url: str

    image_url: str | None = None

    image_path: str | None = None