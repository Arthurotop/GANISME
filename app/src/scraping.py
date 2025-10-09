from __future__ import annotations

import mimetypes
import os
import re
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import Session


def _sanitize_filename(value: str) -> str:
    """Collapse non-alphanumerics into underscores and strip extras."""
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_")
    return cleaned or "untitled"


def _extract_letter_from_url(url: str) -> Optional[str]:
    """Return the first path fragment under /html/ (e.g., 'a' in /html/a/...)."""
    match = re.search(r"/html/([a-z0-9])/", url.lower())
    if match:
        return match.group(1)
    return None


def download_art_images(
    df: pd.DataFrame,
    url_column: str = "URL",
    author_column: str = "AUTHOR",
    title_column: str = "TITLE",
    output_dir: str | Path = "data/img",
    session: Optional[Session] = None,
    timeout: int = 20,
    start_letter: Optional[str] = None,
) -> list[Path]:
    """
    For each row in `df`, download images whose parent `<a>` tag has `/art/` in the href.
    Images are saved under `output_dir` as `<AUTHOR>-<TITLE><suffix>.<ext>`.

    If `start_letter` is provided, rows are skipped until the URL contains a matching
    `/html/<letter>/` segment (case-insensitive), allowing the download to resume from a
    specific letter page.
    Returns the list of stored image paths.
    """
    if url_column not in df:
        raise ValueError(f"DataFrame is missing the required column '{url_column}'")

    normalized_letter: Optional[str] = None
    if start_letter is not None:
        if not isinstance(start_letter, str):
            raise TypeError("start_letter must be a string when provided")
        candidate = start_letter.strip().lower()
        if len(candidate) != 1 or not candidate.isalpha():
            raise ValueError("start_letter must be a single alphabetical character")
        normalized_letter = candidate

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    session = session or requests.Session()
    saved_paths: list[Path] = []

    for row in df.itertuples(index=False):
        page_url = getattr(row, url_column, None)
        author = _sanitize_filename(str(getattr(row, author_column, "unknown")))
        title = _sanitize_filename(str(getattr(row, title_column, "untitled")))

        if not page_url:
            continue

        if normalized_letter is not None:
            url_letter = _extract_letter_from_url(str(page_url))
            if url_letter is None:
                continue
            if url_letter < normalized_letter:
                continue

        response = session.get(page_url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        art_links: Iterable = soup.find_all(
            "a", href=lambda href: isinstance(href, str) and "/art/" in href.lower()
        )

        for idx, tag in enumerate(art_links, start=1):
            href = tag.get("href")
            if not href:
                continue

            image_url = urljoin(page_url, href)
            image_resp = session.get(image_url, timeout=timeout)
            image_resp.raise_for_status()

            ext = os.path.splitext(urlparse(image_url).path)[1]
            if not ext:
                mime = image_resp.headers.get("Content-Type", "").split(";")[0].strip()
                ext = mimetypes.guess_extension(mime) or ".jpg"

            suffix = f"-{idx:02d}" if idx > 1 else ""
            filename = f"{author}-{title}{suffix}{ext}"
            destination = output_path / filename

            with destination.open("wb") as fh:
                fh.write(image_resp.content)

            saved_paths.append(destination)

    return saved_paths
