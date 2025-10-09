import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_ROOT = Path(__file__).resolve().parent.parent / "data" / "img"

try:
    _RESAMPLE_STRATEGY = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    _RESAMPLE_STRATEGY = Image.LANCZOS

# Map public names onto the internal transform keys for convenience.
TRANSFORM_ALIASES: Dict[str, str] = {
    "light": "resize_28",
    "28": "resize_28",
    "28x28": "resize_28",
    "resize28": "resize_28",
    "resize_28": "resize_28",
    "medium": "resize_64",
    "64": "resize_64",
    "64x64": "resize_64",
    "resize64": "resize_64",
    "resize_64": "resize_64",
    "large": "resize_128",
    "128": "resize_128",
    "128x128": "resize_128",
    "resize128": "resize_128",
    "resize_128": "resize_128",
    "patch": "patch_64",
    "patch64": "patch_64",
    "patch_64": "patch_64",
    "crop": "patch_64",
}

# Internal definition of the supported transforms.
TRANSFORM_SPECS: Dict[str, Tuple[str, Tuple[int, int]]] = {
    "resize_28": ("resize", (28, 28)),
    "resize_64": ("resize", (64, 64)),
    "resize_128": ("resize", (128, 128)),
    "patch_64": ("patch", (64, 64)),
}

def display_artwork_image(df, row_index):
    """
    Load and display an image from a DataFrame row.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing AUTHOR and TITLE columns
    row_index : int
        Index of the row to display
    """
    # Get author and title from the row
    author = df.loc[row_index, "AUTHOR"]
    title = df.loc[row_index, "TITLE"]

    author_formatted = _format_for_filename(author)
    title_formatted = _format_for_filename(title)

    # Construct filename
    filename = _build_image_path(author_formatted, title_formatted)
    
    try:
        # Load and display the image
        img = Image.open(filename)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{author}\n{title}", fontsize=12, pad=10)
        plt.show()
        
        print(f"Loaded: {filename}")
        
    except FileNotFoundError:
        print(f"Image not found: {filename}")
        print(f"Author: {author} -> {author_formatted}")
        print(f"Title: {title} -> {title_formatted}")
    except Exception as e:
        print(f"Error loading image: {e}")

def prepare_image_column(
    df: pd.DataFrame,
    column_name: str,
    transform_type: str,
    base_dir: Path = IMAGE_ROOT,
) -> pd.DataFrame:
    """
    Load and transform artwork images for each row in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Source DataFrame containing at least the AUTHOR and TITLE columns.
    column_name : str
        Name of the new column that will store the transformed image arrays.
    transform_type : str
        Transformation identifier. Accepted values (case insensitive):
        - light, 28, 28x28, resize_28
        - medium, 64, 64x64, resize_64
        - large, 128, 128x128, resize_128
        - patch, patch_64, crop
    base_dir : pathlib.Path, optional
        Directory containing the artwork images. Defaults to the project image folder.

    Returns
    -------
    pandas.DataFrame
        A copy of the original DataFrame filtered to the rows whose images were found,
        with the new column containing float32 RGB arrays scaled to [0, 1].
    """
    if "AUTHOR" not in df.columns or "TITLE" not in df.columns:
        raise ValueError("DataFrame must contain 'AUTHOR' and 'TITLE' columns.")

    transform_key = _normalize_transform_name(transform_type)
    mode, target_size = TRANSFORM_SPECS[transform_key]

    image_store: Dict[int, np.ndarray] = {}

    for idx, row in df.iterrows():
        author = row["AUTHOR"]
        title = row["TITLE"]
        author_formatted = _format_for_filename(author)
        title_formatted = _format_for_filename(title)
        image_path = base_dir / f"{author_formatted}-{title_formatted}.jpg"

        if not image_path.exists():
            continue

        try:
            with Image.open(image_path) as img:
                image_rgb = img.convert("RGB")
                transformed = _apply_transform(image_rgb, mode, target_size)
        except (OSError, ValueError):
            continue

        array = np.asarray(transformed, dtype=np.float32) / 255.0
        image_store[idx] = array

    image_series = pd.Series(image_store, dtype=object)
    processed_df = df.loc[image_series.index].copy()
    processed_df[column_name] = image_series

    total_rows = len(df)
    processed_rows = len(processed_df)
    dropped_rows = total_rows - processed_rows

    print(
        f"Transform '{transform_type}' processed {processed_rows}/{total_rows} images."
    )
    if dropped_rows:
        print(f"Dropped {dropped_rows} rows due to missing or unreadable images.")

    return processed_df


def _normalize_transform_name(transform_type: str) -> str:
    key = transform_type.lower().replace(" ", "")
    if key not in TRANSFORM_ALIASES:
        raise ValueError(
            f"Unknown transform '{transform_type}'. Supported values: "
            f"{', '.join(sorted(TRANSFORM_ALIASES))}"
        )
    resolved = TRANSFORM_ALIASES[key]
    if resolved not in TRANSFORM_SPECS:
        raise ValueError(f"Transform '{resolved}' is not configured.")
    return resolved


def _apply_transform(image: Image.Image, mode: str, target_size: Tuple[int, int]) -> Image.Image:
    if mode == "resize":
        return image.resize(target_size, _RESAMPLE_STRATEGY)

    if mode == "patch":
        return _center_crop(image, target_size)

    raise ValueError(f"Unsupported transform mode '{mode}'.")


def _center_crop(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    target_w, target_h = target_size
    width, height = image.size

    if width < target_w or height < target_h:
        scale = max(target_w / width, target_h / height)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        image = image.resize((new_width, new_height), _RESAMPLE_STRATEGY)
        width, height = image.size

    left = (width - target_w) // 2
    top = (height - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    return image.crop((left, top, right, bottom))


def _format_for_filename(value: str) -> str:
    formatted = re.sub(r"[^a-zA-Z0-9\-]+", "_", value)
    formatted = re.sub(r"_+", "_", formatted)
    return formatted.strip("_")


def _build_image_path(author_formatted: str, title_formatted: str) -> Path:
    return IMAGE_ROOT / f"{author_formatted}-{title_formatted}.jpg"
