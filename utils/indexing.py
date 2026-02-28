from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def next_indexed_out_dir(
    root: Path,
    *,
    flat: bool = False,
) -> Path:
    day = datetime.now().strftime("%Y%m%d")
    root.mkdir(parents=True, exist_ok=True)

    if not flat:
        day_root = root / day
        day_root.mkdir(parents=True, exist_ok=True)

        max_idx = 0
        for child in day_root.iterdir():
            if not child.is_dir():
                continue
            if re.fullmatch(r"\d{4}", child.name):
                max_idx = max(max_idx, int(child.name))

        return day_root / f"{max_idx + 1:04d}"

    pattern = re.compile(rf"^{day}_(\d{{4}})(?:_.+)?$")
    max_idx = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.fullmatch(child.name)
        if not match:
            continue
        max_idx = max(max_idx, int(match.group(1)))

    next_idx = max_idx + 1
    return root / f"{day}" / f"{next_idx:04d}"
