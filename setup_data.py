"""准备 SECOM 数据集：下载（或复用本地文件）并解压到 data/raw/。

用法：
    python setup_data.py                    # 自动下载
    python setup_data.py --zip D:/path/secom.zip   # 使用本地已下载的压缩包
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from zhixi.config import RAW_DIR, SECOM_URL

REQUIRED = ["secom.data", "secom_labels.data"]


def already_ready(raw: Path) -> bool:
    return all((raw / f).exists() for f in REQUIRED)


def extract(zip_path: Path, raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            flat = Path(name).name
            if not flat:
                continue
            with zf.open(name) as src, open(raw / flat, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", dest="zip_path", default="", help="本地已下载的 secom.zip 路径")
    ap.add_argument("--force", action="store_true", help="已存在时仍然重新解压")
    args = ap.parse_args()

    raw = Path(RAW_DIR)
    if already_ready(raw) and not args.force:
        print(f"数据集已就绪：{raw}")
        return

    if args.zip_path:
        zip_path = Path(args.zip_path)
        print(f"使用本地压缩包：{zip_path}")
    else:
        zip_path = raw.parent / "secom.zip"
        print(f"正在下载：{SECOM_URL}")
        urllib.request.urlretrieve(SECOM_URL, zip_path)
        print(f"已保存到：{zip_path}")

    extract(zip_path, raw)
    for f in REQUIRED:
        p = raw / f
        print(f"  {'✓' if p.exists() else '✗'} {p} ({p.stat().st_size if p.exists() else 0} bytes)")
    print("完成。")


if __name__ == "__main__":
    main()
