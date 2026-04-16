#!/usr/bin/env python3
"""Download whl files from PyPI for Python 3.12 + linux x86_64."""

import json
import os
import subprocess
import sys

DEST = "/mnt/llm-train-5p/shenzhennan/pkgs_perl"

# (package_name, version_constraint_or_None)
# None means latest
PACKAGES = [
    ("trl", None),
    ("peft", None),
    ("fire", None),
    ("math-verify", None),
    ("liger-kernel", None),
    ("deepspeed", None),
]

PYTHON_TAG = "cp312"
PLATFORM_TAGS = ["manylinux_2_17_x86_64", "manylinux2014_x86_64", "linux_x86_64", "any"]


def fetch_json(url):
    """Fetch JSON from URL using curl (supports SOCKS proxy via env vars)."""
    result = subprocess.run(
        ["curl", "-sL", url],
        capture_output=True, text=True, timeout=30
    )
    return json.loads(result.stdout)


def pick_whl(urls, pkg_name):
    """Pick best wheel for cp312 + linux x86_64, fallback to py3-none-any."""
    candidates = [u for u in urls if u["filename"].endswith(".whl")]

    # prefer cp312 platform-specific
    for u in candidates:
        fn = u["filename"]
        if PYTHON_TAG in fn and any(pt in fn for pt in PLATFORM_TAGS[:3]):
            return u

    # fallback: py3-none-any (pure python)
    for u in candidates:
        fn = u["filename"]
        if "py3-none-any" in fn:
            return u

    # fallback: any cp312
    for u in candidates:
        if PYTHON_TAG in u["filename"]:
            return u

    return None


def download_file(url, dest_path):
    """Download file using curl."""
    print(f"  Downloading {os.path.basename(dest_path)} ...")
    subprocess.run(
        ["curl", "-sL", "-o", dest_path, url],
        timeout=300, check=True
    )


def main():
    os.makedirs(DEST, exist_ok=True)

    for pkg_name, version in PACKAGES:
        print(f"\n=== {pkg_name} ===")

        if version:
            api_url = f"https://pypi.org/pypi/{pkg_name}/{version}/json"
        else:
            api_url = f"https://pypi.org/pypi/{pkg_name}/json"

        try:
            data = fetch_json(api_url)
        except Exception as e:
            print(f"  FAILED to fetch metadata: {e}")
            continue

        ver = data["info"]["version"]
        print(f"  Latest version: {ver}")

        release_urls = data["urls"]
        whl = pick_whl(release_urls, pkg_name)

        if whl:
            dest_path = os.path.join(DEST, whl["filename"])
            if os.path.exists(dest_path):
                print(f"  Already exists: {whl['filename']}")
            else:
                download_file(whl["url"], dest_path)
                size_mb = os.path.getsize(dest_path) / (1024 * 1024)
                print(f"  Done: {whl['filename']} ({size_mb:.1f} MB)")
        else:
            print(f"  WARNING: No suitable wheel found for {pkg_name} {ver}")
            print(f"  Available files:")
            for u in release_urls[:5]:
                print(f"    {u['filename']}")

    print(f"\n=== Downloaded files ===")
    for f in sorted(os.listdir(DEST)):
        size = os.path.getsize(os.path.join(DEST, f)) / (1024*1024)
        print(f"  {f} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
