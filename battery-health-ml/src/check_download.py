"""
Check and download NASA Battery Dataset from official S3 repository.
"""
import os
import sys
import urllib.request
import zipfile

DATA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))

def check_url():
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print("Status:", resp.status)
            size_mb = int(resp.headers.get("Content-Length", 0)) / (1024 * 1024)
            print(f"Content-Length: {size_mb:.2f} MB")
            return True, size_mb
    except Exception as e:
        print("Error checking URL:", e)
        return False, 0

if __name__ == "__main__":
    check_url()
