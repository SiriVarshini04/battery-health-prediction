"""
Download and extract the NASA Li-ion Battery Aging Dataset (PCoE).
"""
import os
import sys
import time
import urllib.request
import zipfile

DATA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
ZIP_PATH = os.path.join(RAW_DIR, "battery_data_set.zip")

def download_file(url, target_path):
    print(f"Connecting to {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))
        total_mb = total_size / (1024 * 1024)
        print(f"Total file size: {total_mb:.2f} MB")
        
        start_time = time.time()
        downloaded = 0
        chunk_size = 1024 * 512 # 512 KB
        
        with open(target_path, "wb") as f:
            last_report = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_report >= 3 or downloaded == total_size:
                    percent = (downloaded / total_size) * 100 if total_size else 0
                    speed = (downloaded / (1024 * 1024)) / max(0.01, now - start_time)
                    print(f"Progress: {downloaded / (1024 * 1024):.1f}/{total_mb:.1f} MB ({percent:.1f}%) at {speed:.2f} MB/s")
                    last_report = now
                    
    print(f"Download complete: {target_path}")

def extract_dataset(zip_path, extract_dir):
    print(f"Extracting {zip_path} into {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        # Check files inside
        namelist = z.namelist()
        print(f"Found {len(namelist)} items in archive.")
        z.extractall(extract_dir)
    print("Archive extraction complete.")

    # Check if there are nested zips (e.g. 1. BatteryAgingARC-FY08Q4.zip)
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.endswith(".zip") and f != "battery_data_set.zip":
                nested_zip = os.path.join(root, f)
                nested_dest = os.path.join(root, os.path.splitext(f)[0])
                print(f"Unpacking nested archive: {f} -> {nested_dest}")
                try:
                    with zipfile.ZipFile(nested_zip, "r") as nz:
                        nz.extractall(nested_dest)
                except Exception as ex:
                    print(f"Warning unpacking {f}: {ex}")

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        download_file(DATA_URL, ZIP_PATH)
    else:
        print(f"Archive already exists at {ZIP_PATH} ({os.path.getsize(ZIP_PATH)/(1024*1024):.2f} MB).")
        
    extract_dataset(ZIP_PATH, RAW_DIR)

if __name__ == "__main__":
    main()
