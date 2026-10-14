"""
Fast multi-part concurrent downloader using HTTP Range requests.
"""
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

DATA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
ZIP_PATH = os.path.join(RAW_DIR, "battery_data_set.zip")

def download_part(url, start_byte, end_byte, part_file):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Range": f"bytes={start_byte}-{end_byte}"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        with open(part_file, "wb") as f:
            while True:
                chunk = resp.read(1024 * 64)
                if not chunk:
                    break
                f.write(chunk)

def download_parallel(url, target_path, num_workers=10):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))
        
    print(f"Total size: {total_size / (1024*1024):.2f} MB across {num_workers} threads.")
    part_size = total_size // num_workers
    tasks = []
    part_files = []
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            start_byte = i * part_size
            end_byte = (start_byte + part_size - 1) if i < num_workers - 1 else total_size - 1
            part_file = f"{target_path}.part{i}"
            part_files.append(part_file)
            tasks.append(executor.submit(download_part, url, start_byte, end_byte, part_file))
            
        for i, t in enumerate(tasks):
            t.result()
            print(f"Part {i+1}/{num_workers} finished ({time.time() - start_time:.1f}s)")
            
    # Combine parts
    print("Combining parts...")
    with open(target_path, "wb") as out_f:
        for part_file in part_files:
            with open(part_file, "rb") as pf:
                while True:
                    chunk = pf.read(1024 * 512)
                    if not chunk:
                        break
                    out_f.write(chunk)
            os.remove(part_file)
            
    print(f"All done in {time.time() - start_time:.1f}s. Saved to {target_path}")

def extract_all():
    print(f"Extracting {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(RAW_DIR)
        
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(".zip") and f != "battery_data_set.zip":
                nested = os.path.join(root, f)
                dest = os.path.join(root, os.path.splitext(f)[0])
                print(f"Unpacking nested archive: {f} -> {dest}")
                try:
                    with zipfile.ZipFile(nested, "r") as nz:
                        nz.extractall(dest)
                except Exception as e:
                    print(f"Error unpacking {f}: {e}")
    print("Extraction completely finished.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--extract-only":
        extract_all()
    else:
        download_parallel(DATA_URL, ZIP_PATH)
        extract_all()
