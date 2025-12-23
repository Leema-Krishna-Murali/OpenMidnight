import cv2
import random
from pathlib import Path
from openslide import OpenSlide
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing as mp
import shutil
import time

data_root = Path("/data/TCGA")
output_filename = "/data/TCGA/sample_dataset_25M_256x256.txt"
patch_size = 256 #224
max_tries = 1000
max_patches = 25_000_000
patches_per_level = 100
seed = 0
workers = 10
num_shards = 8

def hsv(tile_rgb):
    """
    Checks if a given tile has a high concentration of tissue based on an HSV mask.
    """
    tile = np.array(tile_rgb)
    # Convert from RGB to HSV color space
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    min_ratio = .6

    # Define the color range for tissue in HSV
    lower_bound = np.array([90, 8, 103])
    upper_bound = np.array([180, 255, 255])

    # Create a mask for the specified color range
    mask = cv2.inRange(tile, lower_bound, upper_bound)

    # Calculate the ratio of tissue pixels
    ratio = np.count_nonzero(mask) / mask.size
    
    if ratio > min_ratio:
        return tile_rgb
    else:
        return None

svs_files = sorted(str(path) for path in data_root.rglob("*.svs"))
if not svs_files:
    raise RuntimeError(f"No SVS files found under {data_root}")

def sample_slide(args):
    path, slide_idx, pass_idx = args
    random.seed(seed + pass_idx * 10_000 + slide_idx)
    image = OpenSlide(path)
    collected_lines = []
    width, height = image.level_dimensions[0]
    if width < patch_size or height < patch_size:
        image.close()
        return []
    for level in range(0, image.level_count):
        collected = 0
        tries = 0
        while collected < patches_per_level and tries < max_tries:
            tries += 1

            # Randomly select a top-left coordinate for the patch
            x = random.randint(0, width - patch_size)
            y = random.randint(0, height - patch_size)

            # Read the region from the slide
            patch = image.read_region((x, y), level=level, size=(patch_size, patch_size))

            # Check if the patch contains enough tissue
            res = hsv(patch)

            if res is not None:
                # If the patch is valid, record its info for output
                collected_lines.append(f"{path} {x} {y} {level}\n")
                collected += 1
    image.close()
    return collected_lines

def part_filename(shard_idx):
    return str(Path(output_filename).with_suffix(f".part{shard_idx}.txt"))

def run_shard(shard_idx, progress_counts):
    shard_files = [(path, idx) for idx, path in enumerate(svs_files) if idx % num_shards == shard_idx]
    target_patches = max_patches // num_shards
    if shard_idx < max_patches % num_shards:
        target_patches += 1
    with open(part_filename(shard_idx), 'w') as f:
        if not shard_files or target_patches == 0:
            return
        patches_written = 0
        pass_idx = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            while patches_written < target_patches:
                patches_before = patches_written
                tasks = ((path, idx, pass_idx) for path, idx in shard_files)
                for lines in executor.map(sample_slide, tasks):
                    remaining = target_patches - patches_written
                    if remaining <= 0:
                        break
                    if lines:
                        count = len(lines)
                        if count > remaining:
                            count = remaining
                            lines = lines[:count]
                        f.writelines(lines)
                        patches_written += count
                        progress_counts[shard_idx] = patches_written
                    if patches_written >= target_patches:
                        break
                pass_idx += 1
                if patches_written == patches_before:
                    break
        progress_counts[shard_idx] = patches_written

def main():
    print(
        f"Starting patch sampling (target: {max_patches} patches, shards: {num_shards}). "
        f"Parts will be saved to {output_filename}.partN.txt"
    )
    progress_counts = mp.Array('Q', num_shards, lock=False)
    processes = []
    for shard_idx in range(num_shards):
        process = mp.Process(target=run_shard, args=(shard_idx, progress_counts))
        process.start()
        processes.append(process)
    progress = tqdm(total=max_patches, desc="Patches collected")
    last_total = 0
    while any(process.is_alive() for process in processes):
        total = sum(progress_counts)
        delta = total - last_total
        if delta:
            progress.update(delta)
            last_total = total
        time.sleep(1)
    for process in processes:
        process.join()
    total = sum(progress_counts)
    delta = total - last_total
    if delta:
        progress.update(delta)
    progress.close()
    with open(output_filename, 'w') as out_f:
        for shard_idx in range(num_shards):
            with open(part_filename(shard_idx), 'r') as in_f:
                shutil.copyfileobj(in_f, out_f)
    with open(output_filename, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open(output_filename, 'w') as f:
        f.writelines(lines)
    print("Done")

if __name__ == "__main__":
    main()
