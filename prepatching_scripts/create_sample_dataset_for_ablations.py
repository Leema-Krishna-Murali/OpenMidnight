# This script builds a text manifest of TCGA patch samples for ablation runs by sampling target physical
# resolutions (MPP) from a truncated Fisk distribution, selecting the pyramid level whose native MPP is
# closest to the target, and reading a region sized to up/downsample back to 224x224. It reads base MPP
# metadata, enumerates pyramid levels, draws target MPPs in [0.05, 50.0], computes read coordinates, and
# keeps a patch only if the HSV tissue filter passes. Each accepted line records slide path, level-0
# coordinates, pyramid level, and the target MPPs so downstream parquet generation can re-read and rescale
# consistently.
import cv2
import random
from pathlib import Path
from openslide import OpenSlide
import numpy as np
import scipy.stats as st
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import shutil
import time

data_root = Path("/data/TCGA")
output_filename = "/data/TCGA/sample_dataset_224x224_fisk.txt"
patch_size = 224
max_tries_per_level = 500
max_patches = 25_000_000
patches_per_level = 10
seed = 0
workers = 10
num_shards = 8
dist = st.fisk(c=2.0, scale=np.sqrt(3.0))  # mode (peak) is exactly 1
target_mpp_cdf_min = dist.cdf(0.05)
target_mpp_cdf_max = dist.cdf(50.0)
MPP_X_KEY = "openslide.mpp-x"
MPP_Y_KEY = "openslide.mpp-y"

def hsv(tile_rgb):
    """
    Checks if a given tile has a high concentration of tissue based on an HSV mask.
    """
    tile = np.array(tile_rgb)
    # Convert from RGB to HSV color space
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    min_ratio = .4 # lowered from .6

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

random.seed(seed)
svs_files = sorted(str(path) for path in data_root.rglob("*.svs"))
random.shuffle(svs_files)
if not svs_files:
    raise RuntimeError(f"No SVS files found under {data_root}")

def sample_target_mpp(rng):
    return float(dist.ppf(rng.uniform(target_mpp_cdf_min, target_mpp_cdf_max)))

def sample_slide(args):
    path, slide_idx, pass_idx = args
    slide_seed = seed + pass_idx * 10_000 + slide_idx
    random.seed(slide_seed)
    rng = np.random.default_rng(slide_seed)
    image = OpenSlide(path)
    collected_lines = []

    props = image.properties
    if MPP_X_KEY not in props or MPP_Y_KEY not in props:
        image.close()
        print(f"Skipping slide without MPP metadata: {path}")
        return []

    base_mpp_x = float(props[MPP_X_KEY])
    base_mpp_y = float(props[MPP_Y_KEY])

    level_info = []
    for level in range(image.level_count):
        downsample = float(image.level_downsamples[level])
        level_mpp_x = base_mpp_x * downsample
        level_mpp_y = base_mpp_y * downsample
        width, height = image.level_dimensions[level]
        level_info.append(
            (level, downsample, level_mpp_x, level_mpp_y, width, height, max(level_mpp_x, level_mpp_y))
        )
    if not level_info:
        image.close()
        return []

    target_for_slide = patches_per_level * image.level_count
    max_tries = max_tries_per_level * image.level_count
    collected = 0
    tries = 0
    while collected < target_for_slide and tries < max_tries:
        tries += 1
        target_mpp = sample_target_mpp(rng)
        chosen = None
        chosen_read_w = None
        chosen_read_h = None
        best_diff = None
        for info in level_info:
            level, downsample, level_mpp_x, level_mpp_y, width, height, level_mpp_max = info
            read_w = max(1, int(round(patch_size * (target_mpp / level_mpp_x))))
            read_h = max(1, int(round(patch_size * (target_mpp / level_mpp_y))))
            if read_w > width or read_h > height:
                continue
            diff = abs(level_mpp_max - target_mpp)
            if best_diff is None or diff < best_diff:
                chosen = info
                chosen_read_w = read_w
                chosen_read_h = read_h
                best_diff = diff
        if chosen is None:
            continue
        level, downsample, level_mpp_x, level_mpp_y, width, height, _ = chosen
        read_w = chosen_read_w
        read_h = chosen_read_h
        x_level = random.randint(0, width - read_w)
        y_level = random.randint(0, height - read_h)
        x = int(round(x_level * downsample))
        y = int(round(y_level * downsample))
        patch = image.read_region((x, y), level=level, size=(read_w, read_h)).convert("RGB")
        if read_w != patch_size or read_h != patch_size:
            if random.random() < 0.5:
                patch = cv2.resize(np.array(patch), (patch_size, patch_size), interpolation=cv2.INTER_AREA)
            else:
                patch = patch.resize((patch_size, patch_size), resample=Image.BICUBIC)
        res = hsv(patch)
        if res is not None:
            collected_lines.append(f"{path} {x} {y} {level} {target_mpp} {target_mpp}\n")
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
