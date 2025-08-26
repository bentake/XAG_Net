from pathlib import Path
import glob
import cv2
import numpy as np
import random
import re
from typing import List, Tuple


def extract_identifier(filename: str):
    """
    Extract (subject_id, slice_number) from file names of form: TD01_S1_XXX.png
    Returns (subject_id, slice_num) or (None, None) if not matched.
    """
    m = re.search(r"(TD\d+_S\d+).*?_(\d+)\.png$", filename)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def find_paths(images_glob: str, masks_glob: str) -> Tuple[List[str], List[str]]:
    image_paths = sorted(glob.glob(images_glob))
    mask_paths = sorted(glob.glob(masks_glob))
    return image_paths, mask_paths


def _load_gray(path: str, size: Tuple[int, int]) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    arr = cv2.resize(arr, size).astype(np.float32) / 255.0
    return arr


def load_and_prepare_data(
    image_paths: List[str],
    mask_paths: List[str],
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build 2.5D stacks (prev, current, next) and pair with corresponding masks.
    Skips first/last slices for which neighbors are missing.
    """
    # sort by (subject, slice) for stable grouping
    image_paths = sorted(image_paths, key=lambda p: extract_identifier(Path(p).name))
    mask_paths = sorted(mask_paths, key=lambda p: extract_identifier(Path(p).name))

    # group by subject
    subjects_imgs = {}
    subjects_masks = {}
    for p in image_paths:
        sid, _ = extract_identifier(Path(p).name)
        if sid:
            subjects_imgs.setdefault(sid, []).append(p)
    for p in mask_paths:
        sid, _ = extract_identifier(Path(p).name)
        if sid:
            subjects_masks.setdefault(sid, []).append(p)

    X, y = [], []
    for sid in sorted(subjects_imgs.keys()):
        imgs = sorted(subjects_imgs[sid], key=lambda p: extract_identifier(Path(p).name)[1])
        msks = sorted(subjects_masks.get(sid, []), key=lambda p: extract_identifier(Path(p).name)[1])
        for i in range(1, len(imgs) - 1):
            if i >= len(msks):
                continue
            prev_img = _load_gray(imgs[i - 1], target_size)
            curr_img = _load_gray(imgs[i], target_size)
            next_img = _load_gray(imgs[i + 1], target_size)
            stack = np.stack([prev_img, curr_img, next_img], axis=-1)  # (H, W, 3)
            mask = _load_gray(msks[i], target_size)
            mask = (mask > 0.5).astype(np.float32)  # binarize
            X.append(stack)
            y.append(mask[..., None])  # add channel dim (H, W, 1)

    if not X:
        raise RuntimeError("No samples prepared. Check your file naming and glob patterns.")
    return np.asarray(X), np.asarray(y)


def load_dataset_from_dir(
    root_dir: Path,
    images_glob: str = "images/*.png",
    masks_glob: str = "masks/*.png",
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    root_dir = Path(root_dir)
    img_glob = str(root_dir / images_glob)
    msk_glob = str(root_dir / masks_glob)
    image_paths, mask_paths = find_paths(img_glob, msk_glob)
    return load_and_prepare_data(image_paths, mask_paths, target_size=target_size)


def preview_random_pair(
    image_paths: List[str],
    mask_paths: List[str],
    target_size: Tuple[int, int] = (256, 256),
):
    import matplotlib.pyplot as plt
    if not image_paths or not mask_paths:
        print("No images or masks found.")
        return
    idx = random.randint(0, min(len(image_paths), len(mask_paths)) - 1)
    img = _load_gray(image_paths[idx], target_size)
    msk = _load_gray(mask_paths[idx], target_size)
    msk = (msk > 0.5).astype(np.float32)
    import os
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title(f"Image: {os.path.basename(image_paths[idx])}"); ax[0].axis("off")
    ax[1].imshow(msk, cmap="gray"); ax[1].set_title(f"Mask: {os.path.basename(mask_paths[idx])}"); ax[1].axis("off")
    plt.tight_layout(); plt.show()
