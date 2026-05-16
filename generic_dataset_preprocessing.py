import re
import warnings
import random
from pathlib import Path
from collections import defaultdict
from skimage.transform import resize

from skimage.exposure import equalize_adapthist
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as mpatches
from PIL import Image
import tomli

warnings.filterwarnings("ignore")

with open("params.toml", "rb") as f:
    cfg = tomli.load(f)

PATCH_H = cfg['generic_dataset']['PATCH_H']
PATCH_W = cfg['generic_dataset']['PATCH_W']
STRIDE = cfg['generic_dataset']['STRIDE']
SEED = cfg['generic_dataset']['SEED']
DATASET_ROOT_DIR   = cfg['generic_dataset']['DATASET_ROOT_DIR']
DATASET_OUTPUT_DIR = cfg['generic_dataset']['DATASET_OUTPUT_DIR']
NEG_RATIO = cfg['generic_dataset']['NEG_RATIO']   # negativi totali per ogni positivo
HARD_NEG_RATIO = cfg['generic_dataset']['HARD_NEG_RATIO']   #frazione dei negativi estratti con sliding window
TRAIN_RATIO    = cfg['generic_dataset']['TRAIN_RATIO']

CLAHE_FLAG         = cfg['generic_dataset']['CLAHE_FLAG']
SHOW_FLAG          = cfg['generic_dataset']['SHOW_FLAG']
FIXED_N_FLAG       = cfg['generic_dataset']['FIXED_N_FLAG']
N_PATCH_FIXED      = cfg['generic_dataset']['N_PATCH_FIXED']

DEFAULT_HALF_BOX_PX = cfg['generic_dataset']['DEFAULT_HALF_BOX_PX']

# True = SAMPLE mode, False = GRID mode
SAMPLE_FLAG = cfg['generic_dataset']['SAMPLE_FLAG']

#Flag demo
#Se TRUE elabora solo il primo soggetto (DEMO_SUBJECT_ID), salva patch positive e immagini con bounding box GT; NON esegue il preprocessing completo.
#Se FALSE esegue il preprocessing completo su tutti i soggetti.
DEMO_FLAG        = cfg['generic_dataset']['DEMO_FLAG']
DEMO_SUBJECT_ID  = cfg['generic_dataset']['DEMO_SUBJECT_ID']
DEMO_N_PATCHES   = cfg['generic_dataset']['DEMO_N_PATCHES'] 


def debug_show_patches(patches, labels, n=10):
    for i in range(min(n, len(patches))):
        plt.imshow(patches[i], cmap='gray', vmin=0, vmax=1)
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
        plt.show()

def load_image(image_path: Path) -> np.ndarray:
    img = Image.open(str(image_path))
    if img.mode != 'L':
        img = img.convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def get_subject_images(subject_image_dir: Path) -> list:
    return sorted(subject_image_dir.glob("*.png"))

def parse_labels_file(filepath: Path) -> list:
    entries = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = re.split(r'[\s,\t]+', line)
                if len(parts) >= 2:
                    try:
                        name  = Path(parts[0]).stem
                        label = int(parts[1])
                        entries.append((name, label))
                    except ValueError:
                        continue
    except Exception:
        pass
    return entries


def parse_positions_file(filepath: Path) -> dict:
    positions = {}
    if not filepath or not filepath.exists():
        return positions
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = re.split(r'[\s,\t]+', line)
                if len(parts) >= 3:
                    try:
                        name   = Path(parts[0]).stem
                        row    = int(parts[1])
                        col    = int(parts[2])
                        half_h = int(parts[3]) if len(parts) >= 4 else DEFAULT_HALF_BOX_PX
                        half_w = int(parts[4]) if len(parts) >= 5 else DEFAULT_HALF_BOX_PX
                        positions[name] = (row, col, half_h, half_w)
                    except ValueError:
                        continue
    except Exception:
        pass
    return positions


def find_annotation_files(anno_root: Path, subject_id: str):
    subject_anno = anno_root / subject_id
    if not subject_anno.exists():
        for d in anno_root.iterdir():
            if d.is_dir() and d.name.lower() == subject_id.lower():
                subject_anno = d
                break
        else:
            return None, None
    labels_f    = subject_anno / "labels.txt"
    positions_f = subject_anno / "positions.txt"
    return (labels_f    if labels_f.exists()    else None,
            positions_f if positions_f.exists() else None)


def load_subject_annotations(anno_root: Path, subject_id: str, image_files: list):
    labels_f, positions_f = find_annotation_files(anno_root, subject_id)

    if labels_f is None:
        print(f"  [SKIP] {subject_id}: labels.txt non trovato")
        return None

    label_entries = parse_labels_file(labels_f)
    positions     = parse_positions_file(positions_f) if positions_f else {}

    if len(label_entries) != len(image_files):
        print(f"[WARN] {subject_id}: righe in labels.txt ({len(label_entries)}) non coincidono con le immagini trovate ({len(image_files)}) -> skip")
        return None

    annotations = []
    for img_path, (name, label) in zip(image_files, label_entries):
        stem = img_path.stem
        if label == 1 and stem in positions:
            row, col, half_h, half_w = positions[stem]
        elif label == 1:
            row, col    = None, None
            half_h, half_w = DEFAULT_HALF_BOX_PX, DEFAULT_HALF_BOX_PX
        else:
            row, col, half_h, half_w = None, None, None, None
        annotations.append((label, row, col, half_h, half_w))

    return annotations

def apply_clahe(patch: np.ndarray) -> np.ndarray:
    return equalize_adapthist(patch.astype(np.float64), clip_limit=0.03).astype(np.float32)


def extract_patch_2d(img: np.ndarray, center_r: int, center_c: int, patch_h: int = PATCH_H, patch_w: int = PATCH_W) -> np.ndarray:
    H, W   = img.shape
    half_h = patch_h // 2
    half_w = patch_w // 2

    pad_top    = max(0, half_h - center_r)
    pad_bottom = max(0, center_r + half_h - H + 1)
    pad_left   = max(0, half_w - center_c)
    pad_right  = max(0, center_c + half_w - W + 1)

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)),
                     mode='constant', constant_values=0.0)
        center_r += pad_top
        center_c += pad_left

    patch = img[center_r - half_h: center_r - half_h + patch_h,
                center_c - half_w: center_c - half_w + patch_w]

    if patch.shape != (patch_h, patch_w):
        patch = resize(patch, (patch_h, patch_w), anti_aliasing=True, mode='reflect')

    return patch.astype(np.float64)


def extract_grid_patches(img: np.ndarray, patch_h: int, patch_w: int, stride: int):
    H, W = img.shape
    patches, coords = [], []
    for r in range(0, H - patch_h + 1, stride):
        for c in range(0, W - patch_w + 1, stride):
            patches.append(img[r:r + patch_h, c:c + patch_w].astype(np.float64))
            coords.append((r, c))
    return patches, coords


def extract_fixed_number_grid(img: np.ndarray, patch_h: int, patch_w: int, stride: int, max_patches: int):
    patches, coords = extract_grid_patches(img, patch_h, patch_w, stride)
    if len(patches) > max_patches:
        idx     = np.random.choice(len(patches), max_patches, replace=False)
        patches = [patches[i] for i in idx]
        coords  = [coords[i]  for i in idx]
    return patches, coords

def build_gt_box(img_shape, row, col, half_h, half_w) -> dict:
    H, W = img_shape
    cy   = row if row is not None else H // 2
    cx   = col if col is not None else W // 2
    return dict(
        cy=cy, cx=cx,
        r0=max(0, cy - half_h), c0=max(0, cx - half_w),
        r1=min(H, cy + half_h), c1=min(W, cx + half_w),
        half_h=half_h, half_w=half_w
    )


def patch_overlaps_box(patch_r, patch_c, patch_h, patch_w, box) -> bool:
    center_r = patch_r + patch_h // 2
    center_c = patch_c + patch_w // 2
    return (box['r0'] <= center_r < box['r1'] and box['c0'] <= center_c < box['c1'])

def extract_positive_patches(image_files, annotations, patch_h, patch_w, apply_clahe_flag):
    patches, centroids = [], []
    for img_path, (label, row, col, half_h, half_w) in zip(image_files, annotations):
        if label != 1:
            continue
        img  = load_image(img_path)
        H, W = img.shape
        cy   = row if row is not None else H // 2
        cx   = col if col is not None else W // 2
        patch = extract_patch_2d(img, cy, cx, patch_h, patch_w)
        if apply_clahe_flag:
            patch = apply_clahe(patch)
        patches.append(patch)
        centroids.append((img_path.stem, cy, cx))
    return patches, centroids

def extract_demo_positive_patches(image_files, annotations, patch_h, patch_w, apply_clahe_flag,n_target=10, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)

    base_patches, base_centroids = extract_positive_patches(image_files, annotations, patch_h, patch_w, apply_clahe_flag)
    if not base_patches:
        raise RuntimeError("Nessuna patch positiva estratta.")

    print(f"Patch base estratte: {len(base_patches)} (target={n_target})")

    patches   = list(base_patches)
    centroids = list(base_centroids) 

    if len(patches) >= n_target:
        idx = rng.choice(len(patches), size=n_target, replace=False)
        patches = [patches[i] for i in idx]
        print(f"    → campionate {n_target} patch dalle {len(base_patches)} disponibili")
        return patches, [1.0] * n_target

    JITTER = 3
    half_h, half_w = patch_h // 2, patch_w // 2
    attempts, max_attempts = 0, (n_target - len(patches)) * 50

    img_dict = {p.stem: load_image(p) for p in image_files}

    while len(patches) < n_target and attempts < max_attempts:
        attempts += 1
        base_idx      = int(rng.integers(0, len(base_centroids)))
        stem, cy, cx  = base_centroids[base_idx]
        img           = img_dict[stem]
        H, W          = img.shape
        ny = cy + int(rng.integers(-JITTER, JITTER + 1))
        nx = cx + int(rng.integers(-JITTER, JITTER + 1))
        if not (half_h <= ny < H - half_h): continue
        if not (half_w <= nx < W - half_w): continue
        patch = extract_patch_2d(img, ny, nx, patch_h, patch_w)
        if apply_clahe_flag:
            patch = apply_clahe(patch)
        patches.append(patch)
        centroids.append((stem, ny, nx))

    if len(patches) < n_target:
        print(f"[WARN] Raggiunte solo {len(patches)}/{n_target} patch dopo {max_attempts} tentativi")

    n_jitter = len(patches) - len(base_patches)
    print(f"{len(patches)} patch positive totali ({len(base_patches)} reali + {n_jitter} jitter-augmented)")
    return patches, [1.0] * len(patches)

def _too_close_2d(cy, cx, centers_2d, min_dist=15):
    return any(abs(cy - c_cy) + abs(cx - c_cx) < min_dist
               for _, c_cy, c_cx in centers_2d)


def extract_hard_negatives(image_files, annotations, positive_centroids, n_needed, patch_h, patch_w, apply_clahe_flag, stride=STRIDE):
    hard_negs = []
    pos_stems = set(stem for stem, _, _ in positive_centroids)

    for img_path, (label, row, col, half_h, half_w) in zip(image_files, annotations):
        if img_path.stem not in pos_stems or len(hard_negs) >= n_needed:
            break
        img  = load_image(img_path)
        H, W = img.shape

        for r in range(0, H - patch_h + 1, stride):
            for c in range(0, W - patch_w + 1, stride):
                cy_center = r + patch_h // 2
                cx_center = c + patch_w // 2
                local_centers = [(s, cy, cx) for s, cy, cx in positive_centroids if s == img_path.stem]
                if _too_close_2d(cy_center, cx_center, local_centers, min_dist=patch_h):
                    continue
                patch = img[r:r + patch_h, c:c + patch_w].astype(np.float64)
                if apply_clahe_flag:
                    patch = apply_clahe(patch)
                hard_negs.append(patch)
                if len(hard_negs) >= n_needed:
                    break
            if len(hard_negs) >= n_needed:
                break

    return hard_negs


def extract_random_negatives(image_files, annotations, positive_centroids, n_needed, patch_h, patch_w, apply_clahe_flag, rng):
    patches  = []
    half_h_p = patch_h // 2
    half_w_p = patch_w // 2

    # carica solo le immagini negative
    neg_imgs = []
    for img_path, (label, *_) in zip(image_files, annotations):
        if label == 0:
            neg_imgs.append(load_image(img_path))

    if not neg_imgs:
        return patches

    attempts, max_att = 0, n_needed * 30
    while len(patches) < n_needed and attempts < max_att:
        attempts += 1
        img  = neg_imgs[int(rng.integers(0, len(neg_imgs)))]
        H, W = img.shape
        if H < patch_h or W < patch_w:
            continue
        cy = int(rng.integers(half_h_p, H - half_h_p))
        cx = int(rng.integers(half_w_p, W - half_w_p))
        patch = extract_patch_2d(img, cy, cx, patch_h, patch_w)
        if apply_clahe_flag:
            patch = apply_clahe(patch)
        patches.append(patch)

    return patches


def extract_negative_patches(image_files, annotations, positive_centroids, n_total, patch_h, patch_w, apply_clahe_flag, rng, hard_neg_ratio=HARD_NEG_RATIO):
    n_hard   = int(n_total * hard_neg_ratio)
    n_random = n_total - n_hard

    hard_negs   = extract_hard_negatives(image_files, annotations, positive_centroids, n_hard, patch_h, patch_w, apply_clahe_flag)
    random_negs = extract_random_negatives(image_files, annotations, positive_centroids, n_random, patch_h, patch_w, apply_clahe_flag, rng)

    all_negs = hard_negs + random_negs
    rng.shuffle(all_negs)
    return all_negs

def render_gt_bboxes(image_files, annotations, output_dir, subject_id, patch_h=PATCH_H, patch_w=PATCH_W):
    saved = []
    for img_path, (label, row, col, half_h, half_w) in zip(image_files, annotations):
        if label != 1:
            continue

        img    = load_image(img_path)
        gt_box = build_gt_box(img.shape, row, col, half_h or DEFAULT_HALF_BOX_PX,  half_w or DEFAULT_HALF_BOX_PX)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.add_patch(mpatches.Rectangle((gt_box['c0'], gt_box['r0']), gt_box['c1'] - gt_box['c0'], gt_box['r1'] - gt_box['r0'], linewidth=2, edgecolor='lime', facecolor='none', label='GT'))
        ax.set_title(f"{subject_id}  —  {img_path.name}\n GT: ±{gt_box['half_h']}r ±{gt_box['half_w']}c px", fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('off')
        slice_path = output_dir / f"gt_bbox_{subject_id}_{img_path.stem}_full.png"
        fig.savefig(str(slice_path), dpi=120, bbox_inches='tight')
        plt.close(fig)
        saved.append(slice_path)
        print(f"Immagine salvata: {slice_path.name}")

        cy, cx   = gt_box['cy'], gt_box['cx']
        r_start  = cy - patch_h // 2
        c_start  = cx - patch_w // 2
        patch    = extract_patch_2d(img, cy, cx, patch_h, patch_w)

        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.imshow(patch, cmap='gray', vmin=0, vmax=1, extent=[0, patch_w, patch_h, 0])
        ax2.add_patch(mpatches.Rectangle((gt_box['c0'] - c_start, gt_box['r0'] - r_start), gt_box['c1'] - gt_box['c0'], gt_box['r1'] - gt_box['r0'],linewidth=2, edgecolor='lime', facecolor='none', label='GT'))
        ax2.set_title(f"{subject_id}  {img_path.stem}  patch {patch_h}×{patch_w}px", fontsize=9)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.axis('off')
        patch_path = output_dir / f"gt_bbox_{subject_id}_{img_path.stem}_patch.png"
        fig2.savefig(str(patch_path), dpi=120, bbox_inches='tight')
        plt.close(fig2)
        saved.append(patch_path)
        print(f"Patch salvata: {patch_path.name}")

    return saved

def train_test_split_subjects(subjects: list, train_ratio: float = TRAIN_RATIO, seed: int = SEED):
    rng = np.random.default_rng(seed)
    subjects = list(subjects)
    rng.shuffle(subjects)
    n_train = max(1, int(len(subjects) * train_ratio))
    return subjects[:n_train], subjects[n_train:]

def process_subject_sample(subject_id, image_root, anno_root, patch_h, patch_w, neg_ratio, apply_clahe_flag, rng):

    subject_img_dir = image_root / subject_id
    if not subject_img_dir.exists():
        print(f"[SKIP] {subject_id}: cartella immagini non trovata")
        return [], []

    image_files = get_subject_images(subject_img_dir)
    if not image_files:
        print(f"[SKIP] {subject_id}: nessuna immagine .png trovata")
        return [], []

    annotations = load_subject_annotations(anno_root, subject_id, image_files)
    if annotations is None:
        return [], []

    pos_patches, positive_centroids = extract_positive_patches(image_files, annotations, patch_h, patch_w, apply_clahe_flag)
    if not pos_patches:
        print(f"[WARN] {subject_id}: nessuna immagine positiva -> skip")
        return [], []

    n_neg = int(len(pos_patches) * neg_ratio)
    neg_patches = extract_negative_patches(image_files, annotations, positive_centroids, n_neg, patch_h, patch_w, apply_clahe_flag, rng)

    n_hard = int(n_neg * HARD_NEG_RATIO)
    n_rand = len(neg_patches) - n_hard
    print(f"{subject_id}: {len(pos_patches)} pos | {n_hard} hard neg | {n_rand} random neg")

    patches = pos_patches + neg_patches
    labels  = [1.0] * len(pos_patches) + [0.0] * len(neg_patches)
    return patches, labels

def process_subject_grid(subject_id, image_root, anno_root, tmp_dir: Path):
    subject_img_dir = image_root / subject_id
    if not subject_img_dir.exists():
        print(f"[SKIP] {subject_id}: cartella immagini non trovata")
        return None, None

    image_files = get_subject_images(subject_img_dir)
    if not image_files:
        print(f"[SKIP] {subject_id}: nessuna immagine .png trovata")
        return None, None

    annotations = load_subject_annotations(anno_root, subject_id, image_files)
    if annotations is None:
        return None, None

    patches_all, labels_all = [], []

    for img_path, (label, row, col, half_h, half_w) in tqdm(zip(image_files, annotations), desc=f"{subject_id}", total=len(image_files), leave=False):

        img = load_image(img_path)
        gt_box = None
        if label == 1:
            gt_box = build_gt_box(img.shape, row, col, half_h or DEFAULT_HALF_BOX_PX, half_w or DEFAULT_HALF_BOX_PX)

        if FIXED_N_FLAG:
            patches, coords = extract_fixed_number_grid(img, PATCH_H, PATCH_W, STRIDE, N_PATCH_FIXED)
        else:
            patches, coords = extract_grid_patches(img, PATCH_H, PATCH_W, STRIDE)

        for patch, (r, c) in zip(patches, coords):
            lbl = (1.0 if gt_box is not None and patch_overlaps_box(r, c, PATCH_H, PATCH_W, gt_box) else 0.0)
            patches_all.append(patch)
            labels_all.append(lbl)

    if not patches_all:
        print(f"[WARN] {subject_id}: nessuna patch estratta -> skip")
        return None, None

    data_arr   = np.array(patches_all, dtype=np.float32)
    labels_arr = np.array(labels_all,  dtype=np.float32)
    n_pos = int(np.sum(labels_arr == 1))
    n_neg = int(np.sum(labels_arr == 0))
    print(f"{subject_id}: {len(labels_arr)} patch (pos={n_pos}, neg={n_neg})")

    tmp_data   = tmp_dir / f"{subject_id}_data.npy"
    tmp_labels = tmp_dir / f"{subject_id}_labels.npy"
    np.save(str(tmp_data),   data_arr)
    np.save(str(tmp_labels), labels_arr)
    del patches_all, labels_all, data_arr, labels_arr
    return tmp_data, tmp_labels

def process_subject(subject_id, image_root, anno_root,patch_h=PATCH_H, patch_w=PATCH_W,neg_ratio=NEG_RATIO, apply_clahe_flag=CLAHE_FLAG,rng=None, tmp_dir=None):
    if SAMPLE_FLAG:
        return process_subject_sample(subject_id, image_root, anno_root,patch_h, patch_w, neg_ratio, apply_clahe_flag, rng)
    else:
        return process_subject_grid(subject_id, image_root, anno_root, tmp_dir)

def save_split_sample(patches, labels, output_dir, split_name):
    if not patches:
        print(f"  [WARN] split '{split_name}' vuoto.")
        return None, None
    data_arr   = np.array(patches, dtype=np.float32)
    labels_arr = np.array(labels,  dtype=np.float32)
    data_path   = output_dir / f"sampled_custom_{split_name}_data.npy"
    labels_path = output_dir / f"sampled_custom_{split_name}_label.npy"
    np.save(str(data_path),   data_arr)
    np.save(str(labels_path), labels_arr)
    n_pos = int(np.sum(labels_arr == 1))
    n_neg = int(np.sum(labels_arr == 0))
    print(f"{split_name}: {len(labels_arr)} patch (pos={n_pos}, neg={n_neg})")
    return data_path, labels_path


def save_split_grid(tmp_file_pairs, output_dir, split_name, rng):
    valid_pairs = [(d, l) for d, l in tmp_file_pairs if d is not None]
    if not valid_pairs:
        print(f"[WARN] split '{split_name}' vuoto.")
        return None, None
    total = sum(np.load(str(d), mmap_mode='r').shape[0] for d, _ in valid_pairs)
    all_data   = np.empty((total, PATCH_H, PATCH_W), dtype=np.float32)
    all_labels = np.empty(total, dtype=np.float32)
    cursor = 0
    for d_path, l_path in valid_pairs:
        d = np.load(str(d_path))
        l = np.load(str(l_path))
        n = len(l)
        all_data[cursor:cursor + n]   = d
        all_labels[cursor:cursor + n] = l
        cursor += n
        del d, l
    idx        = rng.permutation(total)
    all_data   = all_data[idx]
    all_labels = all_labels[idx]
    suffix      = "fixed" if FIXED_N_FLAG else "classic"
    data_path   = output_dir / f"custom_{suffix}_{split_name}_data.npy"
    labels_path = output_dir / f"custom_{suffix}_{split_name}_label.npy"
    np.save(str(data_path),   all_data)
    np.save(str(labels_path), all_labels)
    n_pos = int(np.sum(all_labels == 1))
    n_neg = int(np.sum(all_labels == 0))
    print(f"  → {split_name}: {total} patch (pos={n_pos}, neg={n_neg})")
    del all_data, all_labels
    return data_path, labels_path

def write_report(output_dir, train_subjects, test_subjects, patches_per_split):
    report_path = output_dir / "split_report.txt"
    mode_str    = "SAMPLE" if SAMPLE_FLAG else ("GRID_FIXED" if FIXED_N_FLAG else "GRID_CLASSIC")
    with open(report_path, 'w') as f:
        f.write("Custom Dataset — Split Report\n")
        f.write(f"Modalità: {mode_str}\n")
        f.write(f"Hard neg ratio: {HARD_NEG_RATIO:.0%} dei negativi da sliding window\n")
        f.write(f"Neg/Pos ratio: {NEG_RATIO}:1\n")
        f.write(f"Patch size: {PATCH_H}×{PATCH_W} px\n")
        f.write(f"CLAHE: {'TRUE' if CLAHE_FLAG else 'FALSE'}\n\n")
        for split_name, subjects in [("TRAIN", train_subjects), ("TEST", test_subjects)]:
            info = patches_per_split.get(split_name, {})
            f.write(f"[{split_name}]\n"
                    f"Soggetti: {len(subjects)}\n"
                    f"Patch totali: {info.get('total', 0)}\n"
                    f"Positivi: {info.get('pos', 0)}\n"
                    f"Negativi: {info.get('neg', 0)}\n\n")
    print(f"Report salvato in: {report_path}")

def run_demo(image_root, anno_root, output_dir, rng):
    all_subjects = sorted([d.name for d in image_root.iterdir() if d.is_dir()])
    if not all_subjects:
        raise FileNotFoundError(f"Nessun soggetto trovato in {image_root}")

    subject_id = DEMO_SUBJECT_ID if DEMO_SUBJECT_ID else all_subjects[0]
    if subject_id not in all_subjects:
        print(f"  [WARN] DEMO_SUBJECT_ID '{subject_id}' non trovato, "
              f"uso '{all_subjects[0]}'")
        subject_id = all_subjects[0]

    mode_str = "SAMPLE" if SAMPLE_FLAG else ("GRID_FIXED" if FIXED_N_FLAG else "GRID_CLASSIC")
    print(f"[DEMO] Soggetto: {subject_id}")
    print(f"[DEMO] Modalità: {mode_str}")
    print(f"[DEMO] Output: {output_dir}")
    print(f"[DEMO] Patch size: {PATCH_H}×{PATCH_W} px")
    print(f"[DEMO] N patches: {DEMO_N_PATCHES}")
    print(f"[DEMO] CLAHE: {'TRUE' if CLAHE_FLAG else 'FALSE'}\n")

    subject_img_dir = image_root / subject_id
    image_files     = get_subject_images(subject_img_dir)
    annotations     = load_subject_annotations(anno_root, subject_id, image_files)

    if annotations is None:
        raise RuntimeError(f"Annotazioni non trovate per {subject_id}")

    print("Estrazione patch positive")
    if SAMPLE_FLAG:
        patches, labels = extract_demo_positive_patches(
            image_files, annotations, PATCH_H, PATCH_W, CLAHE_FLAG,
            n_target=DEMO_N_PATCHES, rng=rng
        )
        data_path   = output_dir / "sampled_demo_positive_patches.npy"
        labels_path = output_dir / "sampled_demo_positive_labels.npy"
    else:
        patches, labels = [], []
        for img_path, (label, row, col, half_h, half_w) in tqdm(
                zip(image_files, annotations), total=len(image_files), desc="slices"):
            img    = load_image(img_path)
            gt_box = None
            if label == 1:
                gt_box = build_gt_box(img.shape, row, col,
                                       half_h or DEFAULT_HALF_BOX_PX,
                                       half_w or DEFAULT_HALF_BOX_PX)
            if FIXED_N_FLAG:
                ps, coords = extract_fixed_number_grid(
                    img, PATCH_H, PATCH_W, STRIDE, N_PATCH_FIXED)
            else:
                ps, coords = extract_grid_patches(img, PATCH_H, PATCH_W, STRIDE)
            for p, (r, c) in zip(ps, coords):
                lbl = (1.0 if gt_box is not None and
                       patch_overlaps_box(r, c, PATCH_H, PATCH_W, gt_box) else 0.0)
                patches.append(p)
                labels.append(lbl)
        data_path   = output_dir / "demo_grid_patches.npy"
        labels_path = output_dir / "demo_grid_labels.npy"

    data_arr   = np.array(patches, dtype=np.float32)
    labels_arr = np.array(labels,  dtype=np.float32)
    np.save(str(data_path),   data_arr)
    np.save(str(labels_path), labels_arr)
    print(f"Patch salvate : {data_path} shape={data_arr.shape}")
    print(f"Label salvate : {labels_path} shape={labels_arr.shape}")

    print("Generazione immagini con bounding box GT")
    saved_imgs = render_gt_bboxes(image_files, annotations, output_dir, subject_id,patch_h=PATCH_H, patch_w=PATCH_W)
    print(f"{len(saved_imgs)} immagini GT salvate.")
    for p in saved_imgs:
        print(f"   {p.name}")

    if SHOW_FLAG:
        debug_show_patches(patches, labels, n=10)

    return data_path, labels_path


def main_preprocessing(image_root, anno_root, output_dir, rng):
    mode_str = "SAMPLE" if SAMPLE_FLAG else ("GRID_FIXED" if FIXED_N_FLAG else "GRID_CLASSIC")
    print(f"Modalità: {mode_str}")
    print(f"Hard neg ratio: {HARD_NEG_RATIO:.0%} dei negativi da sliding window")
    print(f"Neg/Pos ratio: {NEG_RATIO}:1")
    print(f"CLAHE: {'TRUE' if CLAHE_FLAG else 'FALSE'}\n")

    all_subjects = sorted([d.name for d in image_root.iterdir() if d.is_dir()])
    print(f"Soggetti trovati: {len(all_subjects)}\n")

    train_subjects, test_subjects = train_test_split_subjects(all_subjects)
    all_splits    = {"train": train_subjects, "test": test_subjects}
    patches_per_split = {}

    tmp_dir = output_dir / "_tmp_subjects"
    if not SAMPLE_FLAG:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    if SAMPLE_FLAG:
        for split_name, subject_list in all_splits.items():
            print(f"══ {split_name.upper()} ({len(subject_list)} soggetti) ══")
            split_patches, split_labels = [], []

            for subject_id in subject_list:
                patches, labels = process_subject(subject_id, image_root, anno_root,PATCH_H, PATCH_W, NEG_RATIO, CLAHE_FLAG, rng)
                split_patches.extend(patches)
                split_labels.extend(labels)

            combined = list(zip(split_patches, split_labels))
            rng.shuffle(combined)
            if combined:
                split_patches, split_labels = zip(*combined)
                split_patches = list(split_patches)
                split_labels  = list(split_labels)

            n_pos = sum(1 for l in split_labels if l == 1.0)
            n_neg = sum(1 for l in split_labels if l == 0.0)
            patches_per_split[split_name.upper()] = {
                "total": len(split_labels), "pos": n_pos, "neg": n_neg}
            save_split_sample(split_patches, split_labels, output_dir, split_name)

        if SHOW_FLAG:
            debug_show_patches(split_patches, split_labels, n=10)

    else:
        for split_name, subject_list in all_splits.items():
            print(f"{split_name.upper()} ({len(subject_list)} soggetti)")
            tmp_pairs = []
            for subject_id in subject_list:
                d_path, l_path = process_subject(
                    subject_id, image_root, anno_root, tmp_dir=tmp_dir)
                tmp_pairs.append((d_path, l_path))
            save_split_grid(tmp_pairs, output_dir, split_name, rng)

            suffix   = "fixed" if FIXED_N_FLAG else "classic"
            lbl_path = output_dir / f"custom_{suffix}_{split_name}_label.npy"
            if lbl_path.exists():
                lbl_arr = np.load(str(lbl_path))
                patches_per_split[split_name.upper()] = {
                    "total": len(lbl_arr),
                    "pos":   int(np.sum(lbl_arr == 1)),
                    "neg":   int(np.sum(lbl_arr == 0))}
                del lbl_arr
            print()

        print("Pulizia file temporanei …")
        for f in tmp_dir.glob("*.npy"):
            f.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
        print("✓ Fatto.\n")

    write_report(output_dir, train_subjects, test_subjects, patches_per_split)

    print("\n" + "=" * 55)
    print("RIEPILOGO FINALE")
    print("=" * 55)
    for split_name, info in patches_per_split.items():
        bal = info['pos'] / info['total'] * 100 if info['total'] > 0 else 0
        print(f"  {split_name:6s}: {info['total']:5d} patch "
              f"| pos={info['pos']:4d} ({bal:.0f}%) "
              f"| neg={info['neg']:4d}")


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    root       = Path(DATASET_ROOT_DIR)
    image_root = root / "images"
    anno_root  = root / "annotations"
    output_dir = Path(DATASET_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_root.exists():
        raise FileNotFoundError(f"Cartella immagini non trovata: {image_root}")
    if not anno_root.exists():
        raise FileNotFoundError(f"Cartella annotazioni non trovata: {anno_root}")

    print(f"Dataset root  : {root}")
    print(f"Immagini      : {image_root}")
    print(f"Annotazioni   : {anno_root}")
    print(f"Output        : {output_dir}")
    print()

    if DEMO_FLAG:
        run_demo(image_root, anno_root, output_dir, rng)
    else:
        main_preprocessing(image_root, anno_root, output_dir, rng)