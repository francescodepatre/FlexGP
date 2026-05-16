#custom_LYMPH_preprocessing.py
import re
import warnings
import random
from pathlib import Path
from collections import defaultdict
from skimage.transform import resize

import pydicom
from skimage.exposure import equalize_adapthist
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as mpatches
import tomli

warnings.filterwarnings("ignore")

with open("params.toml", "rb") as f:
    cfg = tomli.load(f)

PATCH_H = cfg['preprocessing']['PATCH_H']
PATCH_W = cfg['preprocessing']['PATCH_W']
STRIDE = cfg['preprocessing']['STRIDE']
HU_MIN = cfg['preprocessing']['HU_MIN']
HU_MAX = cfg['preprocessing']['HU_MAX']
SEED = cfg['preprocessing']['SEED']
DATASET_ROOT_DIR = cfg['preprocessing']['DATASET_ROOT_DIR']
DATASET_OUTPUT_DIR = cfg['preprocessing']['DATASET_OUTPUT_DIR']
NEG_RATIO = cfg['preprocessing']['NEG_RATIO']
HARD_NEG_RATIO = cfg['preprocessing']['HARD_NEG_RATIO']
TRAIN_RATIO = cfg['preprocessing']['TRAIN_RATIO']
CLAHE_FLAG = cfg['preprocessing']['CLAHE_FLAG']
SHOW_FLAG = cfg['preprocessing']['SHOW_FLAG']
FIXED_N_FLAG = cfg['preprocessing']['FIXED_N_FLAG']
N_PATCH_FIXED = cfg['preprocessing']['N_PATCH_FIXED']
SAMPLE_FLAG = cfg['preprocessing']['SAMPLE_FLAG']
DEMO_FLAG = cfg['preprocessing']['DEMO_FLAG']
DEMO_PATIENT_ID = cfg['preprocessing']['DEMO_PATIENT_ID']
DEMO_N_PATCHES = cfg['preprocessing']['DEMO_N_PATCHES']

def debug_show_patches(patches, labels, n=10):
    for i in range(min(n, len(patches))):
        plt.imshow(patches[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
        plt.show()

def get_ct_series(patient_dir):
    series_map = defaultdict(list)
    for f in patient_dir.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            if hasattr(ds, "Modality") and ds.Modality != "CT":
                continue
            series_map[ds.SeriesInstanceUID].append(f)
        except Exception:
            continue
    if not series_map:
        raise ValueError("Nessuna serie CT trovata")
    return sorted(max(series_map.values(), key=len))


def load_dicom_volume(patient_dir: Path):
    dcm_files = get_ct_series(patient_dir)
    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f))
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)
        except Exception:
            continue
    if not slices:
        raise ValueError(f"Nessuno slice valido in {patient_dir}")

    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except Exception:
        try:
            slices.sort(key=lambda s: int(s.InstanceNumber))
        except Exception:
            pass

    arrays = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, 'RescaleSlope',     1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        arrays.append(arr * slope + intercept)

    #Formato: Z, Y, X
    volume = np.stack(arrays, axis=0)   

    try:
        dz = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
    except Exception:
        dz = float(getattr(slices[0], 'SliceThickness', 1.0))

    try:
        dy, dx = [float(x) for x in slices[0].PixelSpacing]
    except Exception:
        dy, dx = 1.0, 1.0

    spacing = np.array([dz, dy, dx])

    try:
        origin = np.array([float(x) for x in slices[0].ImagePositionPatient])
        origin = origin[[2, 1, 0]]
    except Exception:
        origin = np.zeros(3)

    return volume, spacing, origin

def parse_voxel_annotation_file(filepath: Path):
    centroids = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = re.split(r'[\s,\t]+', line)
                if len(parts) >= 3:
                    try:
                        x_col = int(parts[0])
                        y_row = int(parts[1])
                        z_sli = int(parts[2])
                        centroids.append((z_sli, y_row, x_col))
                    except ValueError:
                        continue
    except Exception:
        return []
    return centroids


def find_annotation_files(anno_dir: Path, patient_id: str):
    if not anno_dir or not anno_dir.exists():
        return []
    patient_anno = anno_dir / patient_id
    if not patient_anno.exists():
        for d in anno_dir.iterdir():
            if d.is_dir() and d.name.lower() == patient_id.lower():
                patient_anno = d
                break
        else:
            return []
    files = sorted(patient_anno.glob("*indices*.txt"))
    if not files:
        files = sorted(patient_anno.glob("*.txt"))
    return files


def parse_candidate_file(filepath: Path):
    candidates = []
    if not filepath or not filepath.exists():
        return candidates
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = re.split(r'[\s,\t]+', line.strip())
                if len(parts) >= 3:
                    try:
                        i, j, k = int(parts[0]), int(parts[1]), int(parts[2])
                        r = float(parts[3]) if len(parts) >= 4 else 5.0
                        candidates.append((i, j, k, r))
                    except ValueError:
                        continue
    except Exception:
        pass
    return candidates

def window_and_normalize(volume, hu_min=HU_MIN, hu_max=HU_MAX):
    vol = np.clip(volume, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min)
    return vol.astype(np.float32)


def apply_clahe(patch):
    return equalize_adapthist(patch.astype(np.float64), clip_limit=0.03)


def extract_patch_2d(slice_2d, center_r, center_c, patch_h=PATCH_H, patch_w=PATCH_W):
    H, W = slice_2d.shape
    half_h, half_w = patch_h // 2, patch_w // 2

    pad_top    = max(0, half_h - center_r)
    pad_bottom = max(0, center_r + half_h - H + 1)
    pad_left   = max(0, half_w - center_c)
    pad_right  = max(0, center_c + half_w - W + 1)

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        padded = np.pad(slice_2d,((pad_top, pad_bottom), (pad_left, pad_right)),mode='constant', constant_values=0.0)
        cr = center_r + pad_top
        cc = center_c + pad_left
    else:
        padded, cr, cc = slice_2d, center_r, center_c

    patch = padded[cr - half_h: cr - half_h + patch_h, cc - half_w: cc - half_w + patch_w]

    if patch.shape != (patch_h, patch_w):
        patch = resize(patch, (patch_h, patch_w), anti_aliasing=True, mode='reflect')

    return patch.astype(np.float64)

def extract_positive_patches(volume_norm, anno_files, patch_h, patch_w, apply_clahe_flag):
    patches, centroids = [], []
    for anno_file in anno_files:
        coords = parse_voxel_annotation_file(anno_file)
        for (cz, cy, cx) in coords:
            if not (0 <= cz < volume_norm.shape[0]):
                continue
            patch = extract_patch_2d(volume_norm[cz], cy, cx, patch_h, patch_w)
            if apply_clahe_flag:
                patch = apply_clahe(patch)
            patches.append(patch)
            centroids.append((cz, cy, cx))
    return patches, centroids

def extract_demo_positive_patches(volume_norm, anno_files, patch_h, patch_w, apply_clahe_flag, n_target=10, rng=None):

    if rng is None:
        rng = np.random.default_rng(SEED)

    base_patches, base_centroids = extract_positive_patches(
        volume_norm, anno_files, patch_h, patch_w, apply_clahe_flag
    )
    if not base_patches:
        raise RuntimeError("Nessuna patch positiva estratta.")

    print(f"    Patch base estratte: {len(base_patches)} (target={n_target})")

    patches   = list(base_patches)
    centroids = list(base_centroids)

    if len(patches) >= n_target:
        idx = rng.choice(len(patches), size=n_target, replace=False)
        patches = [patches[i] for i in idx]
        print(f"    → campionate {n_target} patch dalle {len(base_patches)} disponibili")
        return patches, [1.0] * n_target

    JITTER = 3
    Z, Y, X = volume_norm.shape
    half_h, half_w = patch_h // 2, patch_w // 2
    attempts, max_attempts = 0, (n_target - len(patches)) * 50

    while len(patches) < n_target and attempts < max_attempts:
        attempts += 1
        base_idx = int(rng.integers(0, len(base_centroids)))
        cz, cy, cx = base_centroids[base_idx]
        nz = cz + int(rng.integers(-JITTER, JITTER + 1))
        ny = cy + int(rng.integers(-JITTER, JITTER + 1))
        nx = cx + int(rng.integers(-JITTER, JITTER + 1))
        if not (0 <= nz < Z): continue
        if not (half_h <= ny < Y - half_h): continue
        if not (half_w <= nx < X - half_w): continue
        patch = extract_patch_2d(volume_norm[nz], ny, nx, patch_h, patch_w)
        if apply_clahe_flag:
            patch = apply_clahe(patch)
        patches.append(patch)
        centroids.append((nz, ny, nx))

    if len(patches) < n_target:
        print(f"[WARN] Raggiunte solo {len(patches)}/{n_target} patch dopo {max_attempts} tentativi")

    n_jitter = len(patches) - len(base_patches)
    print(f"{len(patches)} patch positive totali ({len(base_patches)} reali + {n_jitter} jitter-augmented)")
    return patches, [1.0] * len(patches)

def _too_close(z, y, x, centers, min_dist=15):
    return any(abs(z - cz) + abs(y - cy) + abs(x - cx) < min_dist
               for cz, cy, cx in centers)


def extract_hard_negatives(volume_norm, positive_centroids, n_needed, patch_h, patch_w, apply_clahe_flag, stride=STRIDE):
    hard_negs = []
    Z, H, W = volume_norm.shape
    ln_slices = sorted(set(cz for cz, _, _ in positive_centroids))

    for cz in ln_slices:
        if len(hard_negs) >= n_needed:
            break
        slice_2d = volume_norm[cz]
        for r in range(0, H - patch_h + 1, stride):
            for c in range(0, W - patch_w + 1, stride):
                cy_center = r + patch_h // 2
                cx_center = c + patch_w // 2
                if _too_close(cz, cy_center, cx_center, positive_centroids,
                              min_dist=patch_h):
                    continue
                patch = slice_2d[r:r + patch_h, c:c + patch_w].astype(np.float64)
                if apply_clahe_flag:
                    patch = apply_clahe(patch)
                hard_negs.append(patch)
                if len(hard_negs) >= n_needed:
                    break
    return hard_negs

def extract_random_negatives(volume_norm, positive_centroids, n_needed, patch_h, patch_w, apply_clahe_flag, candidate_file, rng):
    patches = []
    Z, Y, X = volume_norm.shape
    half_h, half_w = patch_h // 2, patch_w // 2
    forbidden = list(positive_centroids)

    if candidate_file is not None:
        candidates = parse_candidate_file(candidate_file)
        for idx in rng.permutation(len(candidates)):
            if len(patches) >= n_needed:
                break
            ci, cj, ck, _ = candidates[idx]
            cz, cy, cx = ck, ci, cj
            if cz >= Z or _too_close(cz, cy, cx, forbidden, 10):
                continue
            patch = extract_patch_2d(volume_norm[cz], cy, cx, patch_h, patch_w)
            if apply_clahe_flag:
                patch = apply_clahe(patch)
            patches.append(patch)
            forbidden.append((cz, cy, cx))

    attempts, max_attempts = 0, n_needed * 30
    while len(patches) < n_needed and attempts < max_attempts:
        attempts += 1
        cz = int(rng.integers(Z // 4, 3 * Z // 4))
        cy = int(rng.integers(half_h, Y - half_h))
        cx = int(rng.integers(half_w, X - half_w))
        if _too_close(cz, cy, cx, forbidden, 15):
            continue
        patch = extract_patch_2d(volume_norm[cz], cy, cx, patch_h, patch_w)
        if apply_clahe_flag:
            patch = apply_clahe(patch)
        patches.append(patch)
        forbidden.append((cz, cy, cx))

    return patches


def extract_negative_patches(volume_norm, positive_centroids, n_total, patch_h, patch_w, apply_clahe_flag, candidate_file, rng, hard_neg_ratio=HARD_NEG_RATIO):
    
    n_hard   = int(n_total * hard_neg_ratio)
    n_random = n_total - n_hard

    hard_negs   = extract_hard_negatives(volume_norm, positive_centroids, n_hard, patch_h, patch_w, apply_clahe_flag)
    random_negs = extract_random_negatives(volume_norm, positive_centroids, n_random, patch_h, patch_w, apply_clahe_flag, candidate_file, rng)

    all_negs = hard_negs + random_negs
    rng.shuffle(all_negs)
    return all_negs

def parse_sizes_file(filepath: Path):
    sizes = []
    if not filepath or not filepath.exists():
        return sizes
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = re.split(r'[\s,\t]+', line)
                if len(parts) >= 2:
                    try:
                        sizes.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
    except Exception:
        pass
    return sizes


def find_sizes_file(anno_dir: Path, patient_id: str):
    patient_anno = anno_dir / patient_id
    if not patient_anno.exists():
        return None
    matches = sorted(patient_anno.glob("*sizes*.txt"))
    return matches[0] if matches else None


def compute_gt_boxes_pixel(centroids_zyx, sizes_mm, spacing_yx):
    dy, dx = spacing_yx
    boxes = []
    for (cz, cy, cx), (short_mm, long_mm) in zip(centroids_zyx, sizes_mm):
        half_r = max(1, round(long_mm / 2.0 / dy))
        half_c = max(1, round(long_mm / 2.0 / dx))
        boxes.append(dict(
            cz=cz, cy=cy, cx=cx,
            r0=cy - half_r, c0=cx - half_c,
            r1=cy + half_r, c1=cx + half_c,
            short_mm=short_mm, long_mm=long_mm,
            half_r=half_r, half_c=half_c
        ))
    return boxes


def load_gt_boxes(anno_files, spacing):
    all_centroids = []
    for f in anno_files:
        all_centroids.extend(parse_voxel_annotation_file(f))
    if not all_centroids:
        return []
    patient_dir = anno_files[0].parent
    sizes_files = list(patient_dir.glob("*sizes*.txt"))
    sizes_mm = parse_sizes_file(sizes_files[0]) if sizes_files else [(15.0, 15.0)] * len(all_centroids)
    n = min(len(all_centroids), len(sizes_mm))
    return compute_gt_boxes_pixel(all_centroids[:n], sizes_mm[:n], spacing_yx=(spacing[1], spacing[2]))


def render_gt_bboxes(volume_norm, gt_boxes, output_dir, patient_id, patch_h=PATCH_H, patch_w=PATCH_W, pred_boxes_per_ln=None):
    saved = []
    for idx, box in enumerate(gt_boxes):
        cz = box['cz']
        if not (0 <= cz < volume_norm.shape[0]):
            print(f"  [WARN] slice {cz} fuori range → skip bbox {idx}")
            continue

        slice_full = volume_norm[cz]

        # Slice completa
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(slice_full, cmap='gray', vmin=0, vmax=1)
        ax.add_patch(mpatches.Rectangle((box['c0'], box['r0']), box['c1'] - box['c0'], box['r1'] - box['r0'], linewidth=2, edgecolor='lime', facecolor='none', label='GT'))
        if pred_boxes_per_ln is not None and idx < len(pred_boxes_per_ln):
            for pidx, pb in enumerate(pred_boxes_per_ln[idx]):
                ax.add_patch(mpatches.Rectangle((pb['c0'], pb['r0']),pb['c1'] - pb['c0'], pb['r1'] - pb['r0'],linewidth=2, edgecolor='red', facecolor='none',label='Pred' if pidx == 0 else '_nolegend_'))
        ax.set_title(f"{patient_id}  LN#{idx+1}  slice={cz}\n GT: {box['short_mm']:.1f}×{box['long_mm']:.1f} mm (±{box['half_r']}r ±{box['half_c']}c px)", fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('off')
        slice_path = output_dir / f"gt_bbox_{patient_id}_ln{idx+1:02d}_slice.png"
        fig.savefig(str(slice_path), dpi=120, bbox_inches='tight')
        plt.close(fig)
        saved.append(slice_path)
        print(f"  → Slice salvata : {slice_path.name}")

        # Patch ritagliata
        half_h, half_w = patch_h // 2, patch_w // 2
        cy, cx = box['cy'], box['cx']
        r_start, c_start = cy - half_h, cx - half_w
        patch = extract_patch_2d(slice_full, cy, cx, patch_h, patch_w)

        fig2, ax2 = plt.subplots(figsize=(5, 2))
        ax2.imshow(patch, cmap='gray', vmin=0, vmax=1,extent=[0, patch_w, patch_h, 0])
        ax2.add_patch(mpatches.Rectangle((box['c0'] - c_start, box['r0'] - r_start),box['c1'] - box['c0'], box['r1'] - box['r0'],linewidth=2, edgecolor='lime', facecolor='none', label='GT'))
        if pred_boxes_per_ln is not None and idx < len(pred_boxes_per_ln):
            for pidx, pb in enumerate(pred_boxes_per_ln[idx]):
                ax2.add_patch(mpatches.Rectangle((pb['c0'] - c_start, pb['r0'] - r_start),pb['c1'] - pb['c0'], pb['r1'] - pb['r0'],linewidth=2, edgecolor='red', facecolor='none',label='Pred' if pidx == 0 else '_nolegend_'))
        ax2.set_title(f"{patient_id}  LN#{idx+1}  patch {patch_h}×{patch_w}px", fontsize=9)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.axis('off')
        patch_path = output_dir / f"gt_bbox_{patient_id}_ln{idx+1:02d}_patch.png"
        fig2.savefig(str(patch_path), dpi=120, bbox_inches='tight')
        plt.close(fig2)
        saved.append(patch_path)
        print(f"Patch salvata : {patch_path.name}")

    return saved

def check_if_positive(z, r, c, gt_boxes):
    for box in gt_boxes:
        if (box['cz'] == z and box['r0'] <= r < box['r1'] and box['c0'] <= c < box['c1']):
            return 1.0
    return 0.0

def extract_grid_patches(slice_2d, patch_h, patch_w, stride):
    H, W = slice_2d.shape
    patches, coords = [], []
    for r in range(0, H - patch_h + 1, stride):
        for c in range(0, W - patch_w + 1, stride):
            patches.append(slice_2d[r:r + patch_h, c:c + patch_w])
            coords.append((r, c))
    return patches, coords


def extract_fixed_number_grid(slice_2d, patch_h, patch_w, stride, max_patches):
    patches, coords = extract_grid_patches(slice_2d, patch_h, patch_w, stride)
    if len(patches) > max_patches:
        idx = np.random.choice(len(patches), max_patches, replace=False)
        patches = [patches[i] for i in idx]
        coords  = [coords[i]  for i in idx]
    return patches, coords

def stratified_train_test_split(patients, train_ratio=TRAIN_RATIO, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for p in patients:
        if "MED" in p:   
            groups["MED"].append(p)
        elif "ABD" in p: 
            groups["ABD"].append(p)
        else:            
            groups["OTHER"].append(p)

    train, test = [], []

    for g in groups.values():
        rng.shuffle(g)
        n_train = int(len(g) * train_ratio)
        train.extend(g[:n_train])
        test.extend(g[n_train:])
    return train, test

def process_patient_sample(patient_id, dicom_root, anno_root, cand_root,patch_h, patch_w, neg_ratio, apply_clahe_flag, rng):
    patient_dicom = dicom_root / patient_id
    if not patient_dicom.exists():
        matches = list(dicom_root.glob(f"*{patient_id}*"))
        if not matches:
            print(f"[SKIP] {patient_id}: cartella DICOM non trovata")
            return [], []
        patient_dicom = matches[0]

    try:
        volume_raw, spacing, _ = load_dicom_volume(patient_dicom)
    except Exception as e:
        print(f"[SKIP] {patient_id}: errore lettura DICOM → {e}")
        return [], []

    volume_norm = window_and_normalize(volume_raw)

    anno_files = find_annotation_files(anno_root, patient_id)
    if not anno_files:
        for d in anno_root.iterdir():
            if patient_id.replace("_", "").lower() in d.name.replace("_", "").lower():
                anno_files = find_annotation_files(anno_root, d.name)
                break

    pos_patches, positive_centroids = extract_positive_patches( volume_norm, anno_files, patch_h, patch_w, apply_clahe_flag)
    if not pos_patches:
        print(f"[WARN] {patient_id}: nessuna annotazione trovata -> skip")
        return [], []

    cand_file = None
    if cand_root is not None:
        for pattern in [f"{patient_id}_candidates.txt", f"{patient_id}.txt", f"{patient_id}_cands.txt"]:
            candidate = cand_root / pattern
            if candidate.exists():
                cand_file = candidate
                break

    n_neg = int(len(pos_patches) * neg_ratio)
    neg_patches = extract_negative_patches(volume_norm, positive_centroids, n_neg, patch_h, patch_w, apply_clahe_flag, cand_file, rng)

    n_hard = int(n_neg * HARD_NEG_RATIO)
    n_rand = len(neg_patches) - n_hard
    print(f"  {patient_id}: {len(pos_patches)} pos | {n_hard} hard neg | {n_rand} random neg (vol={volume_norm.shape})")

    patches = pos_patches + neg_patches
    labels  = [1.0] * len(pos_patches) + [0.0] * len(neg_patches)
    return patches, labels

def process_patient_grid(patient_id, dicom_root, anno_root, tmp_dir: Path):
    patient_dicom = dicom_root / patient_id
    if not patient_dicom.exists():
        matches = list(dicom_root.glob(f"*{patient_id}*"))
        if not matches:
            print(f"[SKIP] {patient_id}: cartella DICOM non trovata")
            return None, None
        patient_dicom = matches[0]

    try:
        volume_raw, spacing, _ = load_dicom_volume(patient_dicom)
    except Exception as e:
        print(f"[SKIP] {patient_id}: errore lettura DICOM → {e}")
        return None, None

    volume_norm = window_and_normalize(volume_raw)

    anno_files = find_annotation_files(anno_root, patient_id)
    if not anno_files:
        for d in anno_root.iterdir():
            if patient_id.replace("_", "").lower() in d.name.replace("_", "").lower():
                anno_files = find_annotation_files(anno_root, d.name)
                break

    gt_boxes = load_gt_boxes(anno_files, spacing)
    patches_all, labels_all = [], []

    for z in tqdm(range(volume_norm.shape[0]),desc=f"{patient_id} slices", leave=False):
        slice_2d = volume_norm[z]
        if FIXED_N_FLAG:
            patches, coords = extract_fixed_number_grid(
                slice_2d, PATCH_H, PATCH_W, STRIDE, N_PATCH_FIXED)
        else:
            patches, coords = extract_grid_patches(
                slice_2d, PATCH_H, PATCH_W, STRIDE)

        for patch, (r, c) in zip(patches, coords):
            lbl = check_if_positive(z, r + PATCH_H // 2, c + PATCH_W // 2, gt_boxes)
            patches_all.append(patch)
            labels_all.append(lbl)

    if not patches_all:
        return None, None

    data_arr   = np.array(patches_all, dtype=np.float32)
    labels_arr = np.array(labels_all,  dtype=np.float32)
    n_pos = int(np.sum(labels_arr == 1))
    n_neg = int(np.sum(labels_arr == 0))
    print(f"{patient_id}: {len(labels_arr)} patch (pos={n_pos}, neg={n_neg})")

    tmp_data   = tmp_dir / f"{patient_id}_data.npy"
    tmp_labels = tmp_dir / f"{patient_id}_labels.npy"
    np.save(str(tmp_data),   data_arr)
    np.save(str(tmp_labels), labels_arr)
    del patches_all, labels_all, data_arr, labels_arr, volume_raw, volume_norm
    return tmp_data, tmp_labels

def process_patient(patient_id, dicom_root, anno_root, cand_root=None,patch_h=PATCH_H, patch_w=PATCH_W,neg_ratio=NEG_RATIO, apply_clahe_flag=CLAHE_FLAG,rng=None, tmp_dir=None):
    if SAMPLE_FLAG:
        return process_patient_sample(patient_id, dicom_root, anno_root, cand_root, patch_h, patch_w, neg_ratio, apply_clahe_flag, rng)
    else:
        return process_patient_grid(patient_id, dicom_root, anno_root, tmp_dir)

def save_split_sample(patches, labels, output_dir, split_name):
    if not patches:
        print(f"[WARN] split '{split_name}' vuoto.")
        return None, None
    data_arr   = np.array(patches, dtype=np.float32)
    labels_arr = np.array(labels,  dtype=np.float32)
    data_path   = output_dir / f"sampled_linfonodi_{split_name}_data.npy"
    labels_path = output_dir / f"sampled_linfonodi_{split_name}_label.npy"
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
    idx = rng.permutation(total)
    all_data   = all_data[idx]
    all_labels = all_labels[idx]
    suffix = "fixed" if FIXED_N_FLAG else "classic"
    data_path   = output_dir / f"linfonodi_{suffix}_{split_name}_data.npy"
    labels_path = output_dir / f"linfonodi_{suffix}_{split_name}_label.npy"
    np.save(str(data_path),   all_data)
    np.save(str(labels_path), all_labels)
    n_pos = int(np.sum(all_labels == 1))
    n_neg = int(np.sum(all_labels == 0))
    print(f"  → {split_name}: {total} patch (pos={n_pos}, neg={n_neg})")
    del all_data, all_labels
    return data_path, labels_path


def write_report(output_dir, train_pts, test_pts, patches_per_split):
    report_path = output_dir / "split_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("NIH CT Lymph Nodes — Split Report\n")
        f.write("=" * 60 + "\n\n")
        mode_str = "SAMPLE" if SAMPLE_FLAG else ("GRID_FIXED" if FIXED_N_FLAG else "GRID_CLASSIC")
        f.write(f"Modalità: {mode_str}  |  hard_neg_ratio={HARD_NEG_RATIO}\n\n")
        for split_name, pts in [("TRAIN", train_pts), ("TEST", test_pts)]:
            info = patches_per_split.get(split_name, {})
            f.write(
                f"[{split_name}]\n"
                f"Pazienti     : {len(pts)}\n"
                f"Patch totali : {info.get('total', 0)}\n"
                f"Positivi     : {info.get('pos', 0)}\n"
                f"Negativi     : {info.get('neg', 0)}\n"
                )

def run_demo():
    rng = np.random.default_rng(SEED)

    root       = Path(DATASET_ROOT_DIR)
    output_dir = Path(DATASET_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    dicom_root = root / "CT-Lymph-Nodes"
    anno_root  = root / "MED_ABD_LYMPH_ANNOTATIONS"

    if not dicom_root.exists():
        candidates = [d for d in root.iterdir() if d.is_dir() and "lymph" in d.name.lower() and "annotation" not in d.name.lower() and "candidate" not in d.name.lower()]
        dicom_root = candidates[0] if candidates else root

    if not anno_root.exists():
        anno_candidates = [d for d in root.iterdir() if d.is_dir() and "annotation" in d.name.lower()]
        anno_root = anno_candidates[0] if anno_candidates else None

    mode_str = "SAMPLE" if SAMPLE_FLAG else ("GRID_FIXED" if FIXED_N_FLAG else "GRID_CLASSIC")
    print(f"[DEMO] Paziente: {DEMO_PATIENT_ID}")
    print(f"[DEMO] Modalità: {mode_str}")
    print(f"[DEMO] DICOM root: {dicom_root}")
    print(f"[DEMO] Anno root: {anno_root}")
    print(f"[DEMO] Output: {output_dir}")
    print(f"[DEMO] Patch size: {PATCH_H}×{PATCH_W} px")
    print(f"[DEMO] Numero patches: {DEMO_N_PATCHES}")
    print(f"[DEMO] CLAHE: {'TRUE' if CLAHE_FLAG else 'FALSE'}\n")

    patient_dicom = dicom_root / DEMO_PATIENT_ID
    if not patient_dicom.exists():
        matches = list(dicom_root.glob(f"*{DEMO_PATIENT_ID}*"))
        if not matches:
            raise FileNotFoundError(
                f"Cartella DICOM non trovata per {DEMO_PATIENT_ID}")
        patient_dicom = matches[0]

    print(f"Carico volume DICOM da {patient_dicom} …")
    volume_raw, spacing, origin = load_dicom_volume(patient_dicom)
    print(f"  Volume shape : {volume_raw.shape}  spacing: {spacing}")
    volume_norm = window_and_normalize(volume_raw)

    anno_files = find_annotation_files(anno_root, DEMO_PATIENT_ID)
    print(f"File annotazioni : {[f.name for f in anno_files]}")
    if not anno_files:
        raise FileNotFoundError(f"Nessun file *indices*.txt trovato per {DEMO_PATIENT_ID}")

    print("\nEstrazione patch positive …")
    if SAMPLE_FLAG:
        patches, labels = extract_demo_positive_patches(volume_norm, anno_files, PATCH_H, PATCH_W, CLAHE_FLAG,n_target=DEMO_N_PATCHES, rng=rng)
        data_path   = output_dir / "sampled_demo_positive_patches.npy"
        labels_path = output_dir / "sampled_demo_positive_labels.npy"
    else:
        gt_boxes = load_gt_boxes(anno_files, spacing)
        patches, labels = [], []
        for z in tqdm(range(volume_norm.shape[0]), desc="slices"):
            slice_2d = volume_norm[z]
            if FIXED_N_FLAG:
                ps, coords = extract_fixed_number_grid(
                    slice_2d, PATCH_H, PATCH_W, STRIDE, N_PATCH_FIXED)
            else:
                ps, coords = extract_grid_patches(
                    slice_2d, PATCH_H, PATCH_W, STRIDE)
            for p, (r, c) in zip(ps, coords):
                lbl = check_if_positive(
                    z, r + PATCH_H // 2, c + PATCH_W // 2, gt_boxes)
                patches.append(p)
                labels.append(lbl)
        data_path   = output_dir / "demo_grid_patches.npy"
        labels_path = output_dir / "demo_grid_labels.npy"

    data_arr   = np.array(patches, dtype=np.float32)
    labels_arr = np.array(labels,  dtype=np.float32)
    np.save(str(data_path),   data_arr)
    np.save(str(labels_path), labels_arr)
    print(f"Patch salvate  : {data_path}, shape={data_arr.shape}")
    print(f"Label salvate  : {labels_path}, shape={labels_arr.shape}")

    print("Generazione bounding box Ground Truth")
    indices_files  = [f for f in anno_files if 'indices' in f.name.lower()]
    real_centroids = []
    for f in indices_files:
        real_centroids.extend(parse_voxel_annotation_file(f))

    sizes_file = find_sizes_file(anno_root, DEMO_PATIENT_ID)
    if sizes_file:
        sizes_mm = parse_sizes_file(sizes_file)
        print(f"Sizes file: {sizes_file.name}, {len(sizes_mm)} linfonodi")
    else:
        print("[WARN] *_sizes.txt non trovato: uso box default 15×15 mm")
        sizes_mm = [(15.0, 15.0)] * len(real_centroids)

    n_ln = min(len(real_centroids), len(sizes_mm))
    real_centroids = real_centroids[:n_ln]
    sizes_mm       = sizes_mm[:n_ln]

    try:
        first_dcm = pydicom.dcmread(
            str(get_ct_series(patient_dicom)[0]), stop_before_pixels=True)
        dy, dx = [float(v) for v in first_dcm.PixelSpacing]
    except Exception:
        dy, dx = 1.0, 1.0
        print("[WARN] PixelSpacing non letto: uso 1.0 mm/px")
    print(f"PixelSpacing: dy={dy:.4f} mm  dx={dx:.4f} mm")

    gt_boxes = compute_gt_boxes_pixel(
        real_centroids, sizes_mm, spacing_yx=(dy, dx))

    print(f"Salvataggio immagini GT in {output_dir}")
    saved_imgs = render_gt_bboxes(
        volume_norm, gt_boxes,
        output_dir=output_dir, patient_id=DEMO_PATIENT_ID,
        patch_h=PATCH_H, patch_w=PATCH_W,
        pred_boxes_per_ln=None
    )
    print(f"\n{len(saved_imgs)}immagini GT salvate.")
    for p in saved_imgs:
        print(f"   {p.name}")

    if SHOW_FLAG:
        debug_show_patches(patches, labels, n=10)

    return data_path, labels_path

def main_preprocessing():
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    root       = Path(DATASET_ROOT_DIR)
    output_dir = Path(DATASET_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    dicom_root = root / "CT-Lymph-Nodes"
    anno_root  = root / "MED_ABD_LYMPH_ANNOTATIONS"
    cand_root  = root / "MED_ABD_LYMPH_CANDIDATES"

    if not dicom_root.exists():
        candidates = [d for d in root.iterdir()
                      if d.is_dir() and "lymph" in d.name.lower()
                      and "annotation" not in d.name.lower()
                      and "candidate" not in d.name.lower()]
        dicom_root = candidates[0] if candidates else root

    if not anno_root.exists():
        anno_candidates = [d for d in root.iterdir()
                           if d.is_dir() and "annotation" in d.name.lower()]
        anno_root = anno_candidates[0] if anno_candidates else None

    if not cand_root.exists():
        cand_root = None

    mode_str = "SAMPLE" if SAMPLE_FLAG else ("GRID_FIXED" if FIXED_N_FLAG else "GRID_CLASSIC")
    print(f"Modalità: {mode_str}")
    print(f"Hard neg ratio: {HARD_NEG_RATIO:.0%} dei negativi da sliding window")
    print(f"Neg/Pos ratio: {NEG_RATIO}:1")
    print(f"CLAHE: {'TRUE' if CLAHE_FLAG else 'FALSE'}\n")

    all_patients = sorted([d.name for d in dicom_root.iterdir() if d.is_dir()])
    print(f"Pazienti trovati: {len(all_patients)}\n")

    train_pts, test_pts = stratified_train_test_split(all_patients)
    all_splits = {"train": train_pts, "test": test_pts}
    patches_per_split = {}

    tmp_dir = output_dir / "_tmp_patients"
    if not SAMPLE_FLAG:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    if SAMPLE_FLAG:
        for split_name, patient_list in all_splits.items():
            print(f"{split_name.upper()} ({len(patient_list)} pazienti)")
            split_patches, split_labels = [], []

            for patient_id in patient_list:
                patches, labels = process_patient(
                    patient_id, dicom_root, anno_root, cand_root,
                    PATCH_H, PATCH_W, NEG_RATIO, CLAHE_FLAG, rng)
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
            print()

        if SHOW_FLAG:
            debug_show_patches(split_patches, split_labels, n=10)

    else:
        for split_name, patient_list in all_splits.items():
            print(f"{split_name.upper()} ({len(patient_list)} pazienti)")
            tmp_pairs = []
            for patient_id in patient_list:
                d_path, l_path = process_patient(
                    patient_id, dicom_root, anno_root, tmp_dir=tmp_dir)
                tmp_pairs.append((d_path, l_path))
            save_split_grid(tmp_pairs, output_dir, split_name, rng)
            suffix   = "fixed" if FIXED_N_FLAG else "classic"
            lbl_path = output_dir / f"linfonodi_{suffix}_{split_name}_label.npy"
            if lbl_path.exists():
                lbl_arr = np.load(str(lbl_path))
                patches_per_split[split_name.upper()] = {
                    "total": len(lbl_arr),
                    "pos": int(np.sum(lbl_arr == 1)),
                    "neg": int(np.sum(lbl_arr == 0))}
                del lbl_arr
            print()

        print("Pulizia file temporanei")
        for f in tmp_dir.glob("*.npy"):
            f.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
        print("terminato.\n")

    write_report(output_dir, train_pts, test_pts, patches_per_split)

    print("RIEPILOGO FINALE")
    for split_name, info in patches_per_split.items():
        bal = info['pos'] / info['total'] * 100 if info['total'] > 0 else 0
        print(f"{split_name:6s}: {info['total']:5d} patch | pos={info['pos']:4d} ({bal:.0f}%)| neg={info['neg']:4d}")

if __name__ == "__main__":
    if DEMO_FLAG:
        run_demo()
    else:
        main_preprocessing()