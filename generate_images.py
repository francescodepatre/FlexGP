"""
generate_gt_samples.py
======================
Genera 10 immagini di slice CT (5 positive, 5 negative) SENZA patch,
più le stesse 10 immagini CON bounding box Ground Truth sovrapposti.

Output (in OUTPUT_DIR):
  pos_clean_01..05.png   → slice positive senza annotazioni
  neg_clean_01..05.png   → slice negative senza annotazioni
  pos_bbox_01..05.png    → slice positive con bounding box GT (verde)
  neg_bbox_01..05.png    → slice negative con bounding box GT (rosso = nessun LN in slice)

Compatibile con la tua pipeline esistente (stesso preprocessing).
"""

import re
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pydicom
from skimage.exposure import equalize_adapthist

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
#  CONFIGURAZIONE — modifica questi percorsi
# ═══════════════════════════════════════════════════════════
DATASET_ROOT_DIR = '../TCIA_CT_Lymph_Nodes_03-31-2023'
OUTPUT_DIR       = './immagini/'
PATIENT_ID       = 'ABD_LYMPH_001'    # paziente da usare

N_POSITIVE = 5   # numero di slice positive da salvare
N_NEGATIVE = 5   # numero di slice negative da salvare

HU_MIN     = -200.0
HU_MAX     =  300.0
CLAHE_FLAG = True
SEED       = 42
DPI        = 150

# ═══════════════════════════════════════════════════════════
#  LETTURA DICOM  (identica alla tua pipeline)
# ═══════════════════════════════════════════════════════════

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

    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except Exception:
        try:
            slices.sort(key=lambda s: int(s.InstanceNumber))
        except Exception:
            pass

    arrays = []
    for ds in slices:
        arr   = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope',     1.0))
        inter = float(getattr(ds, 'RescaleIntercept', 0.0))
        arrays.append(arr * slope + inter)

    volume = np.stack(arrays, axis=0)

    try:
        dy, dx = [float(x) for x in slices[0].PixelSpacing]
    except Exception:
        dy, dx = 1.0, 1.0

    try:
        dz = abs(float(slices[1].ImagePositionPatient[2]) -
                 float(slices[0].ImagePositionPatient[2]))
    except Exception:
        dz = float(getattr(slices[0], 'SliceThickness', 1.0))

    spacing = np.array([dz, dy, dx])
    return volume, spacing, slices


def window_and_normalize(volume, hu_min=HU_MIN, hu_max=HU_MAX):
    vol = np.clip(volume, hu_min, hu_max)
    return ((vol - hu_min) / (hu_max - hu_min)).astype(np.float32)


def apply_clahe(slice_2d):
    return equalize_adapthist(slice_2d.astype(np.float64), clip_limit=0.03)

# ═══════════════════════════════════════════════════════════
#  LETTURA ANNOTAZIONI  (identica alla tua pipeline)
# ═══════════════════════════════════════════════════════════

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
                        centroids.append((int(parts[2]), int(parts[1]), int(parts[0])))
                    except ValueError:
                        continue
    except Exception:
        return []
    return centroids


def parse_sizes_file(filepath: Path):
    sizes = []
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


def find_annotation_files(anno_root: Path, patient_id: str):
    patient_anno = anno_root / patient_id
    if not patient_anno.exists():
        for d in anno_root.iterdir():
            if d.is_dir() and d.name.lower() == patient_id.lower():
                patient_anno = d
                break
        else:
            return []
    files = sorted(patient_anno.glob("*indices*.txt"))
    return files if files else sorted(patient_anno.glob("*.txt"))


def compute_gt_boxes(centroids_zyx, sizes_mm, spacing_yx):
    dy, dx = spacing_yx
    boxes  = []
    for (cz, cy, cx), (short_mm, long_mm) in zip(centroids_zyx, sizes_mm):
        half_r = max(1, round(long_mm / 2.0 / dy))
        half_c = max(1, round(long_mm / 2.0 / dx))
        boxes.append(dict(
            cz=cz, cy=cy, cx=cx,
            r0=cy - half_r, c0=cx - half_c,
            r1=cy + half_r, c1=cx + half_c,
            short_mm=short_mm, long_mm=long_mm
        ))
    return boxes

# ═══════════════════════════════════════════════════════════
#  SALVATAGGIO IMMAGINI
# ═══════════════════════════════════════════════════════════

def save_slice_clean(slice_img, filepath: Path, title: str):
    """Salva la slice CT senza annotazioni."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(slice_img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(str(filepath), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {filepath.name}")


def save_slice_with_bbox(slice_img, boxes_on_slice, filepath: Path,
                         title: str, is_positive: bool):
    """Salva la slice CT con bounding box GT sovrapposti."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(slice_img, cmap='gray', vmin=0, vmax=1)

    if is_positive and boxes_on_slice:
        for box in boxes_on_slice:
            w = box['c1'] - box['c0']
            h = box['r1'] - box['r0']
            ax.add_patch(mpatches.Rectangle(
                (box['c0'], box['r0']), w, h,
                linewidth=2, edgecolor='lime', facecolor='none', label='GT LN'
            ))
            # etichetta dimensioni
            ax.text(box['c0'], box['r0'] - 4,
                    f"{box['short_mm']:.0f}×{box['long_mm']:.0f}mm",
                    color='lime', fontsize=7,
                    bbox=dict(facecolor='black', alpha=0.4, pad=1))
        handles = [mpatches.Patch(edgecolor='lime', facecolor='none', label='GT Linfonodo')]
        ax.legend(handles=handles, loc='upper right', fontsize=8)
    else:
        # slice negativa: scrivi "No LN" nell'angolo
        ax.text(5, 15, "No LN in questa slice",
                color='tomato', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, pad=2))

    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(str(filepath), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {filepath.name}")

# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    root       = Path(DATASET_ROOT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── percorsi dataset ────────────────────────────────────
    dicom_root = root / "CT_Lymph_Nodes"
    anno_root  = root / "MED_ABD_LYMPH_ANNOTATIONS"

    if not dicom_root.exists():
        candidates = [d for d in root.iterdir()
                      if d.is_dir()
                      and "lymph" in d.name.lower()
                      and "annotation" not in d.name.lower()]
        dicom_root = candidates[0] if candidates else root

    if not anno_root.exists():
        anno_candidates = [d for d in root.iterdir()
                           if d.is_dir() and "annotation" in d.name.lower()]
        anno_root = anno_candidates[0] if anno_candidates else None

    # ── carica volume ───────────────────────────────────────
    patient_dicom = dicom_root / PATIENT_ID
    if not patient_dicom.exists():
        matches = list(dicom_root.glob(f"*{PATIENT_ID}*"))
        if not matches:
            raise FileNotFoundError(f"Cartella DICOM non trovata per {PATIENT_ID}")
        patient_dicom = matches[0]

    print(f"Carico volume DICOM da: {patient_dicom}")
    volume_raw, spacing, slices_ds = load_dicom_volume(patient_dicom)
    print(f"  Volume shape : {volume_raw.shape}")
    print(f"  Spacing (mm) : {spacing}")

    volume_norm = window_and_normalize(volume_raw)

    # ── carica annotazioni ──────────────────────────────────
    anno_files = find_annotation_files(anno_root, PATIENT_ID)
    if not anno_files:
        raise FileNotFoundError(f"Nessun file *indices*.txt trovato per {PATIENT_ID}")
    print(f"File annotazioni: {[f.name for f in anno_files]}")

    all_centroids = []
    for f in anno_files:
        all_centroids.extend(parse_voxel_annotation_file(f))
    print(f"  Centroidi trovati: {len(all_centroids)}")

    # ── carica sizes ────────────────────────────────────────
    patient_anno_dir = anno_root / PATIENT_ID
    sizes_files      = list(patient_anno_dir.glob("*sizes*.txt"))
    if sizes_files:
        sizes_mm = parse_sizes_file(sizes_files[0])
        print(f"  Sizes file: {sizes_files[0].name}  → {len(sizes_mm)} LN")
    else:
        print("  [WARN] sizes file non trovato → uso default 15×15 mm")
        sizes_mm = [(15.0, 15.0)] * len(all_centroids)

    n_ln = min(len(all_centroids), len(sizes_mm))
    gt_boxes = compute_gt_boxes(
        all_centroids[:n_ln], sizes_mm[:n_ln],
        spacing_yx=(spacing[1], spacing[2])
    )

    # ── identifica slice positive e negative ────────────────
    # Positiva = ha almeno un linfonodo annotato
    positive_slices = sorted({box['cz'] for box in gt_boxes
                               if 0 <= box['cz'] < volume_norm.shape[0]})

    all_slice_idxs  = set(range(volume_norm.shape[0]))
    negative_slices = sorted(all_slice_idxs - set(positive_slices))

    print(f"\nSlice positive disponibili : {len(positive_slices)}")
    print(f"Slice negative disponibili : {len(negative_slices)}")

    # campiona N_POSITIVE e N_NEGATIVE slice
    chosen_pos = list(rng.choice(positive_slices,
                                  size=min(N_POSITIVE, len(positive_slices)),
                                  replace=False))
    chosen_neg = list(rng.choice(negative_slices,
                                  size=min(N_NEGATIVE, len(negative_slices)),
                                  replace=False))

    print(f"\nSlice positive scelte : {chosen_pos}")
    print(f"Slice negative scelte : {chosen_neg}")

    # mappa slice → bounding box che insistono su quella slice
    def boxes_on(z):
        return [b for b in gt_boxes if b['cz'] == z]

    # ── genera e salva immagini ─────────────────────────────
    print(f"\nSalvo immagini in: {output_dir}\n")

    # POSITIVE
    print("── POSITIVE ──────────────────────────────────────")
    for i, z in enumerate(chosen_pos, start=1):
        sl = volume_norm[z]
        if CLAHE_FLAG:
            sl = apply_clahe(sl)

        bboxes = boxes_on(z)
        label  = f"Positivo #{i}  (slice {z})  LN={len(bboxes)}"

        # senza bbox
        save_slice_clean(
            sl,
            output_dir / f"pos_clean_{i:02d}.png",
            title=label
        )
        # con bbox GT
        save_slice_with_bbox(
            sl, bboxes,
            output_dir / f"pos_bbox_{i:02d}.png",
            title=label + " — GT BBox",
            is_positive=True
        )

    # NEGATIVE
    print("\n── NEGATIVE ──────────────────────────────────────")
    for i, z in enumerate(chosen_neg, start=1):
        sl = volume_norm[z]
        if CLAHE_FLAG:
            sl = apply_clahe(sl)

        label = f"Negativo #{i}  (slice {z})  LN=0"

        # senza bbox
        save_slice_clean(
            sl,
            output_dir / f"neg_clean_{i:02d}.png",
            title=label
        )
        # con testo "No LN"
        save_slice_with_bbox(
            sl, [],
            output_dir / f"neg_bbox_{i:02d}.png",
            title=label + " — GT BBox",
            is_positive=False
        )

    # ── riepilogo ────────────────────────────────────────────
    all_files = sorted(output_dir.glob("*.png"))
    print(f"\n{'='*50}")
    print(f"RIEPILOGO  —  {len(all_files)} immagini salvate in {output_dir}")
    print(f"{'='*50}")
    print(f"  pos_clean_*.png  : {N_POSITIVE} slice positive SENZA bbox")
    print(f"  neg_clean_*.png  : {N_NEGATIVE} slice negative SENZA bbox")
    print(f"  pos_bbox_*.png   : {N_POSITIVE} slice positive CON bbox GT (lime)")
    print(f"  neg_bbox_*.png   : {N_NEGATIVE} slice negative CON etichetta 'No LN'")


if __name__ == "__main__":
    main()