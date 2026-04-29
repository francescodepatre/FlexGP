from matplotlib.patches import Rectangle
from matplotlib.image import imread
from scipy.ndimage import label
import matplotlib.pyplot as plt
import numpy as np
import dill as d
import random
import gp_restrict as gp_restrict
from deap import base, creator, tools, gp
from strongGPDataType import Int1, Int2, Int3, Float1, Float2, Float3, Img, Img1, Vector
import fgp_functions as fe_fs
import warnings

warnings.filterwarnings("ignore")

MODELS_DIR= './models/'
PATCH_WIDTH= 50
PATCH_HEIGHT= 50
STRIDE= 25 # 50% overlap
SVM_THRESHOLD= 1.5 # soglia su decision_function (alzare per meno FP)
MIN_CLUSTER_PATCHES= 2    # finestre positive minime per tenere un cluster
NMS_IOU_THRESHOLD= 0.3  # IoU sopra cui una box viene soppressa
OUTPUT_DIR= './results/'
INITIAL_MIN_DEPTH= 2
INITIAL_MAX_DEPTH= 6
IMAGE_PATH = './immagini/pos_clean_02.png'
MODEL_NAME = 'modello_funzionante.pkl'

pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')
pset.addPrimitive(fe_fs.root_conVector2, [Img1, Img1], Vector, name='Root2')
pset.addPrimitive(fe_fs.root_conVector3, [Img1, Img1, Img1], Vector, name='Root3')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector, name='Roots2')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector, name='Roots3')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector, Vector], Vector, name='Roots4')
pset.addPrimitive(fe_fs.global_hog_small, [Img1], Vector, name='Global_HOG')
pset.addPrimitive(fe_fs.all_lbp,          [Img1], Vector, name='Global_uLBP')
pset.addPrimitive(fe_fs.all_sift,         [Img1], Vector, name='Global_SIFT')
pset.addPrimitive(fe_fs.global_hog_small, [Img],  Vector, name='FGlobal_HOG')
pset.addPrimitive(fe_fs.all_lbp,          [Img],  Vector, name='FGlobal_uLBP')
pset.addPrimitive(fe_fs.all_sift,         [Img],  Vector, name='FGlobal_SIFT')
pset.addPrimitive(fe_fs.maxP,    [Img1, Int3, Int3], Img1, name='MaxPF')
pset.addPrimitive(fe_fs.gau,     [Img1, Int1],        Img1, name='GauF')
pset.addPrimitive(fe_fs.gauD,    [Img1, Int1, Int2, Int2], Img1, name='GauDF')
pset.addPrimitive(fe_fs.gab,     [Img1, Float1, Float2],   Img1, name='GaborF')
pset.addPrimitive(fe_fs.laplace, [Img1], Img1, name='LapF')
pset.addPrimitive(fe_fs.gaussian_Laplace1, [Img1], Img1, name='LoG1F')
pset.addPrimitive(fe_fs.gaussian_Laplace2, [Img1], Img1, name='LoG2F')
pset.addPrimitive(fe_fs.sobelxy, [Img1], Img1, name='SobelF')
pset.addPrimitive(fe_fs.sobelx,  [Img1], Img1, name='SobelXF')
pset.addPrimitive(fe_fs.sobely,  [Img1], Img1, name='SobelYF')
pset.addPrimitive(fe_fs.medianf, [Img1], Img1, name='MedF')
pset.addPrimitive(fe_fs.meanf,   [Img1], Img1, name='MeanF')
pset.addPrimitive(fe_fs.minf,    [Img1], Img1, name='MinF')
pset.addPrimitive(fe_fs.maxf,    [Img1], Img1, name='MaxF')
pset.addPrimitive(fe_fs.lbp,     [Img1], Img1, name='LBPF')
pset.addPrimitive(fe_fs.hog_feature, [Img1], Img1, name='HoGF')
pset.addPrimitive(fe_fs.mixconadd, [Img1, Float3, Img1, Float3], Img1, name='W-AddF')
pset.addPrimitive(fe_fs.mixconsub, [Img1, Float3, Img1, Float3], Img1, name='W-SubF')
pset.addPrimitive(fe_fs.sqrt, [Img1], Img1, name='SqrtF')
pset.addPrimitive(fe_fs.relu, [Img1], Img1, name='ReLUF')
pset.addPrimitive(fe_fs.maxP,    [Img, Int3, Int3], Img1, name='MaxP')
pset.addPrimitive(fe_fs.gau,     [Img, Int1],  Img,  name='Gau')
pset.addPrimitive(fe_fs.gauD,    [Img, Int1, Int2, Int2], Img, name='GauD')
pset.addPrimitive(fe_fs.gab,     [Img, Float1, Float2],   Img, name='Gabor')
pset.addPrimitive(fe_fs.laplace, [Img], Img, name='Lap')
pset.addPrimitive(fe_fs.gaussian_Laplace1, [Img], Img, name='LoG1')
pset.addPrimitive(fe_fs.gaussian_Laplace2, [Img], Img, name='LoG2')
pset.addPrimitive(fe_fs.sobelxy, [Img], Img, name='Sobel')
pset.addPrimitive(fe_fs.sobelx,  [Img], Img, name='SobelX')
pset.addPrimitive(fe_fs.sobely,  [Img], Img, name='SobelY')
pset.addPrimitive(fe_fs.medianf, [Img], Img, name='Med')
pset.addPrimitive(fe_fs.meanf,   [Img], Img, name='Mean')
pset.addPrimitive(fe_fs.minf,    [Img], Img, name='Min')
pset.addPrimitive(fe_fs.maxf,    [Img], Img, name='Max')
pset.addPrimitive(fe_fs.lbp,     [Img], Img, name='LBP-F')
pset.addPrimitive(fe_fs.hog_feature, [Img], Img, name='HOG-F')
pset.addPrimitive(fe_fs.mixconadd, [Img, Float3, Img, Float3], Img, name='W-Add')
pset.addPrimitive(fe_fs.mixconsub, [Img, Float3, Img, Float3], Img, name='W-Sub')
pset.addPrimitive(fe_fs.sqrt, [Img], Img, name='Sqrt')
pset.addPrimitive(fe_fs.relu, [Img], Img, name='ReLU')
pset.renameArguments(ARG0='Image')
pset.addEphemeralConstant('Singma',     lambda: random.randint(1, 4),        Int1)
pset.addEphemeralConstant('Order',      lambda: random.randint(0, 3),        Int2)
pset.addEphemeralConstant('Theta',      lambda: random.randint(0, 8),        Float1)
pset.addEphemeralConstant('Frequency',  lambda: random.randint(0, 5),        Float2)
pset.addEphemeralConstant('n',          lambda: round(random.random(), 3),   Float3)
pset.addEphemeralConstant('KernelSize', lambda: random.randrange(2, 5, 2),   Int3)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("expr",       gp_restrict.genHalfAndHalfMD, pset=pset, min_=INITIAL_MIN_DEPTH, max_=INITIAL_MAX_DEPTH)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile",    gp.compile, pset=pset)
toolbox.register("mapp",       map)
toolbox.register("select",        tools.selTournament, tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate",          gp.cxOnePoint)
toolbox.register("expr_mut",      gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate",        gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def normalize_feature(feat_raw, feat_p1, feat_p99, scaler):
    feat_clipped = np.clip(feat_raw, feat_p1, feat_p99)
    return scaler.transform(feat_clipped)

def sliding_window(image, patch_h, patch_w, stride):
    H, W = image.shape[:2]
    for r in range(0, H - patch_h + 1, stride):
        for c in range(0, W - patch_w + 1, stride):
            yield r, c, image[r:r + patch_h, c:c + patch_w]

def iou(a, b):
    r0 = max(a[0], b[0]); c0 = max(a[1], b[1])
    r1 = min(a[2], b[2]); c1 = min(a[3], b[3])
    inter = max(0, r1 - r0) * max(0, c1 - c0)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections, iou_thr):
    if not detections:
        return []
    dets = sorted(detections, key=lambda x: x['score'], reverse=True)
    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets if iou(best['box'], d['box']) < iou_thr]
    return kept

def cluster_to_boxes(positive_wins, H, W, patch_h, patch_w, min_wins):
    score_map = np.full((H, W), -np.inf)
    for r0, c0, score in positive_wins:
        r1 = min(r0 + patch_h, H)
        c1 = min(c0 + patch_w, W)
        score_map[r0:r1, c0:c1] = np.maximum(score_map[r0:r1, c0:c1], score)

    binary = (score_map > -np.inf).astype(int)
    labeled, n_feat = label(binary, structure=np.ones((3, 3), dtype=int))

    boxes = []
    for rid in range(1, n_feat + 1):
        n_wins = sum(1 for r0, c0, _ in positive_wins
                     if labeled[r0, c0] == rid)
        if n_wins < min_wins:
            continue
        pos = np.argwhere(labeled == rid)
        r_min, c_min = pos.min(axis=0)
        r_max, c_max = pos.max(axis=0) + 1
        cluster_scores = [s for r0, c0, s in positive_wins
                          if labeled[r0, c0] == rid]
        boxes.append({
            'box':   (int(r_min), int(c_min), int(r_max), int(c_max)),
            'score': float(max(cluster_scores))
        })
    return boxes

def main():
    image = imread(IMAGE_PATH)
    if image.ndim == 3:
        image = image[..., 0]

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    with open(MODELS_DIR + MODEL_NAME, "rb") as f:
        data = d.load(f)

    loaded_tree = data["tree"]
    loaded_scaler = data["scaler"]
    loaded_svm = data["lsvm"]
    feat_p1 = data["feat_p1"]   # percentile 1  salvato dal training
    feat_p99 = data["feat_p99"]  # percentile 99 salvato dal training

    best_function = toolbox.compile(expr=loaded_tree)

    H, W = image.shape
    print(f"Immagine: {H}×{W} | dimensioni patch: {PATCH_HEIGHT}×{PATCH_WIDTH} | stride: {STRIDE}")
    print(f"feat_p1: [{feat_p1.min():.4f}, {feat_p1.max():.4f}]")
    print(f"feat_p99: [{feat_p99.min():.4f}, {feat_p99.max():.4f}]")

    positive_wins = []
    all_scores    = []

    for r0, c0, patch in sliding_window(image, PATCH_HEIGHT, PATCH_WIDTH, STRIDE):
        feat_raw  = np.asarray(best_function(patch)).reshape(1, -1)
        feat_norm = normalize_feature(feat_raw, feat_p1, feat_p99, loaded_scaler)
        score     = loaded_svm.decision_function(feat_norm)[0]
        all_scores.append(score)
        if score > SVM_THRESHOLD:
            positive_wins.append((r0, c0, score))

    print(f"\nFinestre totali: {len(all_scores)}")
    print(f"Finestre positive: {len(positive_wins)}  (soglia={SVM_THRESHOLD})")
    print(f"Score: min={min(all_scores):.3f}  max={max(all_scores):.3f} mean={np.mean(all_scores):.3f}")

    raw_boxes   = cluster_to_boxes(positive_wins, H, W, PATCH_HEIGHT, PATCH_WIDTH, MIN_CLUSTER_PATCHES)
    final_boxes = nms(raw_boxes, NMS_IOU_THRESHOLD)

    print(f"Cluster dopo filtraggio dimensione: {len(raw_boxes)}")
    print(f"Bounding box dopo NMS: {len(final_boxes)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    score_map_vis = np.full((H, W), np.nan)
    for r0, c0, score in positive_wins:
        r1 = min(r0 + PATCH_HEIGHT, H)
        c1 = min(c0 + PATCH_WIDTH,  W)
        score_map_vis[r0:r1, c0:c1] = np.fmax(score_map_vis[r0:r1, c0:c1], score)

    axes[0].imshow(image, cmap='gray')
    vmax = max(all_scores) if max(all_scores) > SVM_THRESHOLD else SVM_THRESHOLD + 0.1
    hm = axes[0].imshow(score_map_vis, cmap='hot', alpha=0.5,vmin=SVM_THRESHOLD, vmax=vmax)
    plt.colorbar(hm, ax=axes[0], fraction=0.03, label='SVM score')
    axes[0].set_title("Heatmap score SVM (sliding window)", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(image, cmap='gray')
    for det in final_boxes:
        r0, c0, r1, c1 = det['box']
        axes[1].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0,linewidth=2, edgecolor='lime', facecolor='none'))
        axes[1].text(c0, max(r0 - 4, 0), f"{det['score']:.2f}",color='lime', fontsize=7, fontweight='bold')

    axes[1].set_title(f"Bounding boxes finali — {len(final_boxes)} rilevato/i", fontsize=10)
    axes[1].axis('off')

    plt.suptitle(f"Sliding window (stride={STRIDE}) + clip percentile + NMS (IoU>{NMS_IOU_THRESHOLD})",fontsize=11, y=1.01)
    plt.tight_layout()
    out_path = OUTPUT_DIR + "output_sliding_nms.png"
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Salvato in: {out_path}")


if __name__ == "__main__":
    main()