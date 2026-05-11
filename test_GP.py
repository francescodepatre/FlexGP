from matplotlib.patches import Rectangle
from scipy.ndimage import label
import matplotlib.pyplot as plt
import numpy as np
import dill as d
import random
import gp_restrict as gp_restrict
from deap import base, creator, tools, gp
from strongGPDataType import Int1, Int2, Int3, Float1, Float2, Float3, Img, Img1, Vector
import fgp_functions as fe_fs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

import dill
import warnings
warnings.filterwarnings("ignore")

dataSetName = 'sampled_linfonodi'
directory   = './preprocessed_dataset/'
MODELS_DIR  = './models/'
DEMO_FLAG   = True
MODEL_FILE  = "modello_full.pkl"

x_test  = np.load(directory + dataSetName + '_test_data.npy')
y_test  = np.load(directory + dataSetName + '_test_label.npy')
x_train = np.load(directory + dataSetName + '_train_data.npy')
y_train = np.load(directory + dataSetName + '_train_label.npy')

# Normalizza solo se i dati sono ancora in uint8 [0, 255]
if x_train.max() > 1.0:
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32)  / 255.0

x_data_demo = np.load(directory + 'sampled_demo_positive_patches.npy')
y_data_demo = np.load(directory + 'sampled_demo_positive_labels.npy')
if x_data_demo.max() > 1.0:
    x_data_demo = x_data_demo.astype(np.float32) / 255.0

print(f"Train range : [{x_train.min():.4f}, {x_train.max():.4f}]")
print(f"Test  range : [{x_test.min():.4f},  {x_test.max():.4f}]")

population      = 200
generation      = 30
cxProb          = 0.8
mutProb         = 0.15
elitismProb     = 0.05
totalRuns       = 10
initialMinDepth = 2
initialMaxDepth = 6
maxDepth        = 10

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
toolbox.register("expr",       gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile",    gp.compile, pset=pset)
toolbox.register("mapp",       map)
toolbox.register("select",        tools.selTournament, tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate",          gp.cxOnePoint)
toolbox.register("expr_mut",      gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate",        gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def normalize_features(feat_raw, feat_p1, feat_p99, scaler):
    if feat_p1 is not None and feat_p99 is not None:
        feat_raw = np.clip(feat_raw, feat_p1, feat_p99)
    return scaler.transform(feat_raw)

def build_features(func, data, labels, feat_p1, feat_p99, scaler):
    raw_list = []
    for i in range(len(labels)):
        raw_list.append(np.asarray(func(data[i, :, :])))
    feat_raw  = np.asarray(raw_list, dtype=float)
    feat_norm = normalize_features(feat_raw, feat_p1, feat_p99, scaler)
    return feat_raw, feat_norm

def compute_saliency(func, image, feat_p1, feat_p99, scaler, svm, stride=8, baseline_val=0.0):
    h, w = image.shape
    saliency = np.zeros((h, w))

    feat_orig = np.asarray(func(image), dtype=float).reshape(1, -1)
    norm_orig = normalize_features(feat_orig, feat_p1, feat_p99, scaler)
    score_orig = svm.decision_function(norm_orig)[0]

    for i in range(0, h, stride):
        for j in range(0, w, stride):
            perturbed = image.copy()
            perturbed[i:i+stride, j:j+stride] = baseline_val
            try:
                feat_pert  = np.asarray(func(perturbed), dtype=float).reshape(1, -1)
                norm_pert  = normalize_features(feat_pert, feat_p1, feat_p99, scaler)
                score_pert = svm.decision_function(norm_pert)[0]
                value      = abs(score_orig - score_pert)
            except Exception:
                value = 0.0
            saliency[i:i+stride, j:j+stride] = value

    return saliency


def plot_heatmap(image, saliency, pred_label, true_label, title=""):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(image, cmap='gray')
    ax.imshow(saliency, cmap='hot', alpha=0.5)
    color = 'lime' if pred_label == true_label else 'red'
    ax.set_title(f"{title}\nLabel reale: {int(true_label)}\nPredizione: {int(pred_label)}", color=color, fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def get_bounding_boxes(saliency, threshold_percentile=80, min_area=30):
    thresh = np.percentile(saliency, threshold_percentile)
    mask   = saliency > thresh
    labeled_arr, n = label(mask)
    boxes = []
    for region_id in range(1, n + 1):
        coords = np.where(labeled_arr == region_id)
        if len(coords[0]) < min_area:
            continue
        r0, r1 = coords[0].min(), coords[0].max()
        c0, c1 = coords[1].min(), coords[1].max()
        boxes.append((r0, c0, r1 - r0, c1 - c0))
    return boxes


def plot_with_boxes(image, saliency, pred_label, true_label, color='cyan'):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(image, cmap='gray')
    for (r0, c0, h, w) in get_bounding_boxes(saliency):
        ax.add_patch(Rectangle(
            (c0, r0), w, h,
            linewidth=2, edgecolor=color, facecolor='none'))
    label_color = 'lime' if pred_label == true_label else 'red'
    ax.set_title(f"Bounding boxes predetti\nLabel reale: {int(true_label)} Predizione: {int(pred_label)}", color=label_color, fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_test_set(svm, feat_norm, test_labels):
    y_pred = svm.predict(feat_norm)

    acc  = accuracy_score(test_labels, y_pred)
    prec = precision_score(test_labels, y_pred, average='macro', zero_division=0)
    rec  = recall_score(test_labels, y_pred,    average='macro', zero_division=0)
    f1   = f1_score(test_labels, y_pred,        average='macro', zero_division=0)
    cm   = confusion_matrix(test_labels, y_pred)

    #Score SVM per analisi soglia
    scores = svm.decision_function(feat_norm)

    roc_auc = roc_auc_score(test_labels, scores)

    fpr, tpr, thresholds = roc_curve(test_labels, scores)

    precision, recall, thresholds = precision_recall_curve(test_labels, scores)

    pr_auc = auc(recall, precision)

    print("VALUTAZIONE SUL TEST SET")
    print(f"Campioni totali: {len(test_labels)}")
    print(f"Positivi (1): {int(np.sum(test_labels == 1))}")
    print(f"Negativi (0): {int(np.sum(test_labels == 0))}\n")
    print(f"Accuracy: {acc  * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {rec  * 100:.2f}%")
    print(f"F1-score: {f1   * 100:.2f}%")
    print("Score SVM (decision_function):")
    print(f"Positivi: min={scores[test_labels==1].min():.3f} max={scores[test_labels==1].max():.3f} mean={scores[test_labels==1].mean():.3f}")
    print(f"Negativi: min={scores[test_labels==0].min():.3f} max={scores[test_labels==0].max():.3f} mean={scores[test_labels==0].mean():.3f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print("Classification Report:")
    print(classification_report(test_labels, y_pred, target_names=['Negativo (0)', 'Positivo (1)'], zero_division=0))

    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Neg (0)', 'Pos (1)'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"Confusion Matrix\nAccuracy={acc*100:.1f}%  F1={f1*100:.1f}%")
    plt.tight_layout()
    plt.show()

    # Distribuzione score per classe — utile per scegliere SVM_THRESHOLD
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(scores[test_labels == 0], bins=40, alpha=0.6, label='Negativi (0)', color='steelblue')
    ax.hist(scores[test_labels == 1], bins=40, alpha=0.6, label='Positivi (1)', color='tomato')
    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='soglia=0')
    ax.set_xlabel('SVM decision_function score')
    ax.set_ylabel('Conteggio patch')
    ax.set_title('Distribuzione score SVM per classe')
    ax.legend()
    plt.tight_layout()
    plt.show()

    #Plot curva ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')  # random baseline
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    #Plot PR_AUC
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
    return y_pred, scores

def main():

    with open(MODELS_DIR + MODEL_FILE, "rb") as f:
        data = d.load(f)

    loaded_tree = data["tree"]
    loaded_scaler = data["scaler"]
    loaded_svm = data["lsvm"]

    # Carica i percentili se presenti (modelli aggiornati con train_gp.py nuovo)
    # Per modelli vecchi feat_p1/feat_p99 saranno None → solo scaler.transform()
    feat_p1  = data.get("feat_p1",  None)
    feat_p99 = data.get("feat_p99", None)

    if feat_p1 is None:
        print("ATTENZIONE:\nmodello senza feat_p1/feat_p99, riaddestra con il main_training.py aggiornato per avere la pipeline completa clip+scaler.")
    else:
        print(f"feat_p1  range: [{feat_p1.min():.4f}, {feat_p1.max():.4f}]")
        print(f"feat_p99 range: [{feat_p99.min():.4f}, {feat_p99.max():.4f}]")

    print("Albero GP caricato:", loaded_tree)
    best_func = toolbox.compile(expr=loaded_tree)

    print("Estrazione feature dal training set")
    _, train_norm = build_features(best_func, x_train, y_train, feat_p1, feat_p99, loaded_scaler)

    print("Estrazione feature dal test set")
    _, test_norm = build_features(best_func, x_test, y_test, feat_p1, feat_p99, loaded_scaler)

    print(f"Train features shape: {train_norm.shape}")
    print(f"Test  features shape: {test_norm.shape}")

    y_pred_all, svm_scores = evaluate_test_set(loaded_svm, test_norm, y_test)

    if DEMO_FLAG:
        idx = 4 #cambiare indice per analizzare campioni diversi

        image      = x_data_demo[idx]
        true_label = y_data_demo[idx]

        feat_raw   = np.asarray(best_func(image)).reshape(1, -1)
        feat_norm  = normalize_features(feat_raw, feat_p1, feat_p99, loaded_scaler)
        pred_label = loaded_svm.predict(feat_norm)[0]
        svm_score  = loaded_svm.decision_function(feat_norm)[0]

        print(f"Campione idx={idx}")
        print(f"Label reale: {int(true_label)}")
        print(f"Label predetta: {int(pred_label)} ({'CORRETTA' if pred_label == true_label else 'ERRATA'})")
        print(f"SVM score: {svm_score:.4f}")
        print(f"Image shape: {image.shape}")
        print(f"Feature shape: {feat_raw.shape}")

        print("Calcolo saliency map...")
        saliency = compute_saliency(best_func, image, feat_p1, feat_p99, loaded_scaler, loaded_svm)

        plot_heatmap(image, saliency,pred_label=pred_label, true_label=true_label, title=f"Saliency Map — campione {idx}")

        plot_with_boxes(image, saliency, pred_label=pred_label, true_label=true_label)

if __name__ == "__main__":
    main()