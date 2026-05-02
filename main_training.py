import random
import time
import numpy
import warnings
from datetime import datetime

import evalGP_fgp as evalGP
import gp_restrict as gp_restrict
from deap import base, creator, tools, gp
from strongGPDataType import Int1, Int2, Int3, Float1, Float2, Float3, Img, Img1, Vector
import fgp_functions as fe_fs
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import dill as pkl

warnings.filterwarnings("ignore")

randomSeeds  = 12
dataSetName  = 'sampled_linfonodi'
directory    = './preprocessed_dataset/'
models_dir   = './models/'

population      = 200 #80
generation      = 30 #10
cxProb          = 0.8
mutProb         = 0.15
elitismProb     = 0.05
totalRuns       = 10 #5
initialMinDepth = 2
initialMaxDepth = 6 #5
maxDepth        = 10 #8

#Percentili per clip robusto delle feature 
#Valori < P_LOW o > P_HIGH vengono clippati prima del MinMaxScaler.
#Questo rende lo scaler robusto a patch con feature fuori range.
P_LOW  = 1    # percentile inferiore
P_HIGH = 99   # percentile superiore

#CARICAMENTO DATI
#NOTA: i .npy sono già in [0,1] dopo window_and_normalize nel preprocessing.
#non dividere ulteriormente per 255
x_train = numpy.load(directory + dataSetName + '_train_data.npy')
y_train = numpy.load(directory + dataSetName + '_train_label.npy')
x_test  = numpy.load(directory + dataSetName + '_test_data.npy')
y_test  = numpy.load(directory + dataSetName + '_test_label.npy')

print(f"Training data shape: {x_train.shape}")
print(f"Training label shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test label shape: {y_test.shape}")
print(f"Train range: [{x_train.min():.4f}, {x_train.max():.4f}]")
print(f"Test  range: [{x_test.min():.4f},  {x_test.max():.4f}]\n")

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

def evalTrain(individual):
    try:
        func = toolbox.compile(expr=individual)
        train_tf = []
        for i in range(0, len(y_train)):
            train_tf.append(numpy.asarray(func(x_train[i, :, :])))
        train_tf = numpy.asarray(train_tf, dtype=float)
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(train_tf)
        lsvm = LinearSVC()
        accuracy = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=5).mean(), 2)
    except:
        accuracy = 0
    return accuracy,

toolbox = base.Toolbox()
toolbox.register("evaluate", evalTrain)
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

def clip_features(feat_matrix, p1, p99):
    return numpy.clip(feat_matrix, p1, p99)

def evalTrain(individual):
    try:
        func = toolbox.compile(expr=individual)

        train_tf = []
        for i in range(len(y_train)):
            train_tf.append(numpy.asarray(func(x_train[i, :, :])))
        train_tf = numpy.asarray(train_tf, dtype=float)

        p1  = numpy.percentile(train_tf, P_LOW,  axis=0)
        p99 = numpy.percentile(train_tf, P_HIGH, axis=0)
        train_tf_clipped = clip_features(train_tf, p1, p99)

        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(train_tf_clipped)

        lsvm = LinearSVC()
        accuracy = round(
            100 * cross_val_score(lsvm, train_norm, y_train, cv=5).mean(), 2)
    except Exception:
        accuracy = 0
    return accuracy,

def GPMain(randomSeeds):
    random.seed(randomSeeds)
    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()

    stats_fit       = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation, stats=mstats, halloffame=hof, verbose=True)
    return pop, log, hof

def evalTest(toolbox, individual, trainData, trainLabel, testData, testLabel):
    func = toolbox.compile(expr=individual)

    train_tf, test_tf = [], []
    for i in range(len(trainLabel)):
        train_tf.append(numpy.asarray(func(trainData[i, :, :])))
    for j in range(len(testLabel)):
        test_tf.append(numpy.asarray(func(testData[j, :, :])))

    train_tf = numpy.asarray(train_tf, dtype=float)
    test_tf  = numpy.asarray(test_tf,  dtype=float)

    p1  = numpy.percentile(train_tf, P_LOW,  axis=0)
    p99 = numpy.percentile(train_tf, P_HIGH, axis=0)

    train_tf_clipped = clip_features(train_tf, p1, p99)
    test_tf_clipped  = clip_features(test_tf,  p1, p99)   # usa p1/p99 del TRAIN

    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf_clipped)
    test_norm  = min_max_scaler.transform(test_tf_clipped)

    lsvm = LinearSVC()
    lsvm.fit(train_norm, trainLabel)
    accuracy = round(100 * lsvm.score(test_norm, testLabel), 2)

    return (numpy.asarray(train_tf), numpy.asarray(test_tf), trainLabel, testLabel, accuracy, lsvm, min_max_scaler, p1, p99)

def main():
    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    (train_tf, test_tf, trainLabel, testL,
     testResults, lsvm, min_max_scaler, p1, p99) = evalTest(
        toolbox, hof[0], x_train, y_train, x_test, y_test)

    testTime = time.process_time() - endTime

    print(f"\nTest accuracy : {testResults}%")
    print(f"Train time    : {trainTime:.1f}s")
    print(f"Test time     : {testTime:.1f}s")
    print(f"Feature shape : train={train_tf.shape}  test={test_tf.shape}")
    print(f"Albero GP     : {hof[0]}")
    print(f"feat_p1  range: [{p1.min():.4f}, {p1.max():.4f}]")
    print(f"feat_p99 range: [{p99.min():.4f}, {p99.max():.4f}]")

    model_path = models_dir + f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
    with open(model_path, "wb") as f:
        pkl.dump({
            "tree":      hof[0],
            "scaler":    min_max_scaler,
            "lsvm":      lsvm,
            "feat_p1":   p1,
            "feat_p99":  p99,   
        }, f)
    print(f"Modello salvato in: {model_path}")

if __name__ == "__main__":
    main()