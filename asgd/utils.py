import numpy as np


#########
##stats##
#########

def multiclass_stats(actual, predicted, labels):
    accuracy = float(100*(predicted == actual).sum() / float(len(predicted)))
    aps = []
    aucs = []
    if len(labels) == 2:
        labels = labels[1:]
    for label in labels:
        prec, rec = precision_and_recall(actual, predicted, label)
        ap = ap_from_prec_and_rec(prec, rec)
        aps.append(ap)
        auc = auc_from_prec_and_rec(prec, rec)
        aucs.append(auc)
    ap = np.array(aps).mean()
    auc = np.array(aucs).mean()
    return accuracy, ap, auc
    

def average_precision(actual, predicted, labels):
    if len(labels) == 2:
        labels = labels[1:]
    aps = []
    for label in labels:
        prec, rec = precision_and_recall(actual, predicted, label)
        ap = ap_from_prec_and_rec(prec, rec)
        aps.append(ap)
    ap = np.array(aps).mean()
    return ap
    

def ap_from_prec_and_rec(prec, rec):
    ap = 0
    rng = np.arange(0, 1.1, .1)
    for th in rng:
        parray = prec[rec>=th]
        if len(parray) == 0:
            p = 0
        else:
            p = parray.max()
        ap += p / rng.size
    return ap


def area_under_curve(actual, predicted, labels):
    if len(labels) == 2:
        labels = labels[1:]
    aucs = []
    for label in labels:
        prec, rec = precision_and_recall(actual, predicted, label)
        auc = auc_from_prec_and_rec(prec, rec)
        aucs.append(auc)
    auc = np.array(aucs).mean()
    return auc
    

def auc_from_prec_and_rec(prec, rec):
    #area under curve
    h = np.diff(rec)
    auc = np.sum(h * (prec[1:] + prec[:-1])) / 2.0
    return auc


def rsquared(actual, predicted):
    a_mean = actual.mean()
    num = np.linalg.norm(actual - predicted) ** 2
    denom = np.linalg.norm(actual - a_mean) ** 2
    return 1 -  num / denom


def precision_and_recall(actual, predicted, cls):
    c = (actual == cls)
    si = np.argsort(-c)
    tp = np.cumsum(np.single(predicted[si] == cls))
    fp = np.cumsum(np.single(predicted[si] != cls))
    rec = tp /np.sum(predicted == cls)
    prec = tp / (fp + tp)
    return prec, rec
    

#########
##utils##
#########

def normalize(feats, trace_normalize=False, data=None):
    """Performs normalizations before training on a list of feature array/label
    pairs. first feature array in list is taken by default to be training set
    and norms are computed relative to that one.
    """

    if data is None:
        train_f = feats[0]
        m = train_f.mean(axis=0)
        s = np.maximum(train_f.std(axis=0), 1e-8)
    else:
        m = data['train_mean']
        s = data['train_std']
    feats = [(f - m) / s for f in feats]
    if trace_normalize:
        if data is None:
            train_f = feats[0]
            tr = np.maximum(np.sqrt((train_f**2).sum(axis=1)).mean(), 1e-8)
        else:
            tr = data['trace']
    else:
        tr = None
    if trace_normalize:
        feats = [f / tr for f in feats]
    feats = tuple(feats)
    return feats + (m, s, tr)


def mean_and_std(X, min_std):
    # XXX: this loop is more memory efficient than numpy but not as
    # numerically accurate. It would be better to look at the train_mean,
    # and then either use the msq for getting unit norms if the train_means
    # are small-ish, or else use numpy.std if the mean is large enough to
    # cause numerical trouble
    m = np.zeros(X.shape[1], dtype='float64')
    msq = np.zeros(X.shape[1], dtype='float64')
    for i in xrange(X.shape[0]):
        alpha = 1.0 / (i + 1)
        v = X[i]
        m = (alpha * v) + (1 - alpha) * m
        msq = (alpha * v * v) + (1 - alpha) * msq

    train_mean = np.asarray(m, dtype=X.dtype)
    train_std = np.sqrt(np.maximum(
            msq - m * m,
            min_std ** 2)).astype(X.dtype)
    return train_mean, train_std


def split_center_normalize(X, y,
        validset_fraction=.2,
        validset_max_examples=5000,
        inplace=False,
        min_std=1e-4,
        batchsize=1):
    n_valid = int(min(
        validset_max_examples,
        validset_fraction * X.shape[0]))

    # -- increase n_valid to a multiple of batchsize
    while n_valid % batchsize:
        n_valid += 1

    n_train = X.shape[0] - n_valid

    # -- decrease n_train to a multiple of batchsize
    while n_train % batchsize:
        n_train -= 1

    if not inplace:
        X = X.copy()

    train_features = X[:n_train]
    valid_features = X[n_train:n_train + n_valid]
    train_labels = y[:n_train]
    valid_labels = y[n_train:n_train + n_valid]

    train_mean, train_std = mean_and_std(X, min_std=min_std)

    # train features and valid features are aliased to X
    X -= train_mean
    X /= train_std

    return ((train_features, train_labels),
            (valid_features, valid_labels),
            train_mean,
            train_std)


def simple_bracket_min(f, pt0, pt1):
    v0 = f(pt0)
    v1 = f(pt1)
    if v0 > v1:
        while v0 > v1:
            raise NotImplementedError()

