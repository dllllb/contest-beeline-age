import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing.imputation import Imputer

import ds_tools.dstools.ml.transformers as tr
import ds_tools.dstools.ml.xgboost_tools as xgb

import json
import os.path
import time


def update_model_stats(stats_file, params, results):
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = []
        
    stats.append({**results, **params})
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

        
def run_experiment(evaluator, params, stats_file):
    start = time.time()
    scores = evaluator(params)
    exec_time = time.time() - start
    update_model_stats(stats_file, params, {**scores, 'exec-time-sec': exec_time})


def cv_test(est, n_folds):
    df = pd.read_csv('train.csv.gz')

    features = df.drop('y', axis=1)
    target = df.y

    if isinstance(est, tuple):
        transf, estimator = est
        features_t = transf.fit_transform(features, target)
    else:
        estimator = est
        features_t = features

    scores = cross_val_score(
        estimator=estimator,
        X=features_t,
        y=target,
        cv=n_folds,
        verbose=1)

    return {'ROC-AUC-mean': scores.mean(), 'ROC-AUC-std': scores.std()}


def submission(est):
    df = pd.read_csv('train.csv.gz')

    features = df.drop('y', axis=1)
    target = df.y

    if isinstance(est, tuple):
        transf, estimator = est
        pl = make_pipeline(transf, estimator)
    else:
        pl = est

    model = pl.fit(features, target)

    df_test = pd.read_csv('test.csv.gz', index_col='ID')

    y_pred = model.predict(df_test)

    res_df = pd.DataFrame({'y': y_pred}, index=df_test.index)
    res_df.to_csv('results.csv', index_label='ID')
    
def hyperparam_objective(params):
    return validate(params)['ROC-AUC-mean']
    
def hyperopt():
    import hyperopt as hpo

    space = {
        'max_depth': hpo.hp.choice('max_depth', np.arange(5, 20, 1, dtype=int)),
        'min_child_weight': hpo.hp.choice('min_child_weight', np.arange(1, 20, 1, dtype=int)),
        'gamma': hpo.hp.quniform('gamma', 0, 10, 1),
        "objective": "multi:softprob",
        "num_class": 7,
        "eta": 0.001,
        "num_rounds": 10000,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "scale_pos_weight": 1,
        "silent": 2,
        'validation-type': 'cv',
        'n_folds': 8,
        'category_encoding': 'onehot',
        'hkz_threshold': 49
    }

    best = hpo.fmin(hyperparam_objective, space, algo=hpo.tpe.suggest, max_evals=100)
    print(best)
    
    
def eval_accuracy(preds, dtrain):
    labels = dtrain.get_label()
    idx = np.argmax(preds, axis=1)
    res = accuracy_score(labels, idx)
    return 'accuracy', -res


def validate(params):
    category_encoding = params['category_encoding']
    
    if category_encoding == 'onehot':
        transf = make_pipeline(
            tr.high_cardinality_zeroing(params['hkz_threshold']),
            tr.df2dict(),
            DictVectorizer(sparse=False),
            Imputer(strategy='median'),
        )
    elif category_encoding == 'count':
        transf = make_pipeline(
            tr.count_encoder(),
            Imputer(strategy='median'),
        )
    elif category_encoding == 'target_share_hkz':
        transf = make_pipeline(
            tr.high_cardinality_zeroing(top=params['hkz_top']),
            tr.multi_class_target_share_encoder(size_threshold=1),
            Imputer(strategy='median'),
        )
    elif category_encoding == 'empytical_bayes':
        transf = make_pipeline(
            tr.empyrical_bayes_encoder(),
            Imputer(strategy='median'),
        )
    elif category_encoding == 'target_share':
        size_threshold=params['target_share_size_threshold']
        transf = make_pipeline(
            tr.multi_class_target_share_encoder(size_threshold=size_threshold),
            Imputer(strategy='median'),
        )
    
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 7,
        "eta": params['eta'],
        "num_rounds": 10000,
        "max_depth": params['max_depth'],
        "min_child_weight": params['min_child_weight'],
        "gamma": params['gamma'],
        "subsample": params['subsample'],
        "colsample_bytree": params['colsample_bytree'],
        "scale_pos_weight": 1,
        "silent": 0,
        "verbose": 10,
        "eval_func": eval_accuracy,
    }
    
    est = make_pipeline(transf, xgb.XGBoostClassifier(**xgb_params))
    return cv_test(est, n_folds=params['n_folds'])
