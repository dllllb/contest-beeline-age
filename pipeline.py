import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing.imputation import Imputer

import ds_tools.dstools.ml.transformers as tr
import ds_tools.dstools.ml.xgboost_tools as xgb


def update_model_stats(stats_file, params, results):
    import json
    import os.path
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = []
        
    stats.append({**results, **params})
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

        
def run_experiment(evaluator, params, stats_file):    
    import time
    
    start = time.time()
    scores = evaluator(params)
    exec_time = time.time() - start
    update_model_stats(stats_file, params, {**scores, 'exec-time-sec': exec_time})

    
def init_params(overrides):
    defaults = {
        'validation-type': 'cv',
        'n_folds': 8,
        'num_rounds': 10000,
    }
    return {**defaults, **overrides}


def cv_test(est, n_folds, n_rows=None):
    df = pd.read_csv('train.csv.gz', nrows=n_rows)

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
    params['max_depth'] = int(params['max_depth'])
    return validate(params)['ROC-AUC-mean']
    
def hyperopt():
    import hyperopt as hpo

    space = {
        'max_depth': hpo.hp.quniform('max_depth', 5, 20, 1),
        'min_child_weight': hpo.hp.quniform('min_child_weight', 1, 20, 1),
        'gamma': hpo.hp.quniform('gamma', 0, 10, 1),
        "eta": 0.001,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
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
    params = init_params(params)
    
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
            tr.empirical_bayes_encoder(),
            Imputer(strategy='median'),
        )
    elif category_encoding == 'target_share':
        size_threshold=params['target_share_size_threshold']
        transf = make_pipeline(
            tr.multi_class_target_share_encoder(size_threshold=size_threshold),
            Imputer(strategy='median'),
        )
    
    keys = {
        'eta',
        'num_rounds',
        'max_depth',
        'min_child_weight',
        'gamma',
        'subsample',
        'colsample_bytree'
    }
    
    xgb_params = {k: v for k, v in params.items() if k in keys}
    
    xgb_params_all = {
        "objective": "multi:softprob",
        "num_class": 7,
        "scale_pos_weight": 1,
        "silent": 0,
        "verbose": 10,
        "eval_func": eval_accuracy,
        **xgb_params,
    }
    
    est = make_pipeline(transf, xgb.XGBoostClassifier(**xgb_params_all))
    return cv_test(est, n_folds=params['n_folds'], n_rows=params.get('n_rows'))


def test_validate():
    params = {
        "eta": 0.1,
        "max_depth": 9,
        "min_child_weight": 6,
        "gamma": 0,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "category_encoding": "empytical_bayes",
        'num_rounds': 10,
        'n_fodls': 3
    }
    print(validate(params))