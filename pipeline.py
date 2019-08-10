import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from encoders import high_cardinality_zeroing, df2dict, count_encoder


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
    
    params = init_params(params)
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
        scoring='accuracy',
        verbose=0)

    return {'accuracy-mean': scores.mean(), 'accuracy-std': scores.std()}


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

    space = init_params({
        'max_depth': hpo.hp.quniform('max_depth', 5, 20, 1),
        'min_child_weight': hpo.hp.quniform('min_child_weight', 1, 20, 1),
        'gamma': hpo.hp.quniform('gamma', 0, 10, 1),
        "eta": 0.001,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        'validation-type': 'cv',
        'category_encoding': 'onehot',
        'hkz_threshold': 49
    })

    best = hpo.fmin(hyperparam_objective, space, algo=hpo.tpe.suggest, max_evals=100)
    print(best)
    
    
def eval_accuracy(preds, dtrain):
    labels = dtrain.get_label()
    idx = np.argmax(preds, axis=1)
    res = accuracy_score(labels, idx)
    return 'accuracy', -res


def init_xgb_est(params):
    keys = {
        'eta',
        'n_estimators',
        'max_depth',
        'min_child_weight',
        'gamma',
        'subsample',
        'colsample_bytree'
    }
    
    xgb_params = {
        "objective": "multi:softprob",
        "scale_pos_weight": 1,
        **{k: v for k, v in params.items() if k in keys},
    }

    class XGBC(XGBClassifier):
        def fit(self, x, y, **kwargs):
            f_train, f_val, t_train, t_val = train_test_split(x, y, test_size=.05)
            super().fit(
                f_train,
                t_train,
                eval_set=[(f_val, t_val)],
                eval_metric=eval_accuracy,
                early_stopping_rounds=50,
                verbose=10)
    
    return XGBC(**xgb_params)


def validate(params):    
    category_encoding = params['category_encoding']
    
    if category_encoding == 'onehot':
        transf = make_pipeline(
            high_cardinality_zeroing(params['hkz_threshold']),
            df2dict(),
            DictVectorizer(sparse=False),
            SimpleImputer(strategy='median'),
        )
    elif category_encoding == 'count':
        transf = make_pipeline(
            count_encoder(),
            SimpleImputer(strategy='median'),
        )
    else:
        raise AttributeError(f'unknown cetegory endcoding method: {category_encoding}')
    
    est = make_pipeline(transf, init_xgb_est(params))
    return cv_test(est, n_folds=params['n_folds'])


def test_validate():
    params = {
        "eta": 0.1,
        "max_depth": 9,
        "min_child_weight": 6,
        "gamma": 0,
        "subsample": 0.1,
        "colsample_bytree": 0.2,
        "category_encoding": "count",
        'num_rounds': 10,
        'n_folds': 3,
    }
    print(validate(init_params(params)))
