# Created by Dieter Fish at 2021/5/26
import numpy as np
import optuna
from catboost import CatBoostRegressor, Pool
from catboost.utils import eval_metric
from optuna.samplers import TPESampler

from data_helper import PurgedGroupTimeSeriesSplit


def objective(trial, X, y, cv: PurgedGroupTimeSeriesSplit, loss_func):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.9),
         'iterations': trial.suggest_int('iterations', 100, 500, step=25),
        'depth': trial.suggest_int('depth', 3, 15),
        # 'grow_policy':trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise']),
        'od_type':trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
         'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
        'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 1, 10),

    }
    cat_features = list(X.columns.difference(X._get_numeric_data().columns))
    cat_features_idx = np.array([np.where(X.columns == i) for i in cat_features]).flatten()
    rmse = []
    g = X.reset_index()['index'].values
    X = X.values
    y = y.values
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=g)):
        train_x = X[train_idx, :]
        train_y = y[train_idx]
        val_x = X[val_idx, :]
        val_y = y[val_idx]
        train_pool = Pool(train_x, train_y, cat_features=cat_features_idx)
        val_pool = Pool(val_x, val_y, cat_features=cat_features_idx)
        model = CatBoostRegressor(**params, loss_function=loss_func, has_time=True, random_seed=42, task_type="GPU")
        model.fit(train_pool, verbose=0, eval_set=val_pool)
        yhat = model.predict(val_pool)
        rmse.append(eval_metric(val_pool.get_label(), yhat, loss_func))
    return np.average(rmse)


def optimize(X, y, cv, loss_func, n_trials=10, ):
    sampler = TPESampler(seed=1024)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    # obj = objective(trial, X, y, cv,)
    study.optimize(lambda trial: objective(trial, X, y, cv, loss_func), n_trials=n_trials)
    return study