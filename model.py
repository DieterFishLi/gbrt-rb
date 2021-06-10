# Created by Dieter Fish at 2021/5/26
from sklearn.linear_model import ElasticNet

from hypertuning import *


def auto_tuned_model(X,
                     y,
                     X_val,
                     y_val,
                     loss_func,
                     n_splits=2,
                     max_train_group_size=130,
                     max_test_group_size=30,
                     group_gap = 3,
                     ):
    """
    return the model with optimised hyperparameter
    :return:
    """
    cv = PurgedGroupTimeSeriesSplit(n_splits=n_splits,
                                    max_train_group_size=max_train_group_size,
                                    max_test_group_size=max_test_group_size,
                                    group_gap=group_gap)
                                    # max_val_group_size=max_val_group_size,
                                    # val_group_gap=val_group_gap,
                                    # test_group_gap=test_group_gap, )
    study = optimize(X, y, cv, loss_func)
    model = CatBoostRegressor(**study.best_params, loss_function=loss_func, has_time=True, task_type="GPU")
    cat_features = list(X.columns.difference(X._get_numeric_data().columns))
    model.fit(X, y, cat_features=cat_features, eval_set=(X_val, y_val),verbose=False, plot=True)
    return model


def grid_search_model(X, y, X_val, y_val, loss_func=None):
    grid = { 'learning_rate': [0.03, 0.3, 0.6],
            'depth': [2, 4, 6, 8],
            'iterations': [100, 150, 200, 500],
             'l2_leaf_reg': [0.2, 0.5, 1],
    }

    if loss_func is None:
        loss_func = "RMSE"
    model = CatBoostRegressor(loss_function=loss_func,
                              has_time=True,
                              # iterations=100,
                              logging_level="Silent",

                              # boosting_type="No",
                              subsample=1,
                              # task_type="GPU"
                              )

    # X = np.row_stack((X, X_val))
    # y = np.row_stack((y, y_val))
    data = Pool(X, y, cat_features=list(X.columns.difference(X._get_numeric_data().columns)))
    model.grid_search(grid, data, shuffle=False, verbose=False, plot=False, )
    model_2 = CatBoostRegressor(**model.get_params())
    model_2.fit(data, eval_set=(X_val, y_val),)
    return model_2


def baseline_model(X, y):
    """
    The baseline model is ElasticNet, which has better generalization capability than OLS Regression
    :param X:
    :param y:
    :return:
    """
    # cat_features = list(X.columns.difference(X._get_numeric_data().columns))
    # mat1 = X[X.columns.difference(cat_features)].values
    # mat2 = X[cat_features]
    # enc = OneHotEncoder(handle_unknown='ignore')
    # cat_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    # num_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    # pt = PowerTransformer()
    # mt = MinMaxScaler()
    # mat1 = pt.fit_transform(mt.fit_transform(num_imputer.fit_transform(mat1)))
    # mat2 = cat_imputer.fit_transform(mat2)
    # mat2 = enc.fit_transform(mat2).toarray()
    # mat = np.hstack((mat1, mat2))
    model = ElasticNet()
    model.fit(X, y)
    return model
