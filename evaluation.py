# Created by Dieter Fish at 2021/5/27
import datetime
import logging
from glob import glob
import numpy as np
from pathlib import Path
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
try:
    from sktime.forecasting.model_selection import temporal_train_test_split
except:
    from pmdarima.model_selection import train_test_split as temporal_train_test_split

from data_helper import get_feature_dict, prepare_model_data, prepare_model_data_basline, gen_data_matrix, load_rb
from model import auto_tuned_model, baseline_model, grid_search_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%m/%d/ %H:%M:%S %p")


def evalute_model(y, 
                  auto_tune=False, 
                  interval_len=180, 
                  gap=10,
                  loss_func=None):
    if loss_func is None:
        loss_func = "RMSE"
    timeline = y.index
    feature_files = glob(str(Path('data', '*.csv')))
    feature_files.remove(str(Path('data', 'rb_continuous_contracts.csv')))

    y_name = y.columns[0]
    record = []
    fitted_model = []
    predictions = []

    for i in range(0, len(timeline), interval_len + gap):
        cur_interal = timeline[i: i + interval_len + gap][:-gap]
        if len(cur_interal) < interval_len:
            logging.info("The sample data has been exhausted")
            continue
        target = y.loc[cur_interal]
        start = cur_interal[0]
        start_ = cur_interal[0] - datetime.timedelta(days=10)
        end = cur_interal[-1]
        logging.info("Current interval {0} : {1}".format(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))

        feature_dict, cat_features, numerical_features = get_feature_dict(feature_files, start_, end)
        df = gen_data_matrix(target, feature_dict)
        df = df.dropna(how='all', axis=1)
        df = df.loc[start: end]
        df_train, df_test = temporal_train_test_split(df, test_size=0.2)

        df_train_tree, df_val_tree = temporal_train_test_split(df, test_size=0.2)
        X_test_tree, y_test_tree = prepare_model_data(df_test, y_name)
        X_train_tree, y_train_tree = prepare_model_data(df_train_tree, y_name)
        X_val_tree, y_val_tree = prepare_model_data(df_val_tree, y_name)
        if auto_tune:
            tree_model = auto_tuned_model(X_train_tree,
                                          y_train_tree,
                                          X_val_tree,
                                          y_val_tree,
                                          loss_func
                                          )
        else:
            tree_model = grid_search_model(X_train_tree,
                                           y_train_tree,
                                           X_val_tree,
                                           y_val_tree,
                                           loss_func=loss_func
                                           )
        y_hat_tree = tree_model.predict(X_test_tree)
        if loss_func == 'RMSE':
            rmse_tree = mean_squared_error(y_test_tree, y_hat_tree)
        if loss_func == 'MAE':
            rmse_tree = mean_absolute_error(y_test_tree, y_hat_tree)

        r2_tree = r2_score(y_test_tree, tree_model.predict(X_test_tree))

        X_b, y_b = prepare_model_data_basline(df, y_name)
        y_train_b, y_test_b, X_train_b, X_test_b = temporal_train_test_split(y_b, X_b, test_size=0.2)
        baseline = baseline_model(X_train_b, y_train_b)
        y_hat_baseline = baseline.predict(X_test_b)

        if loss_func == 'RMSE':
            rmse_baseline = mean_squared_error(y_test_b, y_hat_baseline)
            rmse_naive = mean_squared_error(df_test[y_name], [df_train[y_name].mean()] * len(df_test))
        if loss_func == 'MAE':
            rmse_baseline = mean_absolute_error(y_test_b, y_hat_baseline)
            rmse_naive = mean_absolute_error(df_test[y_name], [df_train[y_name].mean()] * len(df_test))
        r2_baseline = r2_score(y_test_b, y_hat_baseline)
        r2_naive = r2_score(df_test[y_name], [df_train[y_name].mean()] * len(df_test))

        y_hat = {
            "tree_model": y_hat_tree,
            "baseline": y_hat_baseline,
        }
        predictions.append(y_hat)

        metrics = {loss_func + '_tree': rmse_tree,
                 loss_func + '_baseline': rmse_baseline,
                 loss_func + '_naive': rmse_naive,

                 'r2_tree': r2_tree,
                 'r2_baseline': r2_baseline,
                 'r2_naive': r2_naive,

                 'train_start': df_train.index[0],
                 'train_end': df_train.index[-1],
                 'test_start': df_test.index[0],
                 'test_end': df_test.index[-1],
                 }

        log_string = "\n\t" + loss_func + "_tree: {0}, \n\t" + loss_func + "_baseline: {1},\n\t" + loss_func + \
                     "_naive: {2},\n\t" + "r2_tree: {3},\n\t" + "r2_baseline: {4}, \n\t" +  "r2_naive: {5}"
        logging.info(log_string.format(rmse_tree,rmse_baseline,rmse_naive,r2_tree,r2_baseline,r2_naive))

        model = {"tree": tree_model, "baseline": baseline}
        record.append(metrics)
        fitted_model.append(model)
    return record, fitted_model, predictions



def backtest(prediction_df, record_df, price, f):
    """

    :param prediction_df:
    :param record_df:
    :param price:
    :param f:
    :return:
    """
    pnls = []
    rets = []
    for i in range(len(prediction_df)):
        prediction = prediction_df.iloc[i]
        s, e = record_df[['test_start', 'test_end']].iloc[i].dt
        price_ = price.loc[s:e]
        pnls.append(_backtest(prediction, price_, f))
        rets.append(_backtest2(prediction, price_, f))
    pnls = np.hstack(pnls)
    rets = np.hstack(rets)
    glr, win_rate = cal_glr_wr(pnls)
    sr = cal_sharpe_ratio(rets)

    print("Gain-Loss Ratio: {:.2f} ".format(glr))
    print("Winning Rate: {:.2f}% ".format(win_rate))
    print("Sharpe Ratio: {:.2f} ".format(sr))
    return glr, win_rate, sr



def _backtest2(prediction, price, acct_num,):
    """
    Cal daiy return in % form
    :param prediction:
    :param price:
    :param acct_num:
    :return:
    """
    # starting net val for trading account
    mat = np.ones((acct_num, len(price)))
    # liquidate or build position time
    _idx = np.arange(len(price))
    # price change
    _chg = price.pct_change()

    for i in range(acct_num):
        adjust_time = _idx[i::acct_num]
        for j, k in zip(adjust_time, np.hstack((adjust_time[1:], [-1]))):
            sign = np.sign(prediction[j])
            if k != -1:
                mat[i][j+1:k+1] = 1+sign * _chg[j+1: k+1]
            else:
                mat[i][j+1:] = 1+ sign * _chg[j+1: ]
    mat = mat.cumprod(1).sum(0)
    mat /= mat[0]
    # daily return in % form.
    return 100 * np.diff(mat)/mat[:-1]


def _backtest(prediction, price, f):
    '''
    PnL for each trade
    :param prediction:
    :param start:
    :param end:
    :param price:
    :param f:
    :return:
    '''
    pos = np.where(prediction > 0, 1, -1)[:-f]
    chg = price.diff(f).dropna()
    pnl = chg * pos
    return pnl


def cal_glr_wr(pnl):
    glr = -pnl[pnl > 0].mean() / pnl[pnl < 0].mean()
    win_rate = len(pnl[pnl > 0]) / len(pnl)
    return glr, 100 * win_rate

def cal_sharpe_ratio(r):
    return np.sqrt(252) * r.mean() / r.std()



def evaluate_model2(y,
                    price,
                    auto_tune=False,
                    interval_len=180,
                    test_size=0.2,
                    loss_func=None):
    if loss_func is None:
        loss_func = "RMSE"

    timeline = y.index
    feature_files = glob(str(Path('data', '*.csv')))
    feature_files.remove(str(Path('data', 'rb_continuous_contracts.csv')))

    y_name = y.columns[0]
    logging.info("The target is {0}".format(y_name))
    record = []
    fitted_model = []
    predictions = []
    test_price = []

    start = 0
    end = start + interval_len

    while (end < len(timeline)):
        current_interval = timeline[start: end]
        logging.info("Current interval {0} : {1}".format(current_interval[0].strftime("%Y-%m-%d"),
                                                         current_interval[-1].strftime("%Y-%m-%d")))
        target = y.loc[current_interval]
        s, e = current_interval[0], current_interval[-1]
        feature_dict, cat_features, numerical_features = get_feature_dict(feature_files, s, e)
        df = gen_data_matrix(target, feature_dict)
        df = df.dropna(how='all', axis=1)
        df = df.loc[s: e]
        df_train, df_test = temporal_train_test_split(df, test_size=test_size)
        df_train_tree, df_val_tree = temporal_train_test_split(df, test_size=0.2)
        X_test_tree, y_test_tree = prepare_model_data(df_test, y_name)
        X_train_tree, y_train_tree = prepare_model_data(df_train_tree, y_name)
        X_val_tree, y_val_tree = prepare_model_data(df_val_tree, y_name)
        test_price.append(price.loc[df_test.index].values)
        if auto_tune:
            tree_model = auto_tuned_model(X_train_tree,
                                          y_train_tree,
                                          X_val_tree,
                                          y_val_tree,
                                          loss_func=loss_func
                                          )
        else:
            tree_model = grid_search_model(X_train_tree,
                                           y_train_tree,
                                           X_val_tree,
                                           y_val_tree,
                                           loss_func=loss_func
                                           )
        y_hat_tree = tree_model.predict(X_test_tree)
        if loss_func == 'RMSE':
            rmse_tree_test = mean_squared_error(y_test_tree, y_hat_tree)
            rmse_tree_train = mean_squared_error(y_train_tree, tree_model.predict(X_train_tree))
        if loss_func == 'MAE':
            rmse_tree_test = mean_absolute_error(y_test_tree, y_hat_tree)
            rmse_tree_train = mean_absolute_error(y_train_tree, tree_model.predict(X_train_tree))
        r2_tree_test = r2_score(y_test_tree, tree_model.predict(X_test_tree))
        r2_tree_train = r2_score(y_train_tree, tree_model.predict(X_train_tree))

        X_b, y_b = prepare_model_data_basline(df, y_name)
        y_train_b, y_test_b, X_train_b, X_test_b = temporal_train_test_split(y_b, X_b, test_size=test_size)
        baseline = baseline_model(X_train_b, y_train_b)
        y_hat_baseline = baseline.predict(X_test_b)

        if loss_func == 'RMSE':
            rmse_baseline_test = mean_squared_error(y_test_b, y_hat_baseline)
            rmse_baseline_train = mean_squared_error(y_train_b, baseline.predict(X_train_b))
            rmse_naive_test = mean_squared_error(df_test[y_name], [df_train[y_name].mean()] * len(df_test))
            rmse_naive_train = mean_squared_error(df_train[y_name], [df_train[y_name].mean()] * len(df_train))
        if loss_func == 'MAE':
            rmse_baseline_test = mean_absolute_error(y_test_b, y_hat_baseline)
            rmse_baseline_train = mean_absolute_error(y_train_b, baseline.predict(X_train_b))
            rmse_naive_test = mean_absolute_error(df_test[y_name], [df_train[y_name].mean()] * len(df_test))
            rmse_naive_train = mean_absolute_error(df_train[y_name], [df_train[y_name].mean()] * len(df_train))

        r2_baseline_test, r2_baseline_train = r2_score(y_test_b, y_hat_baseline), r2_score(y_train_b,
                                                                                           baseline.predict(X_train_b))
        r2_naive_test = r2_score(df_test[y_name], [df_train[y_name].mean()] * len(df_test))
        r2_naive_train = r2_score(df_train[y_name], [df_train[y_name].mean()] * len(df_train))

        y_hat = {
            "tree_model": y_hat_tree,
            "baseline": y_hat_baseline,
        }
        predictions.append(y_hat)

        metrics = {loss_func + '_tree' + '_test': rmse_tree_test,
                   loss_func + '_tree' + '_train': rmse_tree_train,
                   loss_func + '_baseline' + '_test': rmse_baseline_test,
                   loss_func + '_baseline' + '_train': rmse_baseline_train,
                   loss_func + '_naive' + '_test': rmse_naive_test,
                   loss_func + '_naive' + '_train': rmse_naive_train,
                   'r2_tree_test': r2_tree_test,
                   'r2_tree_train': r2_tree_train,
                   'r2_baseline_test': r2_baseline_test,
                   'r2_baseline_train': r2_baseline_train,
                   'r2_naive_test': r2_naive_test,
                   'r2_naive_train': r2_naive_train,
                   'train_start': df_train.index[0],
                   'train_end': df_train.index[-1],
                   'test_start': df_test.index[0],
                   'test_end': df_test.index[-1],
                   }

        log_string = []
        for k, v in metrics.items():
            if k not in ['train_start', 'train_end', 'test_start', 'test_end']:
                sub_str = '='.join((k, str(v)))
                log_string.append(sub_str)
        log_string = '\n\t'.join(log_string)
        log_string = '\n\t' + log_string
        logging.info(log_string)

        model = {"tree": tree_model, "baseline": baseline}
        record.append(metrics)
        fitted_model.append(model)
        end = end + len(df_test)
        start = end - interval_len
    else:
        logging.info("The sample data has been exhausted")

    try:
    # Backtest Statistics
        test_price = np.array(test_price).flatten()
        predictions_df = pd.DataFrame(predictions)
        predictions_tree = np.concatenate(predictions_df["tree_model"].tolist())
        predictions_baseline = np.concatenate(predictions_df["baseline"].tolist())
        try:
            assert len(test_price) == len(predictions_tree)
            assert len(test_price) == len(predictions_baseline)
        except AssertionError:
            logging.critical("Check the Prediction Array")
            logging.info("Lenght of test price {0}".format(str(len(price))))
            logging.info("Lenght of predictions {0}".format(str(len(predictions_tree))))
        acct_num = int(y_name.split('_')[-1])
        tree_result = backtest3(predictions_tree, test_price, acct_num)
        baseline_result = backtest3(predictions_baseline, test_price, acct_num)
        backtest_result = {'tree': tree_result, 'baseline': baseline_result}
    except:
        backtest_result = None

    return record, fitted_model, predictions, backtest_result


def backtest3(prediction, price, acct_num):
    """

    :param prediction_df:
    :param record_df:
    :param price:
    :param f:
    :return:
    """
    position = np.where(prediction > 0, 1, -1)[:-acct_num]
    chg = price[acct_num:] - price[:-acct_num]
    pnl = chg * position
    glr = -pnl[pnl > 0].mean() / pnl[pnl < 0].mean()
    win_rate = 100 * len(pnl[pnl > 0]) / len(pnl)
    daily_ret = cal_daily_return(prediction, price, acct_num)
    sharpe_ratio = np.sqrt(252) * daily_ret.mean() / daily_ret.std()
    annualized_return = ((1 + daily_ret.mean() / 100) ** 252 - 1) * 100
    max_down = max_draw_down(daily_ret)
    items = {'gain_loss_ratio': glr,
             'winning_rate': win_rate,
             'sharpe_ratio': sharpe_ratio,
             'annualized_return': annualized_return,
             'max_drawdown': max_down,
             'daily_ret': daily_ret}
    return items


def cal_annualized_return(mean_ret):
    return ((1 + mean_ret) ** 252 - 1) * 100

def max_draw_down(ret: np.array):
    nv = (1+ret/100).cumprod()
    nv = np.insert(nv, 0, 1)
    return -100 * (1 - (nv/np.maximum.accumulate(nv))).max()


def cal_daily_return(prediction, price, acct_num):
    # starting net val for trading account
    mat = np.ones((acct_num, len(price)))
    # liquidate or build position time
    _idx = np.arange(len(price))
    # price change
    _chg = pd.Series(price).pct_change()

    for i in range(acct_num):
        adjust_time = _idx[i::acct_num]
        for j, k in zip(adjust_time, np.hstack((adjust_time[1:], [-1]))):
            sign = np.sign(prediction[j])
            if k != -1:
                mat[i][j + 1:k + 1] = 1 + sign * _chg[j + 1: k + 1]
            else:
                mat[i][j + 1:] = 1 + sign * _chg[j + 1:]
    mat = mat.cumprod(1).sum(0)
    mat /= mat[0]
    # daily return in % form.
    return 100 * np.diff(mat) / mat[:-1]


if __name__ == '__main__':
    price_df = load_rb()
    price = price_df.loc["2014-01-01":].close_adj.to_frame()
    price['pt_ret'] = 100 * price['close_adj'].pct_change().shift(-1)
    price['pt_ret_5'] = 100 * price['close_adj'].pct_change(5).shift(-5)
    price['pt_ret_14'] = 100 * price['close_adj'].pct_change(14).shift(-14)
    price['pt_ret_20'] = 100 * price['close_adj'].pct_change(20).shift(-20)
    price['pt_ret_30'] = 100 * price['close_adj'].pct_change(30).shift(-30)

    # s, e = "2015-06-29", "2016-04-21"
    # feature_files = glob(str(Path('data', '*.csv')))
    # feature_files.remove(str(Path('data', 'rb_continuous_contracts.csv')))
    # feature_dict, cat_features, numerical_features = get_feature_dict(feature_files, s, e)
    # target = price['pt_ret_5'].loc[s: e].to_frame()
    # df = gen_data_matrix(target, feature_dict)
    # df = df.dropna(how='all', axis=1)
    # df = df.loc[s: e]
    # df_train, df_test = temporal_train_test_split(df, test_size=0.2)
    # df_train_tree, df_val_tree = temporal_train_test_split(df, test_size=0.2)
    # X_test_tree, y_test_tree = prepare_model_data(df_test, 'pt_ret_5')
    # X_train_tree, y_train_tree = prepare_model_data(df_train_tree, 'pt_ret_5')
    # X_val_tree, y_val_tree = prepare_model_data(df_val_tree, 'pt_ret_5')
    recordss2, fitted_modelss2, predictionss2, backtest_resultss2 = evaluate_model2(
        price['pt_ret_5'].loc["2020-07-23":].to_frame(), price.close_adj.loc["2020-07-23":].to_frame(),
        interval_len=200, auto_tune=True, loss_func='MAE', test_size=0.3)