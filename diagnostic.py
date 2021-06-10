# Created by Dieter Fish at 2021/5/18
import pandas as pd
import ppscore as pps
import statsmodels.api as sm
from catboost import CatBoostRegressor
from catboost.utils import eval_metric
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.tsa.stattools import kpss, adfuller
from arch.unitroot import VarianceRatio

def stationary_test(ts):
    """
    时间序列的平稳性检验
    :param ts:
    :return:
    """
    ts = ts.dropna()
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    result = dict()
    result['adf'] = dfoutput
    print('Results of KPSS Test:')
    kpsstest = kpss(ts, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)
    result['kpss'] = kpss_output
    return result


def autocorrelation_test(ts):
    """
    序列样本的自相关性检验
    :param ts:
    :return:
    """
    print('Results of Ljung-Box Test')
    dftest = sm.stats.acorr_ljungbox(ts, return_df=True)
    print(dftest)
    return dftest



def randonness_test(ts, lag=None):
    """
    样本的随机性检验；
    :param ts:
    :return:
    """
    # The Run Test
    """
    H0:	the sequence was produced in a random manner
    """
    ts = ts.dropna()
    statistic, pval = runstest_1samp(ts, correction=False)
    print("     The Run Test Result  \n"
          "=====================================\n"
          "Statistic = {0}\n"
          "P-Value = {1}\n"
          "-------------------------------------\n".format(statistic, pval))

    # Variance Ratio Test
    """
    H0: The series is ~ Random Walk
    """
    if lag is None:
        lag = 2
    vr = VarianceRatio(ts, lags=lag)
    print(vr.summary().as_text())
    return (statistic, pval, )


def predict_power_score(df, x:str, y:str):
    """
    PPS 指标；针对树模型特征选择的指标
    :param df:
    :param x:
    :param y:
    :return:
    """
    return pps.score(df, x, y)


def calc_test_quality(train_pool, val_pool, test_pool, **kwargs):
    '''
    模型预测性能评价

    :param train_pool:
    :param val_pool:
    :param test_pool:
    :param kwargs:
    :return:
    '''
    model = CatBoostRegressor(**kwargs, random_seed=42)
    model.fit(train_pool, verbose=0, eval_set=val_pool)
    y_pred = model.predict(test_pool)
    return eval_metric(test_pool.get_label(), y_pred, 'RMSE')



