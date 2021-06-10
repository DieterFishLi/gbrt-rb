import datetime
import datetime
import logging
import random
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import MonthLocator, DateFormatter
from numpy import log, polyfit, sqrt, std, subtract
from pandas.plotting import lag_plot
from scipy.stats import probplot, moment
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S %p")

# plt.style.use('seaborn-whitegrid')
# 加这个两句 可以显示中文
# plt.rcParams['font.sans-serif'] = [u'SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# 默认图像大小
# plt.rcParams["figure.figsize"] = [16, 9]
# 字体大小
# plt.rcParams["font.size"] = 18

# Generate some random number

# it has two cols, tDate, and val
# 周频率数据
wts = pd.DataFrame({'Val': [i + 3 * np.cos(i * 6 / np.pi) for i in np.linspace(0, 12 * np.pi, num=104 + 6)],
                    'tDate': pd.date_range(start='1999-01-01', periods=104 + 6, freq='W')}, )
mts = pd.DataFrame({'Val': [i + 3 * np.cos(i * 6 / np.pi) for i in np.linspace(0, 12 * np.pi, num=36 + 6)],
                    'tDate': pd.date_range(start='1999-01-01', periods=36 + 6, freq='M')}, )


LINE_COLOR = ['tab:blue', 'tab:brown',
              'tab:orange', 'tab:pink',
              'tab:green', 'tab:purple',
              'crimson', 'maroon',
              'black', 'navy']


def seasonal_decomp(x, data, time_index, fmt='%m-%d'):
    '''
    拆分时间序列；返回一个 dataframe，columns 是年，row 是不含年的日期。
    :param x:
    :param data:
    :param time_index:
    :param fmt: 日期显示格式 str
    :return:
    '''
    data = data[[x, time_index]].copy()
    if data[x].isna().all():
        logging.info('Empty Column')
        return
    data[x] = pd.to_numeric(data[x])
    if pd.api.types.is_numeric_dtype(data[time_index]):
        data.loc[:, time_index] = pd.to_datetime(data[time_index].astype(str))
    else:
        data.loc[:, time_index] = pd.to_datetime(data[time_index])
    data['Year'] = data[time_index].dt.year
    data['Month'] = data[time_index].dt.month
    data['DateWOYear'] = data[time_index].dt.strftime(fmt)
    grouped = data.groupby('Year')

    years = []
    for y, group in grouped:
        df = group[['DateWOYear', x]].copy()
        df.rename(columns={x: y}, inplace=True)
        years.append(df)
    df = reduce(lambda x, y: pd.merge(x, y, on='DateWOYear', how='outer'), years)
    return df


def interpolate(df, col, last):
    df_interpolate = df.copy()
    df_interpolate.loc[:, col[:-1]] = df_interpolate.loc[:, col[:-1]].interpolate().fillna(method='bfill')
    df_interpolate.loc[: df_interpolate[last].last_valid_index(), last] = df_interpolate[last].loc[: df_interpolate[
        last].last_valid_index()].interpolate().fillna(method='bfill')
    return df_interpolate


def seasonal_plot(x, data: pd.DataFrame, time_index='tDate', xlabel='Month', title='Demo', ):
    '''
    :param x: data 中的 target col
    :param data: DataFrame
    :param freq: 指标频率 'M' or 'W'
    :param time_index: data 中的时间序列
    :param xlabel: x轴名称
    :param title: 图像标题
    :return:
    '''
    # Prepare data
    fmt = '%m-%d'
    df = seasonal_decomp(x, data, time_index, fmt)
    if df is None:
        return
    df = df.sort_values(by='DateWOYear')
    df.reset_index(drop=True, inplace=True)

    # Pesudo date index
    df['DateWOYear'] = pd.to_datetime(df['DateWOYear'].apply(
        lambda x: datetime.datetime.strptime('2008-' + x, '%Y-%m-%d')))
    pseudo_date = df['DateWOYear'].tolist()
    df.set_index('DateWOYear', inplace=True)
    pseudo_start = datetime.date(2008, 1, 1)
    pseudo_end = datetime.date(2008, 12, 31)
    pseudo_date.extend(list(
        pd.date_range(start=pseudo_start, end=pseudo_end, freq='MS')))
    pseudo_date.sort()
    df = df.reindex(pseudo_date)

    # 插值
    col = sorted([i for i in df.columns])
    last = col[-1]
    df_interpolated = interpolate(df, col, last)

    # Plot the graph
    months = MonthLocator(range(1, 13), bymonthday=1, interval=1)
    months_fmt = DateFormatter("%b'%d")
    color_map = {i: j for i, j in zip(col, random.sample(LINE_COLOR, k=len(col)))}
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt, )
    ax.set_xlim([df.index[0] - datetime.timedelta(days=10), df.index[-1] + datetime.timedelta(days=7)])

    for s in col[:-1]:
        ax.plot_date(df.index, df_interpolated[s], '-',
                     linewidth=1, color=color_map[s], label=str(s))
        ax.plot_date(df.index, df[s], '.', color=color_map[s], markersize=2)

    ax.plot_date(df.index, df_interpolated[last], '-', linewidth=3, color=color_map[last], label=str(last))
    ax.plot_date(df.index, df[last], '.', color=color_map[last], markersize=8)
    ax.plot_date(df[last].last_valid_index(), df[last].loc[df[last].last_valid_index()], 'r^',
                 markersize=16, label='Latest Value')
    ax.autoscale_view()
    ax.legend(bbox_to_anchor=(1.05, 1),  loc='upper left')
    ax.set_title(title, )
    return ax




def plot_correlogram(x, lags=None, title=None):
    def hurst(ts):
        """Returns the Hurst Exponent of the time series vector ts"""
        # Create the range of lag values
        lags = range(2, 100)

        # Calculate the array of the variances of the lagged differences
        tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

        # Use a linear fit to estimate the Hurst Exponent
        poly = polyfit(log(lags), log(tau), 1)

        # Return the Hurst exponent from the polyfit output
        return poly[0] * 2.0


    lags = min(10, int(len(x) / 5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    x.plot(ax=axes[0][0])
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f} \nHurst: {round(hurst(x.values), 2)}'
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
    axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].relim()
    axes[1][1].relim()
    axes[1][0].autoscale_view(True, True, True)
    axes[1][1].autoscale_view(True, True, True)

    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    return axes



def lag_plot_(ts):
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    plt.title('Lag plot')

    # The axis coordinates for the plots
    ax_idcs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1)
    ]

    for lag, ax_coords in enumerate(ax_idcs, 1):
        ax_row, ax_col = ax_coords
        axis = axes[ax_row][ax_col]
        lag_plot(ts, lag=lag, ax=axis)
        axis.set_title(f"Lag={lag}")

    return 




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ax = seasonal_plot(x='Val', data=wts, time_index='tDate', title='Weekly Data Demo', )
    # # ax2 = seasonal_plot(x='Val', data=mts, time_index='tDate', title='Monthly Data Demo', )
    # plt.show()
    df = pd.read_excel('energy.xlsx')
    for col in df.columns:
        if col == 'tDate':
            continue
        ax = seasonal_plot(col , df, title=col)
        plt.show()
