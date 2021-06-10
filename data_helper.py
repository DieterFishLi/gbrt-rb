# Created by Dieter Fish at 2021/5/12
import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fracdiff import FracdiffStat
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OneHotEncoder
from sklearn.utils.validation import _deprecate_positional_args

f = FracdiffStat()
p = PCA(n_components=1)
imp = SimpleImputer(missing_values=np.nan, strategy="mean")


def auto_label_encode(df, num=None, col=None, by=None):
    """
    对序列进行分Bin编码
    :param df:
    :param col:
    :param num:
    :param by:
    :return:
    """
    if not col:
        col = df.columns[0]
    if not num:
        num = 10
    if by == 'entropy':
        min_entropy = -float('inf')
        best_num = None
        for i in range(2, num):
            cur_std = entropy(pd.cut(df[col], bins=i, ).value_counts().values)
            print(cur_std)
            if cur_std > min_entropy:
                best_num = i
                min_entropy = cur_std
        print(best_num)
    else:
        best_num = num
    df1 = pd.cut(df[col], bins=best_num, labels=[str(i) for i in range(best_num)])
    df1.name = col
    return df1.to_frame()


_auto_filter = ["rolling", "expanding", ...]


def auto_filter(df):
    """
    自动对特征序列添加滤波 作为新特征？

    :param df:
    :return:
    """
    return


def load_data():
    file = pathlib.Path("data", "课题一数据.xls")
    df = pd.read_excel(file, sheet_name="螺纹钢相关")
    del df[df.columns[1]]
    df["日期"] = pd.to_datetime(df["日期"])
    df.set_index(df["日期"], inplace=True)

    df['logret'] = 100 * df['RB:活跃合约'].apply(np.log).diff()

    df2 = pd.read_excel(file, sheet_name="宏观数据", index_col=0)
    df2.index = pd.to_datetime(df2.index)
    # 固定资产投资完成额分行业同比(月)
    df3 = pd.read_csv(pathlib.Path("data", "investment.csv"),
                      index_col=0)
    df3.index = pd.DatetimeIndex(pd.to_datetime(df3.index))

    # add more features ...
    return df, df2, df3


def load_rb(start=None, end=None):
    """
    加载合约价格
    :param start:
    :param end:
    :return:
    """
    file = pathlib.Path("data", "rb_continuous_contracts.csv")
    df = pd.read_csv(file, index_col=0)
    df.index = pd.DatetimeIndex(df.index)

    if start == None and end == None:
        df = df
    else:
        df = df.loc[start:end]
    df = _cont_contract_price_adj(df)
    return df


def _cont_contract_price_adj(df, method='ratio'):
    """
    移除换月跳空
    :param df:
    :param method:
    :return:
    """
    df = df.copy()
    df['close_adj'] = df['close']
    df['settle_adj'] = df['settle']
    prv_contract = None
    for idx, row in df.reset_index().iterrows():
        cur_contract = row.trade_hiscode
        if prv_contract is None:
            prv_contract = cur_contract
            continue
        if method == "ratio":
            if cur_contract != prv_contract:
                cur_close = row.close
                cur_settle = row.settle
                prv_close = df.iloc[idx - 1].close
                prv_settle = df.iloc[idx - 1].settle
                close_ratio = cur_close / prv_close
                settle_ratio = cur_settle / prv_settle
                df.iloc[:idx, -2] *= close_ratio
                df.iloc[:idx, -1] *= settle_ratio
        elif method == 'gap':
            if cur_contract != prv_contract:
                cur_close = row.close
                cur_settle = row.settle
                prv_close = df.iloc[idx - 1].close
                prv_settle = df.iloc[idx - 1].settle
                close_gap = cur_close - prv_close
                settle_gap = cur_settle - prv_settle
                df.iloc[:idx, -2] += close_gap
                df.iloc[:idx, -1] += settle_gap
        prv_contract = cur_contract
    return df


def process_feature(file, start, end):
    """
    加载特征数据，简单清洗 和 预处理
    :param file:
    :param start:
    :param end:
    :return:
    """
    if 'CPI' in file:
        return process_cpi(file, start, end)
    elif 'inventory' in file:
        return process_inventory(file, start, end)
    elif 'investment' in file:
        return process_sector_investment(file, start, end)
    elif 'money' in file:
        return process_money_supply(file, start, end)
    elif 'PMI' in file:
        return process_pmi(file, start, end)
    elif 'PPI' in file:
        return process_ppi(file, start, end)
    elif 'SHIBOR' in file:
        return process_shibor(file, start, end)
    elif 'ta.csv' in file:
        return process_ta(file, start, end)
    elif "社会融资规模" in file:
        return process_private_financing(file, start, end)
    elif "freight_cost.csv" in file:
        return process_freight_cost(file, start, end)
    elif "google_trend_" in file:
        return process_google_trend(file, start, end)
    elif "foreign_price.csv" in file:
        return process_foreign_price(file, start, end)
    elif "spot_spread.csv" in file:
        return process_spot_spread(file, start, end)
    elif "mysteel_price_index.csv" in file:
        return process_price_index(file, start, end)
    elif "weather.csv" in file:
        return process_weather(file, start, end)
    elif "purchase_amount.csv" in file:
        return process_purchase_amt(file, start, end)
    elif "房地产开发、销售(月).csv" in file:
        return process_real_estate(file, start, end)


def process_cpi(file, start, end):
    '''
    对 CPI 特征进行清洗
    1. 更新特征
    2. 平稳化 Fractional Difference算法
    :param file:
    :param start:
    :param end:
    :return:
    '''
    df = pd.read_csv(file, index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    df.dropna(inplace=True)
    df = df.shift(1)  # 更新延迟
    df.fillna(method='bfill', inplace=True)
    df = df.loc[ : end]

    cpi_LP = ['CPI:环比', 'CPI:不包括食品和能源(核心CPI):环比']
    cpi_POP = df.columns.difference(cpi_LP)

    df_new = _process_cpi(df, )
    return df_new.loc[start: end]


def _process_cpi(df, ):
    cpi_LP = ['CPI:环比', 'CPI:不包括食品和能源(核心CPI):环比']
    cpi_POP = df.columns.difference(cpi_LP)
    df_new = pd.DataFrame()
    df_new['cpi_LP'] = f.fit_transform(p.fit_transform(df[cpi_LP])).flatten()
    df_new['cpi_POP'] = f.fit_transform(p.fit_transform(df[['CPI:累计同比', 'CPI:不包括食品和能源(核心CPI):当月同比']])).flatten()
    df_new.index = df.index
    return df_new


def process_inventory(file, start, end):
    """
    处理 社会螺纹钢库存指标
    1. 去季节性，取 STL 分解后的残差
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    df.dropna(inplace=True)
    df = df.loc[: end]
    result = _process_inventory(df)
    return result.loc[start: end]


def _process_inventory(df):
    inventory_df = 100 * df.pct_change()
    inventory_df.fillna(method='bfill', inplace=True)
    inventory_resid = sm.tsa.STL(inventory_df['螺纹钢库存'], period=52).fit().resid
    inventory_resid.name = 'spot_inventory_level_resid'
    item = inventory_resid.to_frame()
    return item


def process_sector_investment(file, start, end):
    """
    在看盘软件叠加主力价格和黑色金属冶炼投资累计同比，后者对前者有一定领先。
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0).shift(1).dropna()
    df.index = pd.DatetimeIndex(df.index)
    df = df.loc[: end]
    investment_new = _process_sector_investment(df)
    return investment_new.loc[start: end]


def _process_sector_investment(df):
    investment_new = pd.DataFrame()
    try:
        investment_new['indus_invest'] = f.fit_transform(df.values).flatten()
    except:
        investment_new['indus_invest'] = (1+df.pct_change()).cumprod().values.flatten()
    investment_new.index = df.index
    return investment_new


def process_money_supply(file, start, end):
    """
    处理货币供应特征 M1&M2
    1. 平稳化 Fractional Difference算法处理
    2. 发布延迟处理
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0)
    df = df.dropna()
    df = df.shift(1)
    df = df.dropna()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
    df = df.loc[:end]
    money_new = pd.DataFrame()
    try:
        for col in df.columns:
            money_new[col] = f.fit_transform(df[col].values.reshape(-1, 1)).flatten()
    except:
        money_new['money_supply'] = p.fit_transform(df[df.columns]).flatten()
    money_new.index = df.index
    return  money_new.loc[start:end]



def process_pmi(file, start, end):
    """
    处理 PMI 相关特征
    1. PCA 降维
    2. 发布延迟处理
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0)
    df = df.shift(1)
    df.index = pd.to_datetime(df.index)
    df = df.loc[:end]

    pmi_new = pd.DataFrame()
    pmi_new['pmi'] = p.fit_transform(imp.fit_transform(df)).flatten()
    pmi_new.index = df.index
    return pmi_new.loc[start:end]


def process_ppi(file, start, end):
    """
    PPI 平稳处理
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0)
    df = df.dropna()
    df = df.shift(1)
    df = df.dropna()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
    df = df.loc[:end]
    ppi_new = pd.DataFrame()
    try:
        ppi_new['ppi'] = f.fit_transform(df.values).flatten()
    except:
        ppi_new['ppi'] = df.values.flatten()
    ppi_new.index = df.index

    if not ppi_new.isna().all().all():
        return ppi_new
    else:
        ppi_new[:] = df.values
    return ppi_new.loc[start:end]


def process_shibor(file, start, end):
    """
    处理 SHIBOR
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0)
    df = df.dropna()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
    df = df.loc[:end]
    shibor = pd.DataFrame()
    try:
        shibor['shibor_fd'] = f.fit_transform(df.values).flatten()
    except:
        ...
    shibor['shibor_pct_change'] = 100 * df.pct_change().values.flatten()
    shibor.index = df.index
    return shibor.loc[start:end]


def process_ta(file, start, end):
    """
    技术分析指标
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0).dropna()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
    df['Vol/OI'] = df['Volume'] / df['OI']
    return df.loc[start:end][['RSI', 'KDJ', 'Vol/OI']]


def process_private_financing(file, start, end):
    """
    处理社融指标;
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0).dropna()
    df = df.shift(1)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
    df = df.loc[:end].dropna()
    df['private_financing_fd'] = f.fit_transform(df['社会融资规模:当月值'].values.reshape(-1, 1)).flatten()
    #  df['private_financing_d'] = df['社会融资规模:当月值'].diff()
    if not df['private_financing_fd'].isna().all():
        return df[['private_financing_fd']].loc[start:end]


# --------5/25
def process_freight_cost(file, start, end):
    """
    上游铁矿石运输成本
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0).dropna()
    df.index = pd.DatetimeIndex(df.index)
    df = df.loc[: end]

    bdi = pd.Series(data=f.fit_transform(df.iloc[:, 0].values.reshape(-1, 1)).flatten(), index=df.iloc[:, 0].index)
    bdi.name = 'bdi'
    bdi = bdi.to_frame()

    io_freight_cost = pd.Series(data=f.fit_transform(p.fit_transform(df.iloc[:, 1:].values)).flatten(),
                                index=df.iloc[:, 1:].index)
    io_freight_cost.name = 'io_freight_cost'
    io_freight_cost = io_freight_cost.to_frame()

    return bdi.join(io_freight_cost, how='outer').loc[start:end]


def process_foreign_price(file, start, end):
    """
    国际市场螺纹钢价格
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0).dropna()
    df.index = pd.DatetimeIndex(df.index)
    # df.dropna(inplace=True)
    df = df.shift(1)  # 更新延迟
    # df.fillna(method='bfill', inplace=True)
    df = df.loc[: end].dropna()
    df1 = pd.DataFrame()
    df1['foreign_price'] = f.fit_transform(df.mean(axis=1).values.reshape(-1, 1)).flatten()
    df1.index = df.index
    return df1.loc[start:end]


def process_price_index(file, start, end):
    """
    MySteel 钢价指数
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0).dropna()
    df.index = pd.DatetimeIndex(df.index)
    # df.dropna(inplace=True)
    # df = df.shift(1) # 更新延迟
    # df.fillna(method='bfill', inplace=True)
    df = df.loc[: end]
    df['MySteel_index'] = f.fit_transform(df.mean(axis=1).values.reshape(-1, 1)).flatten()
    return df['MySteel_index'].to_frame().loc[start:end]


def process_purchase_amt(file, start, end):
    """
    终端线螺采购量
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0).dropna()
    df.index = pd.DatetimeIndex(df.index)
    # df.dropna(inplace=True)
    # df = df.shift(1) # 更新延迟
    # df.fillna(method='bfill', inplace=True)
    df = df.loc[: end]
    df['purchase_amount'] = f.fit_transform(df.values).flatten()
    return df['purchase_amount'].to_frame().loc[start:end]


def process_spot_spread(file, start, end):
    """
    现货价差
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0).dropna()
    df.index = pd.DatetimeIndex(df.index)
    df = df.loc[: end]
    region_spread_col = [i for i in df.columns if "区域价差" in i]
    product_spread_col = [i for i in df.columns if "品种价差" in i]

    contract_basis_col = [i for i in df.columns if "合约基差" in i]
    basis_region_spread_col = ['基差:螺纹上海基差', '基差:螺纹天津基差',
                               '基差:螺纹广州基差', '基差:螺纹沈阳基差']

    df['region_spread'] = p.fit_transform(df[region_spread_col])
    df['product_spread'] = p.fit_transform(df[product_spread_col])
    df['contract_basis'] = p.fit_transform(df[contract_basis_col])
    df['region_basis'] = p.fit_transform(df[basis_region_spread_col])
    return df[['region_spread', 'product_spread', 'contract_basis', 'region_basis']].loc[start:end]


def process_weather(file, start, end):
    """
    唐山空气质量
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0).dropna()
    df.index = pd.DatetimeIndex(df.index)
    # df.dropna(inplace=True)
    # df = df.shift(1) # 更新延迟
    # df.fillna(method='bfill', inplace=True)
    df = df.loc[:end]
    if df.empty:
        return
    return auto_label_encode(df, num=4, col='唐山市PM2.5').loc[start: end]
    # return pd.cut(df['唐山市AQI全国当日排名'], bins=10, labels=[str(i) for i in range(10)]).to_frame()


def process_google_trend(file, start, end):
    """
    关键词搜索热度，反应中澳关系 上游铁块成本
    环保限产 供应情况变化
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0).dropna()
    df.index = pd.DatetimeIndex(df.index)
    df = df.loc[start: end]

    if 'google_trend_china_aus_iron_ore.csv' in file:
        return pd.cut(df['china australia iron ore: (全球)'], bins=4, labels=[str(i) for i in range(4)]).to_frame()

    if 'google_trend_去产能.csv' in file:
        return pd.cut(df[df.columns[0]], bins=6, labels=[str(i) for i in range(6)]).to_frame()

    if "google_trend_china_aus_coal.csv" in file:
        return pd.cut(df[df.columns[0]], bins=4, labels=[str(i) for i in range(4)]).to_frame()

    if "google_trend_china_aus_trade.csv" in file:
        return pd.cut(df[df.columns[0]], bins=5, labels=[str(i) for i in range(5)]).to_frame()



##### 6.02
def process_real_estate(file, start, end):
    """
    处理社融指标;
    :param file:
    :param start:
    :param end:
    :return:
    """
    df = pd.read_csv(file, index_col=0).dropna()
    df = df.shift(1)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
    df = df.loc[:end].dropna()
    df['开工'] = f.fit_transform(df['房屋新开工面积:累计同比'].values.reshape(-1, 1)).flatten()
    #  df['private_financing_d'] = df['社会融资规模:当月值'].diff()
    if not df['开工'].isna().all():
        return df[['开工', '房屋新开工面积:累计同比']].loc[start:end]
    else:
        return df[['房屋新开工面积:累计同比']].loc[start:end]

class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """
    From @https://www.kaggle.com/marketneutral/purged-time-series-cv-xgboost-optuna#Optuna-Hyperparam-Search-for-XGBoost

    Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]

                train_array = np.sort(np.unique(
                    np.concatenate((train_array,
                                    train_array_tmp)),
                    axis=None), axis=None)

            train_end = train_array.size

            for test_group_idx in unique_groups[group_test_start:
            group_test_start +
            group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                    np.concatenate((test_array,
                                    test_array_tmp)),
                    axis=None), axis=None)

            test_array = test_array[group_gap:]

            if self.verbose > 0:
                pass

            yield [int(i) for i in train_array], [int(i) for i in test_array]


def gen_data_matrix(y, features):
    mat = y.copy()
    for k, f in features.items():
        if f is not None:
            mat = mat.join(f, how="outer")
    idx = mat[y.columns[0]].isna()
    mat.fillna(method='ffill', inplace=True)
    mat = mat[~idx]
    #     mat.dropna(subset=y.columns, inplace=True)
    return mat


def get_feature_dict(feature_files, start, end):
    feature_dfs = dict()
    cat_features = []
    numerical_features = []
    for i in feature_files:
        d = process_feature(i, start, end)
        if d is not None:
            feature_dfs[i] = d
            try:
                cols = d.columns
            except:
                print(i)
            num_cols = d._get_numeric_data().columns
            if len(num_cols) > 0:
                numerical_features.extend(list(num_cols))
            cats = set(cols) - set(num_cols)
            if len(cats) > 0:
                cat_features.extend(list(cats))
    return feature_dfs, cat_features, numerical_features


def prepare_model_data(data: pd.DataFrame, y: str,):
    x = data[data.columns.difference([y])]
    numerical_features = x._get_numeric_data().columns
    cat_features = x.columns.difference(numerical_features)
    y = data[y]

    cat_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    num_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    try:
        x[cat_features] = cat_imputer.fit_transform(x[cat_features])
    except ValueError:
        x.loc[:, x.isna().all()[lambda x: x].index] = '0'
        x[cat_features] = cat_imputer.fit_transform(x[cat_features])
    x[numerical_features] = num_imputer.fit_transform(x[numerical_features])
    pt = PowerTransformer()
    mt = MinMaxScaler()
    x[numerical_features] = pt.fit_transform(mt.fit_transform(x.loc[:, numerical_features]))
    return x, y


def prepare_model_data_basline(data: pd.DataFrame, y: str,):
    x = data[data.columns.difference([y])]
    numerical_features = x._get_numeric_data().columns
    cat_features = x.columns.difference(numerical_features)
    y = data[y]
    cat_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    num_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    x[cat_features] = cat_imputer.fit_transform(x[cat_features])
    x[numerical_features] = num_imputer.fit_transform(x[numerical_features])
    pt = PowerTransformer()
    mt = MinMaxScaler()
    x[numerical_features] = pt.fit_transform(mt.fit_transform(x.loc[:, numerical_features]))
    mat1 = x[numerical_features].values
    enc = OneHotEncoder(handle_unknown='ignore')
    mat2 = enc.fit_transform(x[cat_features]).toarray()
    mat = np.hstack((mat1, mat2))
    return mat, y




if __name__ == '__main__':
    import datetime

    start, end = (datetime.date(2012, 11, 1), datetime.date(2019, 12, 31))
    test_featurelist = [r'data\\CPI.csv',
                        r'data\\inventory.csv',
                        r'data\\investment.csv',
                        r'data\\money.csv',
                        r'data\\PMI.csv',
                        r'data\\PPI.csv',
                        r'data\\SHIBOR3个月.csv',
                        r'data\\ta.csv',
                        r'data\\社会融资规模.csv']
    # test_featurelist = glob(r"data\\*.csv")
    # for i in test_featurelist:
    #     process_feature(i ,start, end)
    # process_feature(r'data\\PPI.csv', start, end)
    process_feature(r'data\\investment.csv', start, end)
