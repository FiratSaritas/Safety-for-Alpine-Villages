from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)


class TooLazyForRegression(object):
    """
    This class is for the purpose to test multiple models in one iteration over the dataset
    and
    """
    all_models = [LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
                  SVR, KNeighborsRegressor, DecisionTreeRegressor, ExtraTreeRegressor,
                  RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, MLPRegressor,
                  LGBMRegressor, XGBRegressor, CatBoostRegressor(verbose=False)]

    def __init__(self, target_col, feature_cols,
                 save_path='lazy_report.json',
                 sample_size=30000,
                 cross_val_splits=5, n_threads=5,
                 scorer_metrics=('r2', 'neg_mean_absolute_error'),
                 save_estimator=False):
        """

        Parameters
        ----------
        target_col: str
            Prediction Target Column

        feature_cols: list

        save_path:
            path where to save model reports

        sample_size: int or str
            sample size of data to be taken or just 'all' for
            all data

        cross_val_splits:
            number of splits for cv

        n_threads:
            multiple threads use for crossvalidation

        scorer_metrics: tuple
            Tuple of sklearn metrics, Check on Sklearn which metrics
            are availbale.

        save_estimator: bool
            Wether to save an estimator or not as class attr.
        """
        self.save_path = save_path
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.sample_size = sample_size
        self.cross_val_splits = cross_val_splits
        self.n_threads = n_threads
        self.scorer_metrics = scorer_metrics
        self.save_estimator = save_estimator
        # In-Code attrs:
        self.report = None
        self.estimators = {}

    def generate_report(self, df):
        """
        Generates a CV Report for each model defined as class Attribute. The Data will be
        preprocessed in each Steps with the Standard Scaler.

        Returns
        -------
        Report: JSON
            Report as a JSON File
        """
        # Make idx selection
        if self.sample_size != 'all':
            idx = np.random.permutation(df.index.to_list())[:self.sample_size]
            df = df.loc[idx]

        df = df.dropna(axis=0)
        X, y = df[self.feature_cols], df[self.target_col].to_numpy()

        cv_results = {}
        with tqdm(TooLazyForRegression.all_models) as t:
            for model in t:
                t.set_description(desc=f'Fitting {model.__name__}')
                try:
                    # Create Pipeline and fit
                    pipe = Pipeline(steps=[('scaler', StandardScaler()),
                                           (model.__name__, model())])
                    res = cross_validate(estimator=pipe,
                                         X=X, y=y, cv=self.cross_val_splits,
                                         scoring=self.scorer_metrics,
                                         n_jobs=self.n_threads,
                                         return_estimator=self.save_estimator)
                    if self.save_estimator:
                        self.estimators[model.__name__] = res[0]
                        del res[0]

                    # Set negative metric to positive and delete obj
                    for key in cv_results.keys():
                        if key[:3] == 'neg':
                            res[key[4:]] = np.abs(res[key[4:]])
                            del res[key]

                    # Type conversion of arrays
                    for key, val in res.items():
                        if type(val) == np.ndarray:
                            val = val.tolist()
                            res[key] = val

                    cv_results[model.__name__] = res

                except KeyboardInterrupt as ke:
                    # Stops code when interrupted by keyboard
                    raise ke
        # Save Report
        with open(self.save_path, 'w') as json_file:
            json.dump(obj=cv_results, fp=json_file)
        self.report = pd.read_json(self.save_path)


    def plot_report(self, plot_include_time=False):
        """
        Plots Report from JSON Report which was constructed in the generate_report() Method.

        Parameters
        ----------
        plot_include_time: bool
            If time should be included as metric too

        Returns
        -------
        plot: plt.plot_object
        """
        report = pd.read_json(self.save_path)
        self.report = report
        report = report.explode(column=report.columns.to_list()).reset_index()
        report = report.rename(columns={'index': 'scorer'})
        report = report.melt(id_vars='scorer', var_name='model', value_name='score')
        report = report[~report['scorer'].isin(['score_time'])]
        if not plot_include_time:
            report = report[~report['scorer'].isin(['fit_time'])]

        metric_names = report['scorer'].unique()
        ncols = 1
        nrows = int(len(metric_names) / ncols) + 1

        fig = plt.subplots(figsize=(18, 5 * nrows))
        for i in range(len(metric_names)):
            plt.subplot(nrows, ncols, i + 1)
            tmp = report[report['scorer'] == metric_names[i]]
            p = sns.boxplot(data=tmp, x='score', y='model', color='lightskyblue')

            p.set_title(f'{metric_names[i]}', loc='left', fontsize=13)
            sns.despine()
            p.set_xlabel('')
            p.set_ylabel(f'Score')

        plt.subplots_adjust(hspace=.2)
        plt.show()


    def plot_residuals(self):
        """

        Returns
        -------

        """
        if str(self.estimators) == '{}' and self.save_estimator == False:
            raise NotImplementedError('Initiate Class with param save_estimator = True')
        else:
            pass



if __name__ == '__main__':
    # Read data
    data = pd.read_table('../data/data_mpa.txt', sep=' ')

    feature_cols = data.columns.to_list()
    feature_cols.remove('size_mm')
    feature_cols.remove('start_time')

    lazy = TooLazyForRegression(save_path='lazy_report.json',
                                target_col='size_mm', feature_cols=feature_cols,
                                sample_size=30000, cross_val_splits=5, n_threads=5)
    lazy.generate_report(df=data)
