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

    This class is for the purpose to test multiple models in at once on the dataset.
    It should provide a clue what types of model fit to the dataset best.

    """
    all_models = dict(all=[LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
                           KNeighborsRegressor, DecisionTreeRegressor, ExtraTreeRegressor,
                           RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, MLPRegressor,
                           LGBMRegressor, XGBRegressor, CatBoostRegressor],
                      linear=[LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge],
                      tree=[DecisionTreeRegressor, ExtraTreeRegressor, RandomForestRegressor, BaggingRegressor,
                            GradientBoostingRegressor, LGBMRegressor, XGBRegressor, CatBoostRegressor],
                      neighbor=[KNeighborsRegressor],
                      neuronal=[MLPRegressor])

    def __init__(self, target_col: str, feature_cols: list,
                 save_path='lazy_report.json',
                 sample_size='all',
                 fit_model_class='all',
                 cross_val_splits=5, n_threads=5,
                 scorer_metrics=('r2', 'neg_mean_absolute_error'),
                 save_estimator=False):
        """
        After initialization of this class call the method generate_report() to run the model fits
        on the dataset.
        After the JSON report was generated you can call all all other methods like plot_reports()
        It is possible to also initiate the class only and call the plot_reports() if save_path JSON
        was generated in another run.


        Parameters
        ----------
        target_col: str
            Prediction Target Column

        feature_cols: list
            Feature columns for the model.

        save_path:
            Path where to save model reports

        sample_size: int or str
            sample size of data to be taken or just "all" for all data

        fit_model_class: str
            One of: "all", "linear", "tree", "neighbor", "neuronal"

        cross_val_splits:
            Number of splits for cv

        n_threads:
            Multiple threads use for crossvalidation

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
        self.fit_model_class = fit_model_class
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
        with tqdm(TooLazyForRegression.all_models[self.fit_model_class]) as t:
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
                        self.estimators[model.__name__] = res['estimator']
                        del res['estimator']

                    # Set negative metric to positive and delete obj
                    for key in res.keys():
                        if 'neg_' in key:
                            new_key = key.split('_')
                            new_key.remove('neg')
                            new_key = '_'.join(new_key)
                            res[new_key] = np.abs(res[key])
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

    def plot_report(self, plot_include_time=False, plot_include_mae=False):
        """
        Plots Report from JSON Report which was constructed in the generate_report() Method.

        Parameters
        ----------
        plot_include_time: bool
            If time should be included as metric too
        
        plot_include_mae: bool
            If mean absolute error should be included as metric too

        Returns
        -------
        plot: plt.plot_object
        """
        try:
            report = pd.read_json(self.save_path)
        except FileNotFoundError:
            raise FileNotFoundError('Call method generate_report() first or set save_path attribute to an '
                                    'existing JSON-file')

        self.report = report
        report = report.explode(column=report.columns.to_list()).reset_index()
        report = report.rename(columns={'index': 'scorer'})
        report = report.melt(id_vars='scorer', var_name='model', value_name='score')
        report = report[~report['scorer'].isin(['score_time'])]
        y_metric_names = np.asarray(('Training time (s)', 'R2 score', 'mean_absolute_error'))
        if not plot_include_time:
            report = report[~report['scorer'].isin(['fit_time'])] #Training time (s)
            y_metric_names = np.delete(y_metric_names, np.where(y_metric_names == 'Training time (s)'), axis=0)
        if not plot_include_mae:
            report = report[~report['scorer'].isin(['test_mean_absolute_error'])] #Training time (s)
            y_metric_names = np.delete(y_metric_names, np.where(y_metric_names == 'mean_absolute_error'), axis=0)

        
        if self.fit_model_class == "all": #all
            colors = ["#4EA0C4", "#4EA0C4", "#4EA0C4","#4EA0C4","#4EA0C4","#C44E4E","#449D3E","#449D3E","#449D3E","#449D3E","#449D3E","#C4C44E","#449D3E","#449D3E","#449D3E"]

        if self.fit_model_class == "linear": #linear
            colors = ["#4EA0C4", "#4EA0C4", "#4EA0C4","#4EA0C4","#4EA0C4"]   

        if self.fit_model_class == "tree": #tree
            colors = ["#449D3E","#449D3E","#449D3E","#449D3E","#449D3E","#449D3E","#449D3E","#449D3E"]   

        if self.fit_model_class == "neighbor": #neighbor
            colors = ["#C44E4E"]  

        if self.fit_model_class == "neuronal": #neuronal
            colors = ["#C4C44E"] 
            

        metric_names = report['scorer'].unique()
        ncols = 1
        nrows = int(len(metric_names) / ncols) + 1

        fig = plt.subplots(figsize=(16, 7 * nrows))
        for i in range(len(metric_names)):
            plt.subplot(nrows, ncols, i + 1)
            tmp = report[report['scorer'] == metric_names[i]]
            p = sns.boxplot(data=tmp, x='score', y='model', palette=colors)

            p.set_title(f'{y_metric_names[i].upper()}  cross-validated on {self.cross_val_splits} folds',
                        loc='left', fontsize=13)
            sns.despine()
            p.set_xlabel(y_metric_names[i])
            p.set_ylabel("")
            p.grid(axis = 'y')

        plt.subplots_adjust(hspace=.4)
        plt.show()
        report_sum_mean = report.groupby(['model']).mean()
        report_sum_mean['model'] = report_sum_mean.index

        if self.fit_model_class == "all": #all
            options = ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "BayesianRidge"]
            linear_mean_info = report_sum_mean.loc[report_sum_mean["model"].isin(options)].mean() 
            options = ["DecisionTreeRegressor", "ExtraTreeRegressor", "RandomForestRegressor", "BaggingRegressor",
                                    "GradientBoostingRegressor", "LGBMRegressor", "XGBRegressor", "CatBoostRegressor"]
            tree_mean_info = report_sum_mean.loc[report_sum_mean["model"].isin(options)].mean()
            options = ["KNeighborsRegressor"]
            knn_mean_info = report_sum_mean.loc[report_sum_mean["model"].isin(options)].mean()
            options = ["MLPRegressor"]
            mlp_mean_info = report_sum_mean.loc[report_sum_mean["model"].isin(options)].mean() 
            print("Mean of all linear-Models:",np.round(linear_mean_info[0],3))
            print("Mean of all tree-Models:",np.round(tree_mean_info[0],3))
            print("Mean of all neighbor-Models:",np.round(knn_mean_info[0],3))
            print("Mean of all neuronal-Models:",np.round(mlp_mean_info[0],3))

        if self.fit_model_class == "linear": #linear
            options = ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "BayesianRidge"]
            mean_info = report_sum_mean.loc[report_sum_mean["model"].isin(options)].mean()
            print("Mean of all linear-Models:",np.round(mean_info[0],3))

        if self.fit_model_class == "tree": #tree
            options = ["DecisionTreeRegressor", "ExtraTreeRegressor", "RandomForestRegressor", "BaggingRegressor",
                                    "GradientBoostingRegressor", "LGBMRegressor", "XGBRegressor", "CatBoostRegressor"]
            mean_info = report_sum_mean.loc[report_sum_mean["model"].isin(options)].mean()
            print("Mean of all tree-Models:",np.round(mean_info[0],3))

        if self.fit_model_class == "neighbor": #neighbor
            options = ["KNeighborsRegressor"]
            mean_info = report_sum_mean.loc[report_sum_mean["model"].isin(options)].mean() 
            print("Mean of all neighbor-Models:",np.round(mean_info[0],3))

        if self.fit_model_class == "neuronal": #neuronal
            options = ["MLPRegressor"]
            mean_info = report_sum_mean.loc[report_sum_mean["model"].isin(options)].mean()
            print("Mean of all neuronal-Models:",np.round(mean_info[0],3))

    def mean_r2(self):
        """
            r2 mean from JSON Report which was constructed in the generate_report() Method.

            Parameters
            ----------
            None

            Returns
            -------
            Dataframe
            """
        plot_order = ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "BayesianRidge",
                      "KNeighborsRegressor", "DecisionTreeRegressor", "ExtraTreeRegressor",
                      "RandomForestRegressor", "BaggingRegressor", "GradientBoostingRegressor", "MLPRegressor",
                      "LGBMRegressor", "XGBRegressor", "CatBoostRegressor"]
        
        report = self.report
        report = report.explode(column=report.columns.to_list()).reset_index()
        report = report.rename(columns={'index': 'scorer'})
        report = report.melt(id_vars='scorer', var_name='model', value_name='score')
        report = report[~report['scorer'].isin(['score_time'])]
        report = report[~report['scorer'].isin(['fit_time'])] #Training time (s)
        report = report[~report['scorer'].isin(['test_mean_absolute_error'])] #Training time (s)
        return report

            
    def plot_residuals(self, data):
        """

        Returns
        -------

        """
        if str(self.estimators) == '{}' or self.save_estimator is False:
            raise NotImplementedError('Initiate Class with param save_estimator = True')
        else:
            X = data[self.feature_cols]
            y = data[self.target_col]
            ncols = 3
            nrows = int(len(TooLazyForRegression.all_models) / ncols) + 1

            fig = plt.subplots(figsize=(20, 4*nrows))
            for i, key in enumerate(self.estimators.keys()):
                y_pred = self.estimators[key][0].predict(X)
                resid = y - y_pred

                plt.subplot(nrows, ncols, i+1)
                p = sns.scatterplot(x=y_pred, y=resid, alpha=.3, color='lightskyblue')
                plt.hlines(y=0, xmax=np.max(y_pred), xmin=np.min(y_pred),
                           linestyles='--', colors='grey')
                p.set_title(f'{key} Residual Plot')
                p.set_xlabel(r'$\hat{y}$')
                p.set_ylabel(r'Residual $\epsilon$')

            plt.subplots_adjust(hspace=.3)
            plt.show()


if __name__ == '__main__':
    # Read data
    data = pd.read_table('../data/data_mpa.txt', sep=' ')

    feature_cols = data.columns.to_list()
    feature_cols.remove('size_mm')
    feature_cols.remove('start_time')

    lazy = TooLazyForRegression(save_path='../eda/lazy_report.json',
                                target_col='size_mm', feature_cols=feature_cols,
                                fit_model_class='linear',
                                sample_size=1000, cross_val_splits=5, n_threads=5,
                                save_estimator=True)
    lazy.generate_report(df=data)
