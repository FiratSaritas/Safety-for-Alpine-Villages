import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import sys
sys.path.append('..')
import os
import yaml
import datetime as dt
from multiprocessing import Pool
from feature_extract import extract_highest_amplitude_features_with_mp, get_all_sensors_in_df
from plot import plot_residuals, plot_error_per_cat
from lightgbm import LGBMRegressor
import optuna

from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PolynomialFeatures


def objective(trial):
    X_transformed = X_train.copy()
    
    # Pre-Processing
    ## Polynomial Features
    poly = PolynomialFeatures(degree=trial.suggest_int('poly_degree', 1, 4),
                              interaction_only=trial.suggest_categorical('poly_interaction', [True, False]))
    if poly.degree > 1:
        poly_amount_features = trial.suggest_int('poly_amount_features', 1, X_train.shape[1]) 
        X_columns = X_train.columns.to_list()
        poly_feature_select = np.random.choice(a=X_columns, size=poly_amount_features, replace=False)
        trial.set_user_attr('poly_selected_features', poly_feature_select)
        X_poly = poly.fit_transform(X_transformed[poly_feature_select])
        X_transformed = np.concatenate((X_transformed, X_poly), axis=1)
    
    ## PCA for Dim Reduction
    pca = PCA(n_components=trial.suggest_int('n_components', 5, X_transformed.shape[1]))
    X_transformed = pca.fit_transform(X_transformed)        
    
    ## Feature Transformation to normalize data
    apply_feature_transformation = trial.suggest_categorical('apply_feature_transformation', [True, False])
    if apply_feature_transformation:
        transformer = PowerTransformer(standardize=trial.suggest_categorical('pt_standardize', [True, False]))
        X_transformed = transformer.fit_transform(X_train)
        
    
    param = {
        'metric': 'rmse', 
        'random_state': 48,
        'n_estimators': trial.suggest_int('n_estimators', 200, 10000),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
        'max_depth': trial.suggest_int('max_depth', 10, 100),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 500),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
    }
    
    model = LGBMRegressor(**param, n_jobs=8) 
    scores = cross_val_score(estimator=model, 
                             X=X_transformed, y=y_train, 
                             cv=5, scoring='r2')
    return np.mean(scores)


if __name__ == '__main__':
    # 'data_mpa.txt', 'data_spg.txt', 'data_sps.txt'
    config = dict(
        FILEPATHS = ['data_mpa.txt', 'data_spg.txt', 'data_sps.txt'],
        TUNING_ITER = 2,
        N_TRIALS = 50,
        DROP_COLUMNS = ['velocity', 'start_time', 'packnr'],
        LOG_SCALE_TARGET = False,
        MODEL_NAME = 'LGBM',
        SAVE_DIR = './results/20211208/' , 
        EXTRACT_MAX_FEATURES = True,
        DEBUG_RUN = False,
    )
    
    for path in config['FILEPATHS']:
        device_name = path.split('_')[-1].split('.')[0]
        try:
            os.listdir(config['SAVE_DIR'])
        except:
            os.mkdir(config['SAVE_DIR'])
            
        # Save Parametr-configs of file
        with open(config['SAVE_DIR'] + 'config.yaml', 'w') as yaml_file:
            yaml.dump(config, yaml_file)
            
        print(10*'=', f'Starting Study for {device_name}', 10*'=')
        data = pd.read_table(path, sep=' ')
        # Resample data
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)        
        
        # Extract Max Features
        if config['EXTRACT_MAX_FEATURES']:
            print('INFO | Extracting Max Features ...')
            unique_sensors = get_all_sensors_in_df(df=data)
            data = extract_highest_amplitude_features_with_mp(df=data,
                                                              create_one_sensor_feature=False, n_processes=4,
                                                              keep_columns=False)
        
        # Splitting of Data
        print('INFO | Split Data X, y ...')
        feature_cols = data.columns.to_list()
        feature_cols.remove('size_mm')
        for col in config['DROP_COLUMNS']:
            feature_cols.remove(col)
        X_train, y_train = data[feature_cols], data['size_mm']
        
        print('INFO | Train-Test Split ...')
        if config['LOG_SCALE_TARGET']:
            y_train = np.log(y_train)
        
        print('INFO | Tune Model ...')
        for i in range(config['TUNING_ITER']):
            if config['DEBUG_RUN']:
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=1, n_jobs=8)
                break
            else:
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=config['N_TRIALS'], n_jobs=8)
                # Save best params
                study_name = '_'.join([config['MODEL_NAME'], device_name, str(i)])
                        
                with open(f'{config["SAVE_DIR"]}{study_name}.pkl', 'wb') as pkl_file:
                    pickle.dump(study, pkl_file)
                