import pandas as pd
import librosa
from tqdm import tqdm
import numpy as np


def transform_feature_extract_data(df: pd.DataFrame):
    """
    Transforms Dataframe after Feature Extraction of Raw Data to match the existing data.

    Parameters
    ----------
    df: pd.Dataframe
        Dataframe to transfrom

    Returns
    -------
    df: pd.Dataframe
        Transformed Dataframe

    """
    df = df.groupby(['start_time', 'test_no', 'sensor_type']).mean().unstack()
    df.columns = df.columns.get_level_values(0) + '_' + df.columns.get_level_values(1)
    df = df.reset_index(level=[0, 1])

    return df


def extract_chroma_features(fp: str):
    """
    Extracts Chromagram features from given waveform within file.

    Derived features from Chromagram are: mean, median and std.

    Parameters
    ----------
    fp: str
        Filepath where the .vlnd raw data packets are stored.

    Returns
    -------
    df: pd.Dataframe
        Pandas Dataframe with derived features.
    """
    df = pd.DataFrame()
    samples = []
    with open(fp, 'r') as text_file:
        for line in text_file:
            samples.append(line)

    for line in tqdm(samples, desc='Extracting Features'):
        # line = text_file.readlines(1)
        line = line.split(' ')

        # Feature Extraction
        ## Start time
        start_time = line[0] + ' ' + line[1]
        ## Test number
        test_no = line[2]
        ## Sensor type
        sensor = line[3]
        ## Measurements
        package = np.array(line[4:]).astype(float)
        ## Length of a package
        package_len = len(package)
        # Feature Extraction Chroma stft
        chroma_stft = librosa.feature.chroma_stft(y=package, sr=10000).flatten()
        chroma_stft_mean = np.mean(chroma_stft)
        chroma_stft_med = np.median(chroma_stft)
        chroma_stft_std = np.std(chroma_stft)

        # Feature Extraction
        data_dict = dict(start_time=start_time, test_no=test_no, sensor_type=sensor, package_len=package_len,
                         chroma_stft_mean=chroma_stft_mean, chroma_stft_med=chroma_stft_med,
                         chroma_stft_std=chroma_stft_std)
        df_ = pd.DataFrame(data=data_dict, index=list(data_dict.keys()))
        df_ = df_.reset_index(drop=True)

        df = df.append(df_)

    df = df.reset_index(drop=True)

    return transform_feature_extract_data(df=df)