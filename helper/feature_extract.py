import pandas as pd
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool


def multiprocessor_wrapper(func_, iterable_: list, processors=3):
    """
    Wrapper function for data reading from vlnd to have access to multiple processes on
    computer. Processing it on multiple processes is much faster than processing it on only one.

    Parameters
    ----------
    func_: function-object
        Function used for multiprocessing. SHould be constructed to process single element like
        one list element

    iterable_: list
        List of elements to map to the func_.

    processors: int
        Number of processes to use.

    Returns
    -------
    res: pd.DataFrame
        Data converted to a pandas.DataFrame
    """

    # Process data
    with Pool(processes=processors) as pool:
        res = pool.map(func=func_, iterable=iterable_)

    return pd.DataFrame.from_records(res)


def transform_feature(df: pd.DataFrame):
    """
    Transforms given feature Dataframe to consistent shape for given Data by WSL.

    This Dataframe needs the columns: start_time, test_no, sensor_type

    start_time	        | test_no	| sensor_type	|   Features...
    2021-06-22 15:34:38	|    1	    |   S01	        |
    2021-06-22 15:34:38	|    1	    |   M01	        |
    2021-06-22 15:34:38	|    1	    |   G02	        |

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with all extracted feature records init

    Returns
    -------

    """
    df = df.groupby(['start_time', 'test_no', 'sensor_type']).mean().unstack()
    df.columns = df.columns.get_level_values(0) + '_' + df.columns.get_level_values(1)
    df = df.reset_index(level=[0, 1])

    return df


def mp_extract_chroma_features(sample: str):
    """
    Extracts Chromagraph features from data. This function was written to correspond with the
    multiprocessing module.

    Parameters
    ----------
    sample: list
        One Sample as Listelement

    Returns
    -------
    data_dict: dict
        Returns a Dictionary with given samples
    """
    sample = sample.split(' ')

    # Feature Extraction
    ## Start time
    start_time = sample[0] + ' ' + sample[1]
    ## Test number
    test_no = sample[2]
    ## Sensor type
    sensor = sample[3]
    ## Measurements
    package = np.array(sample[4:]).astype(float)
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

    return data_dict



