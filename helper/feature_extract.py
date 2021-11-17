import pandas as pd
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")
from multiprocess import Pool

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


def transform_extracted_features(df: pd.DataFrame):
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


def extract_highest_amplitude_features_with_mp(df: pd.DataFrame, sensor_types: list, 
                                               create_one_sensor_feature=True, n_processes=4,
                                               keep_columns=True) -> pd.DataFrame:
    """
    
    This function extracts all features per Sensor type with the maximum amplitude (mab).
    After the extraction from each feature it adds it to the given dataframe and returns it.
    
    params:
    -------------
    df: pd.DataFrame
        Dataframe to extract max features from
    
    sensor_types:
        list of list. i.e.  [['G01', 'G02'], ['M01'], ['S01']]
    
    create_one_sensor_feature: Bool
        Only applicable if there are sensor types with a single sensor. It makes no sense
        to take max values if only one sensor is in data.
            
    n_processes: int
        Number of concurrent processes should be used
        
    returns:
    -------------
    df: pd.DataFrame
        Dataframe with concatenated max feature Values at last column position for each sensor type.
    
    """
    # List w. all columns
    all_columns = df.columns.to_list()
    
    for types in sensor_types:
        if not create_one_sensor_feature and len(types) == 1:
            print(f'INFO || Not Creating Max-Features for: {types}')
            continue
        print(f'INFO || Extracting Max Features for types: {types}')
            
        tmp = df[['mab_'+t for t in types]]
        
        # Extract maximum argument
        max_val_sensor = np.argmax(tmp.to_numpy(), axis=1) 
        max_sensors_per_row = [types[s] for s in max_val_sensor]

        # Create new feature for each feature with sensors with max each
        ## Create iterable for MP
        max_sensors_per_row = list(zip(np.arange(len(max_sensors_per_row)), max_sensors_per_row))
        
        ## Define inner func for MP
        def extract_max_feature_mp(max_sensor, df=df):
            max_row = df.loc[max_sensor[0], [col for col in df.columns if max_sensor[1] in col]].to_numpy()
            return max_row
        
        # Extract features with MP
        with Pool(processes=n_processes) as pool:
            res_mp = pool.map(func=extract_max_feature_mp, iterable=max_sensors_per_row)
        
        # Create and concat extracted Features with df
        new_feat_df = pd.DataFrame.from_records(data=res_mp, columns=['max_' + '_'.join(col.split('_')[:-1])+ '_' +
                                                    types[0][0] for col in df.columns if types[0] in col])
        df = pd.concat([df, new_feat_df], axis=1)
    
    if not keep_columns:
        for types in sensor_types:
            for i in range(len(types)):
                df = df.drop([col for col in df.columns if types[i] in col and 'max' not in col], axis=1)
    
    return df




