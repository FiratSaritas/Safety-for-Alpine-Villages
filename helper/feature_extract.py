import pandas as pd
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")
from multiprocessing.pool import ThreadPool as Pool
from datetime import datetime


class BaseFeatureTransform(object):

    def __init__(self):
        super(BaseFeatureTransform, self).__init__()

    def transform_extracted_features(self, df: pd.DataFrame):
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
        df: pd.DataFrame
            Transformed dataframe
        """
        df = df.groupby(['start_time', 'packnr', 'sensor_type']).mean().unstack()
        df.columns = df.columns.get_level_values(0) + '_' + df.columns.get_level_values(1)
        df = df.reset_index(level=[0, 1])
        df['start_time'] = pd.to_datetime(df['start_time'])
        return df.sort_values(by='packnr', ascending=True)

    def feature_join(self, data_to_join: pd.DataFrame, data_extracted: pd.DataFrame):
        """
        Joins initial Dataframe with pre-processed data of WSL with our extracted feature Dataframe.

        Parameters
        ----------
        data_to_join: pd.DataFrame
            Preprocessed dataframe from wsl

        data_extracted: pd.DataFrame
            Extracted feature dataframe

        Returns
        -------
        data_extracted: pd.DataFrame
            Joined Dataframe
        """
        # Data Type Conversion for Join
        data_to_join['start_time'] = data_to_join['start_time'].astype(str)
        data_extracted['start_time'] = data_extracted['start_time'].astype(str)
        data_to_join['packnr'] = data_to_join['packnr'].astype(int)
        data_extracted['packnr'] = data_extracted['packnr'].astype(int)
        # Join
        data_extracted = pd.merge(left=data_to_join, right=data_extracted, how='left',
                                  left_on=['start_time', 'packnr'], right_on=['start_time', 'packnr'])
        # Type conversion back to datetime
        data_extracted['start_time'] = pd.to_datetime(data_extracted['start_time'])
        return data_extracted


class SignalFeatureExtractor(BaseFeatureTransform):
    """

    """

    def __init__(self, raw_data_path: str, extract_chromafeatures: True,
                 n_processes: int = 3):
        """

        Parameters
        ----------
        raw_data_path
        extract_chromafeatures
        n_processes

        """
        super(SignalFeatureExtractor, self).__init__()
        self.raw_data_path = raw_data_path
        self.extract_chromafeatures = extract_chromafeatures
        self.n_processes = n_processes

        print('SignalFeatureExtractor: ', self.__dict__)

    @staticmethod
    def load_raw_as_list(fp: str):
        """
        Reads the raw data file and returns it.

        Parameters
        ----------
        fp: str
            File path to raw data file

        Returns
        -------
        raw_data: list
            list of raw data measurments
        """
        raw_data = []
        with open(fp, 'r') as text_file:
            for line in text_file:
                raw_data.append(line)
        return raw_data


    def extract(self, processed_data: pd.DataFrame):
        """
        Extracts new features and joins it to the given pre-processed dataframe by WSL.

        Parameters
        ----------
        processed_data: pd.DataFrame
            Already processed Dataframe by WSL

        Returns
        -------
        data: pd.DataFrame
            Dataframe with concatenated new extracted features
        """
        raw_data = self.load_raw_as_list(fp=self.raw_data_path)

        if self.extract_chromafeatures:
            print(f'INFO || {datetime.now().strftime("%y.%m.%d_%H:%M")} | Extracting ChromaFeatures from Raw')
            data = SignalFeatureExtractor._multiprocessor_wrapper(func_=SignalFeatureExtractor.mp_extract_chroma_features,
                                                                  iterable_=raw_data,
                                                                  processors=self.n_processes)
            print(f'INFO || {datetime.now().strftime("%y.%m.%d_%H:%M")} | Transform ChromaFeatures')
            data = self.transform_extracted_features(df=data)

            print(f'INFO || {datetime.now().strftime("%y.%m.%d_%H:%M")} | Joining ChromaFeatures')
            data = self.feature_join(data_to_join=processed_data, data_extracted=data)

        return data

    def extract_with_custom_func(self, processed_data: pd.DataFrame, custom_func):
        """
        This method is to call the extraction process with a custom processing function.
        The function should take a list of strings as inputs. The string consist of one measurement sample

        Parameters
        ----------
        processed_data: pd.DataFrame
            Dataframe for data of WSl
        custom_func: func
            Pyhton function object which takes one argument(list as input) and outputs the processed sample.
            like:
            def this_is_a_function(list):
                ....
        Returns
        -------
        data:
            extracted data with the custom functin joined to the WSL dataframe.
        """
        raw_data = self.load_raw_as_list(fp=self.raw_data_path)

        print(f'INFO || {datetime.now().strftime("%y.%m.%d_%H:%M")} | Extracting ChromaFeatures from Raw')
        data = SignalFeatureExtractor._multiprocessor_wrapper(
            func_=custom_func,
            iterable_=raw_data,
            processors=self.n_processes)
        print(f'INFO || {datetime.now().strftime("%y.%m.%d_%H:%M")} | Transform ChromaFeatures')
        data = self.transform_extracted_features(df=data)

        print(f'INFO || {datetime.now().strftime("%y.%m.%d_%H:%M")} | Joining ChromaFeatures')
        data = self.feature_join(data_to_join=processed_data, data_extracted=data)

        return data

    @staticmethod
    def _multiprocessor_wrapper(func_, iterable_: list, processors=3):
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


    def mp_extract_chroma_features(sample: str):
        """
        Extracts Chromagraph features from data. This function was written to correspond with the
        multiprocessing module.

        Parameters
        ----------
        sample: list
            One Sample as List element of list of strings

        Returns
        -------
        data_dict: dict
            Returns a Dictionary with given samples
        """
        # generate List from string
        sample = sample.split(' ')

        # Feature Extraction from list of strings
        start_time = sample[0] + ' ' + sample[1]
        packnr = sample[2]
        sensor = sample[3]
        package = np.array(sample[4:]).astype(float)
        package_len = len(package)
        chroma_stft = librosa.feature.chroma_stft(y=package, sr=10000).flatten()
        chroma_stft_mean = np.mean(chroma_stft)
        chroma_stft_med = np.median(chroma_stft)
        chroma_stft_std = np.std(chroma_stft)

        # Create Dictionary
        data_dict = dict(start_time=start_time, packnr=packnr, sensor_type=sensor, len=package_len,
                         chroma_stft_mean=chroma_stft_mean, chroma_stft_med=chroma_stft_med,
                         chroma_stft_std=chroma_stft_std)
        return data_dict


# Standalone Functions below here

def extract_highest_amplitude_features_with_mp(df: pd.DataFrame, create_one_sensor_feature=True, n_processes=4,
                                               keep_columns=True, verbose=True, **kwargs) -> pd.DataFrame:
    """

    This function extracts all features per Sensor type with the maximum amplitude (mab).
    After the extraction from each feature it adds it to the given dataframe and returns it.
    
    Updates:
    Added global statement to make inner function to work more stable:
    https://stackoverflow.com/questions/52265120/python-multiprocessing-pool-attributeerror

    params:
    -------------
    df: pd.DataFrame
        Dataframe to extract max features from

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
    # extract sensor types from columns
    sensor_types = get_all_sensors_in_df(df=df)

    # List w. all columns
    all_columns = df.columns.to_list()

    for types in sensor_types:
        if not create_one_sensor_feature and len(types) == 1:
            print(f'INFO || Not Creating Max-Features for: {types}')
            continue
        print(f'INFO || Extracting Max Features for types: {types}')

        tmp = df[['mab_' + t for t in types]]
    
        # Extract maximum argument
        if verbose:
            print(f'INFO ||| Extracting Maximum Argument ')
        max_val_sensor = np.argmax(tmp.to_numpy(), axis=1)
        max_sensors_per_row = [types[s] for s in max_val_sensor]

        # Create new feature for each feature with sensors with max each
        ## Create iterable for MP
        if verbose:
            print(f'INFO ||| Creating iterable for Multiprocessing')
        max_sensors_per_row = list(zip(np.arange(len(max_sensors_per_row)), max_sensors_per_row))

        # Extract features with MP
        if verbose:
            print(f'INFO ||| Call Pool for Multiprocessing')
        def extract_max_feature_mp(max_sensor, df=df):
            """Inner func for extract_highest_amplitude_features_with_mp()"""
            max_row = df.loc[max_sensor[0], [col for col in df.columns if max_sensor[1] in col]].to_numpy()
            return max_row
        
        with Pool(processes=n_processes) as pool:
            res_mp = pool.map(func=extract_max_feature_mp, iterable=max_sensors_per_row)
        
        if verbose:
            print(f'INFO ||| Creating Dataset from processed features')
        # Create and concat extracted Features with df
        new_feat_df = pd.DataFrame.from_records(data=res_mp, columns=['max_' + '_'.join(col.split('_')[:-1]) + '_' +
                                                                      types[0][0] for col in df.columns if
                                                                      types[0] in col])
        df = pd.concat([df, new_feat_df], axis=1)

    if not keep_columns:
        if verbose:
            print(f'INFO ||| Dropping columns else than new generated max-features')
        for types in sensor_types:
            for i in range(len(types)):
                df = df.drop([col for col in df.columns if types[i] in col and 'max' not in col], axis=1)

    return df


def get_all_sensors_in_df(df: pd.DataFrame) -> [list]:
    """
    Takes Dataframe and extracts all unique sensor names and returns it as a list of list.
    This function can be used with extract_highest_amplitude_features_with_mp() as sensor_types input.
    """
    sensors = [col.split('_')[-1] for col in df.columns]
    sensors = list(set([s for s in sensors if s[-1].isnumeric()]))
    all_sensors = []
    tmp = {}
    for sensor in sensors:
        if not sensor[0] in list(tmp.keys()):
            tmp[sensor[0]] = [sensor]
        else:
            tmp[sensor[0]].append(sensor)
    for k, sensors in tmp.items():
        all_sensors.append(sensors)
    return all_sensors




def mean_feature_per_measurement(df: pd.DataFrame, drop_columns=True)  -> pd.DataFrame:
    """
    Takes mean features of each measurement from all sensors. 
    Then appends it to given Dataframe
    
    arguments:
    -----------
    df: pd.Dataframe
        pandas dataframe to extract features from
    
    returns:
    -----------
    df: pd.Dataframe
         Given Df with appended features
        
    """
    print('INFO || Extracting Mean Features')
    # Init empty df
    mean_df = pd.DataFrame()
    # Select measurement values
    columns = np.array([col[:-4] for col in df.columns])
    for col in set(columns[4:]):
        # Create mask for column selection
        column_mask =  columns == col
        column_selection = df.columns.to_numpy()[column_mask]
        # Take mean over equal measurements
        if len(column_selection) <=1:
            continue        
        mean_df[col+'_mean'] = df[column_selection].mean(axis=1)
        
    # Append to df
    if not drop_columns:
        df = pd.concat([df, mean_df], axis=1)
    else:
        df = mean_df
    
    return df


def feature_extractor_wrapper(df: pd.DataFrame, extract_max_features=True, extract_mean_features=True) -> pd.DataFrame:
    """Wrapper Function for Mean and Max Features"""
    if extract_mean_features and extract_max_features:
        mean_features = mean_feature_per_measurement(df)
        max_features = extract_highest_amplitude_features_with_mp(df=df, create_one_sensor_feature=False,
                                                                  n_processes=6, keep_columns=False, verbose=False)
        df = pd.concat([mean_features, max_features], axis=1)
    elif extract_mean_features and not extract_max_features:
        df_ = mean_feature_per_measurement(df)
        df = df[[ 'start_time', 'size_mm', 'velocity']]
        df = pd.concat([df, df_], axis=1)
    elif extract_max_features and not extract_mean_features:
        df = extract_highest_amplitude_features_with_mp(df=df, create_one_sensor_feature=False,
                                                        n_processes=6, keep_columns=False, verbose=False)
        
    return df





