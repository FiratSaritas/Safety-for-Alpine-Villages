import pandas as pd
import pandas_profiling
import os
import re

print('Start Process...')

# Extract text file data from data dir
data_files = os.listdir('../data')
data_files = [f for f in data_files if '.txt' in f]

# Make new dir report if not exists
if 'reports' not in os.listdir():
    os.mkdir('../eda/reports')

for file in data_files:
    # Read dataset
    df = pd.read_table(f'../data/{file}', sep=' ')
    # Generate Report
    title = file[5:8]," PD Report"
    title = ''.join(title)
    profile = df.profile_report(title=title,
                                pool_size=4,
                                missing_diagrams=dict(heatmap=True),
                                correlations=dict(pearson=dict(calculate=True)))
    # Extract measuring device name
    dataset_name = re.findall(pattern='_(\w+).', string=file)[0]
    profile.to_file(f'./reports/{dataset_name}_report.html')