import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_packages(fp: str, read_packages: int, take_random_sample=True, 
                  sample_size=10):
    """
    Reads Measurement file and plots given amount of samples from it.
    
    params:
    ---------
    fp: str
        Filepath to the raw data file.
    
    read_packages: int
        Amount of packages which should be read
    
    sample_from_packages: int
        How many samples should be drawn from the read measurements.
        
    returns:
    ----------
    plt.figure
    
    """
    with open(fp, 'r') as file:
        samples = []
        for line in file:
            if len(samples) > read_packages:
                break
            samples.append(line) 
    
    if take_random_sample:
        samples = np.random.permutation(samples)[:sample_size]
    else:
        samples = samples[:sample_size]
    samples = [sample.split(' ') for sample in samples]
    ## Start time
    start_time = [sample[0] + ' ' + sample[1] for sample in samples]
    ## Test number
    test_no = [sample[2] for sample in samples]
    ## Sensor type
    sensor = [sample[3] for sample in samples]
    ## Measurements
    package = [np.array(sample[4:]).astype(float) for sample in samples]
    
    fig = plt.subplots(figsize=(18, int(4*len(samples)/5)))

    ncol = 5
    nrow = int(len(samples) / ncol) + 1

    for i in range(len(samples)):
        plt.subplot(nrow, ncol, i+1)
        p = sns.lineplot(x=np.arange(len(package[i])), y=package[i], label=start_time[i])
        p.set_title(f'Sensor: {sensor[i]} / Measurement: {test_no[i]}', fontsize=10)
        p.set_xlabel('Steps')
        if i % ncol == 0:
            p.set_ylabel('Amplitude')

    plt.subplots_adjust(hspace=.4)
    plt.show()