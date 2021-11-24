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
    
    
def plot_error_per_cat(y_true, y_pred, show_strip=True, show_outliers=False):
    """
    Plots error by using Boxplots. 
    Predicitons where automatically binned on True values.
    
    params:
    ----------
    y_true: np.ndarray
        True values    
    y_pred: np.ndarray
        predicted values
    show_strip: Bool
        show stripplot if true else not
    show_outlier: Bool
        shows outlier of boxplot in black if true else not.
        
    returns:
    ----------
    plt.plot
    
    """
    fig = plt.subplots(figsize=(10, 5))
    sns.set_palette('Greens', 15)
    sns.set_style('darkgrid')
    p = sns.boxplot(y=y_pred, x=y_test, showfliers=show_outliers)
    if show_strip:
        p2 = sns.stripplot(y=y_pred, x=y_test, alpha=.5, color='grey', size=3)
    p.set_xlabel('True')
    p.set_ylabel('Predicted')
    p.set_title('True vs. Predicted per Category', loc='left')
    plt.show()
    

def plot_predictions_to_nearest_class_heatmap(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Plots a Confusion Matrix of the predictions mapped to the nearest Grainsize Class.
    The actual Regression Problem was abstracted to a classification problem to see which 
    values were mixed up a lot.
    
    params:
    ---------
    y_true: np.ndarray
        Array of True labels
    
    y_pred: np.ndarray
        Array of predicted labels
    
    returns:
    -------------
    plt.figure
    
    """
    y_true_cluster = y_true.astype('float32').reshape(-1,1)
    y_pred_cluster = y_pred.astype('float32').reshape(-1,1)
    
    cluster = KMeans(n_clusters=len(np.unique(y_true_cluster)))
    cluster.fit(y_true_cluster)
    
    y_true_cluster = cluster.predict(y_true_cluster)
    y_pred_cluster = cluster.predict(y_pred_cluster)
    
    fig = plt.subplots(figsize=(10, 5))
    
    p = sns.heatmap(confusion_matrix(y_true=y_test_cluster, y_pred=y_pred_cluster), 
                    cmap='Reds',linewidths=.1, linecolor='black', annot=True, fmt='2d')
    p.set_xlabel('Predicted')
    p.set_ylabel('True')
    p.set_xticklabels(np.unique(y_test), rotation=30)
    p.set_yticklabels(np.unique(y_test), rotation=30)
    plt.title('Predictions mapped to nearest Grainsize Class with KMeans. KMeans Centers defined as Grainsize-classes of True Values', fontsize=8)
    plt.suptitle('Confusion Matrix')
    plt.show()
    
def plot_residuals(y_pred, y_test):
    """"""
    fig = plt.subplots(figsize=(10, 5))
    p = sns.residplot(x=y_pred, y=y_test, color='grey')
    p.set_title('Residualplot', loc='left')
    p.set_xlabel('Predicted size_mm')
    p.set_ylabel('Residuals')
    plt.show()
    

def plot_kde(y_test, y_pred):
    fig = plt.subplots(figsize=(10, 5))
    sns.set_palette('Paired', 2)
    p = sns.kdeplot(y_test, cumulative=True, bw=.01, label='True', linestyle='--')
    p = sns.kdeplot(y_pred, cumulative=True, bw=.01, label='Predicted', color='grey')
    p.set_title('Cumulative Density Plot Predicted and True', loc='left')
    plt.xlim(0, np.max(y_pred))
    plt.legend()
    plt.show()