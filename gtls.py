import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd

def import_network(file_name, NetworkClass):
    """ Import pickled network from file
    """
    file = open(file_name, 'br')
    data_pickle = file.read()
    file.close()
    net = NetworkClass()
    net.__dict__ = pickle.loads(data_pickle)
    return net
    
def export_network(file_name, net) -> None:
    """ Export pickled network to file
    """
    file = open(file_name, 'wb')
    file.write(pickle.dumps(net.__dict__))
    file.close()

def load_file(file_name) -> np.ndarray:
    """ Load dataset from file
    """
    reader = csv.reader(open(file_name, "r"), delimiter=',')
    x_rdr = list(reader)
    return  np.array(x_rdr).astype('float')

def normalize_data(data) -> np.ndarray:
    """ Normalize data vectors
    """
    for i in range(0, data.shape[1]):
        max_col = max(data[:, i])
        min_col = min(data[:, i])
        for j in range(0, data.shape[0]):
            data[j, i] = (data[j, i] - min_col) / (max_col - min_col)
    return data

def plot_network(net, edges, labels) -> None:
    """ 2D plot
    """        
    # Plot network
    # This just plots the first two dimensions of the weight vectors.
    # For better visualization, PCA over weight vectors must be performed.
    ccc = ['black','blue','red','green','yellow','cyan','magenta','0.75','0.15','1']
    plt.figure()
    dim_net = True if len(net.weights[0].shape) < 2 else False
    for ni in range(len(net.weights)):
        plindex = np.argmax(net.alabels[ni])
        if labels:
            if dim_net:
                plt.scatter(net.weights[ni][0], net.weights[ni][1], color=ccc[plindex], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], color=ccc[plindex], alpha=.5)
        else:
            if dim_net:
                plt.scatter(net.weights[ni][0], net.weights[ni][1], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], alpha=.5)
        if edges:
            for nj in range(len(net.weights)):
                if  net.edges[ni, nj] > 0:
                    if dim_net:
                        plt.plot([net.weights[ni][0], net.weights[nj][0]], 
                                 [net.weights[ni][1], net.weights[nj][1]],
                                 'gray', alpha=.3)
                    else:
                        plt.plot([net.weights[ni][0, 0], net.weights[nj][0, 0]], 
                                 [net.weights[ni][0, 1], net.weights[nj][0, 1]],
                                 'gray', alpha=.3)                        
    plt.show()

def plot_network_hs(net, edges, labels) -> None:
    """ 2D plot
    """
    # Define a list of colors for 21 different labels
    ccc = ['black', 'blue', 'red', 'green', 'yellow', 'cyan', 'magenta', '0.75', '0.15', '1',
           'orange', 'purple', 'brown', 'pink', 'grey', 'lime', 'indigo', 'teal', 'maroon', 'gold', 'orchid']
    
    plt.figure()
    dim_net = True if len(net.weights[0].shape) < 2 else False
    for ni in range(len(net.weights)):
        plindex = np.argmax(net.alabels[ni])
        if labels:
            if dim_net:
                plt.scatter(net.weights[ni][0], net.weights[ni][1], color=ccc[plindex], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], color=ccc[plindex], alpha=.5)
        else:
            if dim_net:
                plt.scatter(net.weights[ni][0], net.weights[ni][1], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], alpha=.5)
        if edges:
            for nj in range(len(net.weights)):
                if net.edges[ni, nj] > 0:
                    if dim_net:
                        plt.plot([net.weights[ni][0], net.weights[nj][0]],
                                 [net.weights[ni][1], net.weights[nj][1]],
                                 color=ccc[plindex], alpha=.3)
                    else:
                        plt.plot([net.weights[ni][0, 0], net.weights[nj][0, 0]],
                                 [net.weights[ni][0, 1], net.weights[nj][0, 1]],
                                 color=ccc[plindex], alpha=.3)
    plt.show()

def export_result(file_name, net, ds) -> None:
    net.bmus_index = -np.ones(net.samples)
    net.bmus_activation = np.zeros(net.samples)
    for i in range(0, net.samples):
        input = ds.vectors[i]
        b_index, b_distance = net.find_bmus(input)
        net.bmus_index[i] = b_index
        net.bmus_activation[i] = math.exp(-b_distance)
    gwr_node = [x[0] for x in net.weights]
    df_node = pd.DataFrame(gwr_node)
    weights_file_name= str(file_name+"weight.csv")
    df_node.to_csv(path_or_buf=weights_file_name, index=True)
    df = pd.DataFrame(net.bmus_index)
    #print(df)
    df.to_csv(path_or_buf=file_name, index=False, header=False)
    plt.figure(figsize=(20, 10))
    plt.xlabel('Number of frame')
    plt.ylabel('Number of cluster')
    sns.scatterplot(net.bmus_index)
    plt.show()
    
class Dataset:
    """ Create an instance dataset
    """
    def __init__(self, file, normalize):
        self.name = 'data'
        self.file = file
        self.normalize = normalize
        self.num_classes = 21
        
        raw_data = load_file(self.file)
        
        self.labels = raw_data[:, raw_data.shape[1]-1]
        self.vectors = raw_data[:, 0:raw_data.shape[1]-1]
        
        label_list = list()
        for label in self.labels:
            if label not in label_list:
                label_list.append(label)
        n_classes = len(label_list)
        
        assert self.num_classes == n_classes, "Inconsistent number of classes"
        
        if self.normalize:
            self.vectors = normalize_data(self.vectors)

# Defining a method to get the number of nodes in the GammaGWR class
def get_num_classes(self):
    return self.num_classes