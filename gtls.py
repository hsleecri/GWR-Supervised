import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
import os
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import adjusted_rand_score

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

def export_network_to_local(file_name, net, export_directory) -> None:
    """ Export pickled network to file in local environment
    """ 
    # Generate a directory if the file path doesn't exist
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)

    # Combine the file's name and path
    full_path = os.path.join(export_directory, file_name)
    
    with open(full_path, 'wb') as file:
        file.write(pickle.dumps(net.__dict__))
    print(f"Network saved to {full_path}")


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

def plot_network_multi_labels(net, edges, labels) -> None:
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

def generate_colors(n):
    """Generate a list of distinct colors.

    Args:
        n: Number of distinct colors to generate.

    Returns:
        A list of RGB tuples representing distinct colors.
    """
    return plt.cm.jet(np.linspace(0, 1, n))


def plot_network_with_pca(net, edges, labels) -> None:
    """ 2D plot after applying PCA to net.weights
    """
    # Apply PCA to net.weights and get the first two principal components
    weights = np.array([w[0] if len(w.shape) < 2 else w[0, :] for w in net.weights])
    pca = PCA(n_components=2)
    transformed_weights = pca.fit_transform(weights)

    # Generate a color palette with as many colors as there are nodes
    ccc = generate_colors(len(transformed_weights))

    plt.figure()
    for ni in range(len(transformed_weights)):
        plindex = np.argmax(net.alabels[ni])
        x, y = transformed_weights[ni][0], transformed_weights[ni][1]  # Extracting the coordinates
        if labels:
            plt.scatter(x, y, color=ccc[ni], alpha=.5)  # Using a unique color for each node
        else:
            plt.scatter(x, y, alpha=.5)
        
        # Adding the cluster index as a label near the point
        plt.text(x, y, str(ni), fontsize=9)
        
        if edges:
            for nj in range(len(transformed_weights)):
                if net.edges[ni, nj] > 0:
                    plt.plot([transformed_weights[ni][0], transformed_weights[nj][0]], 
                             [transformed_weights[ni][1], transformed_weights[nj][1]],
                             'gray', alpha=.3)
                    
    # Adding labels for the axes
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

def show_result(file_name, net, ds) -> None:
    """Saves the network weights to a CSV file and plots the BMUs.
    """
    # Initialize BMUs index and activation
    net.bmus_index = -np.ones(net.samples)
    net.bmus_activation = np.zeros(net.samples)

    # Calculate BMUs index and activation
    for i in range(net.samples):
        input_vector = ds.vectors[i]
        b_index, b_distance = net.find_bmus(input_vector)
        net.bmus_index[i] = b_index
        net.bmus_activation[i] = math.exp(-b_distance)
    '''
    # Save weights to CSV
    gwr_node_weights = [x[0] for x in net.weights]
    df_node = pd.DataFrame(gwr_node_weights)
    weights_file_name = f"{file_name}weight.csv"
    df_node.to_csv(path_or_buf=weights_file_name, index=True)

    # Save BMUs index to CSV
    df_bmus = pd.DataFrame(net.bmus_index)
    df_bmus.to_csv(path_or_buf=file_name, index=False, header=False)
    '''
    # Generate a color palette with as many colors as there are nodes
    number_of_nodes = len(net.weights)
    ccc = generate_colors(number_of_nodes)

    # Create a mapping from cluster index to color
    cluster_colors = [ccc[int(index)] for index in net.bmus_index]

    # Plot BMUs index with the corresponding colors
    plt.figure(figsize=(20, 10))
    plt.title('Cluster Indices by Model Across Frames')  # Add title
    plt.xlabel('Number of frames')
    plt.ylabel('Cluster Index')
    plt.scatter(x=range(len(net.bmus_index)), y=net.bmus_index, c=cluster_colors)
    plt.show()

def show_original_clusters(ds) -> None:
    """Plots the original cluster indices from the dataset across frames.
    """
    # Generate a color palette with as many colors as there are classes
    number_of_classes = ds.num_classes
    ccc = generate_colors(number_of_classes)

    # Create a mapping from cluster index to color
    cluster_colors = [ccc[int(index)] for index in ds.labels]

    # Plot original cluster indices with the corresponding colors
    plt.figure(figsize=(20, 10))
    plt.title('Original Cluster Indices Across Frames')  # Add title
    plt.xlabel('Number of frames')
    plt.ylabel('Cluster Index')
    plt.scatter(x=range(len(ds.labels)), y=ds.labels, c=cluster_colors)

    # Set y-axis tick locations to integers
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

def show_result_segmentation(net, ds) -> None:
    """Plots the cluster indices by the model across frames as colored rectangles.
    """
    # Initialize BMUs index and activation
    net.bmus_index = -np.ones(net.samples)
    net.bmus_activation = np.zeros(net.samples)

    # Calculate BMUs index and activation
    for i in range(net.samples):
        input_vector = ds.vectors[i]
        b_index, b_distance = net.find_bmus(input_vector)
        net.bmus_index[i] = b_index
        net.bmus_activation[i] = math.exp(-b_distance)

    # Generate a color palette with as many colors as there are nodes
    number_of_nodes = len(net.weights)
    ccc = generate_colors(number_of_nodes)

    # Plot colored rectangles for each cluster index
    plt.figure(figsize=(10, 2))
    plt.title('Cluster Indices by Model Across Frames')

    plt.ylim(0, 1)  # Set y-axis limits to create a single line effect
    for i in range(len(net.bmus_index)):
        color = ccc[int(net.bmus_index[i])]
        plt.axvspan(i, i + 1, facecolor=color, alpha=1)  # Draw a colored rectangle for each frame

    plt.xticks([])  # Hide x-axis ticks
    plt.yticks([])  # Hide y-axis ticks
    plt.show()

def show_original_clusters_segmentation(ds) -> None:
    """Plots the original cluster indices from the dataset across frames as colored rectangles.
    """
    # Generate a color palette with as many colors as there are classes
    number_of_classes = ds.num_classes
    ccc = generate_colors(number_of_classes)

    # Plot colored rectangles for each original cluster index
    plt.figure(figsize=(10, 2))
    plt.title('Original Cluster Indices Across Frames')
    plt.xlabel('Number of frames')
    plt.ylim(0, 1)  # Set y-axis limits to create a single line effect
    for i in range(len(ds.labels)):
        color = ccc[int(ds.labels[i])]
        plt.axvspan(i, i + 1, facecolor=color, alpha=1)  # Draw a colored rectangle for each frame

    plt.xticks([])  # Hide x-axis ticks
    plt.yticks([])  # Hide y-axis ticks
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