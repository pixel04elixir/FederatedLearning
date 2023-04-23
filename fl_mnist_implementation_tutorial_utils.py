
from turtle import distance
import numpy as np
import math
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K



def load(paths, verbose=-1):
    '''expects images for each class in seperate dir, 
    e.g all digits in 0 class in the directory named 0 '''
    data = list()
    labels = list()
    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten()
        label = imgpath.split(os.path.sep)[-2]
        # scale the image to [0, 1] and add to list
        data.append(image/255)
        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    # return a tuple of the data and labels
    return data, labels


def create_clients(image_list, label_list, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''

    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))}



def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

import numpy as np
import tensorflow as tf


def poision_data(data_shard, bs=32, flip_fraction=0.001):
    '''Takes in a client's data shard, create a tfds object off it,
    and perform label flipping on a fraction of the samples.
    Args:
        data_shard: a list of tuples (data, label) constituting a client's data shard
        bs: batch size
        flip_fraction: the fraction of samples to flip their labels
    Returns:
        tfds object'''
    # Separate shard into data and labels lists
    data, label = zip(*data_shard)
    label = np.array(label)
    
    # Perform label flipping on a fraction of the samples
    flip_indices = np.random.choice(len(label), size=int(len(label) * flip_fraction), replace=False)
    label[flip_indices] = 9 - label[flip_indices]  # Flip the labels of the selected samples
    # Convert the data and labels lists into TensorFlow tensors
    data = tf.convert_to_tensor(list(data), dtype=tf.float32)
    label = tf.convert_to_tensor(label, dtype=tf.int32)
    
    # Create a TensorFlow Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    return dataset.shuffle(len(label)).batch(bs)


class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
    

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad


def test_model(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss

def euclidean_distance(point1, point2):
    """
    Computes the Euclidean distance between two points in 2D space.
    """
    # Calculate the difference between the weights and biases of the two models
    diff = [np.subtract(point1[0][i], point2[0][i]) for i in range(len(point1[0]))]
    
    # Calculate the sum of squares of the differences
    squared_diff = [np.square(d) for d in diff]
    sum_squared_diff = np.sum(squared_diff)
    
    # Take the square root of the sum of squares to get the Euclidean distance
    distance = np.sqrt(sum_squared_diff)
    return distance
    # distance_sq=0
    # for i in range(len(point1)):
    #     p=0
    #     # distance_sq+=(point1[i]-point2[i])**2
    # return math.sqrt(distance_sq)

def make_edges(coordinates, threshold):
    """
    Creates edges between coordinates if the Euclidean distance between them is less than the threshold.
    """
    edges = set()
    length=len(coordinates)
    for i in range(length):
        for j in range(i+1, length):
            if euclidean_distance(coordinates[i], coordinates[j]) < threshold:
                edges.add(frozenset({i, j}))
    return edges

def bron_kerbosch(graph):
    R = set()
    P = set(graph.nodes())
    X = set()
    max_clique = set()
    
    def expand(R, P, X):
        nonlocal max_clique
        
        if not P and not X:
            if len(R) > len(max_clique):
                max_clique = set(R)
            return
        
        for v in list(P):
            expand(R | {v}, P & set(graph.neighbors(v)), X & set(graph.neighbors(v)))
            P.remove(v)
            X.add(v)
    
    expand(R, P, X)
    return max_clique
