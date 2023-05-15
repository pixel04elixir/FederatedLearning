import os
import random

import cv2
import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from sklearn.metrics import accuracy_score


def load(paths, verbose=False):
    """
    Returns data, labels by iterating image paths
    """
    data, labels = [], []

    for i, path in enumerate(paths):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = np.array(image).flatten()

        label = path.split(os.path.sep)[-2]
        labels.append(label)

        # Normalize image data between 0 and 1
        data.append(image / 255)

        if verbose and i > 0 and (i + 1) % 10000 == 0:
            print("Processed {}/{}".format(i + 1, len(paths)))

    return data, labels


def create_clients(img_list, lbl_list, num_clients=10, namespace="clients") -> dict:
    """
    args:
        - ``img_list``: list of training images
        - ``lbl_list``: list of binarized labels
        - ``num_clients``: number of clients
        - ``namespace``: name prefix for each client node, e.g, clients_1

    return: clients with data shards
    """

    # create specified number of clients
    clients = [f"{namespace}_{i+1}" for i in range(num_clients)]

    data = list(zip(img_list, lbl_list))
    random.shuffle(data)

    # create data shards for each client
    shard = len(data) // num_clients
    client_data = [data[i : i + shard] for i in range(0, shard * num_clients, shard)]

    assert len(client_data) == num_clients
    return {clients[i]: client_data[i] for i in range(len(clients))}


def client_batch_data(data_shard, batch_size=32):
    """
    Create a tf dataset object from data shard
    """
    # seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(batch_size)


def poison_data(data_shard, batch_size=32):
    """
    Create a tf dataset object from data shard with targeted label flipping
    """
    # Separate shard into data and labels lists
    data, label = zip(*data_shard)
    label = np.array(label)

    # Perform label flipping on a fraction of the samples
    for i, x in enumerate(label):
        if x[6] == 1:
            label[i][6] = 0
            label[i][2] = 1
        if x[8] == 1:
            label[i][8] = 0
            label[i][4] = 1
        if x[9] == 1:
            label[i][9] = 0
            label[i][3] = 1

    # Convert the data and labels lists into TensorFlow tensors
    data = tf.convert_to_tensor(list(data), dtype=tf.float32)
    label = tf.convert_to_tensor(list(label), dtype=tf.int32)

    # Create a TensorFlow Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    return dataset.shuffle(len(label)).batch(batch_size)


class MLP:
    @staticmethod
    def build_model(shape, classes):
        return Sequential(
            [
                layers.Dense(200, input_shape=(shape,)),
                layers.Activation("relu"),
                layers.Dense(200),
                layers.Activation("relu"),
                layers.Dense(classes),
                layers.Activation("softmax"),
            ]
        )


def local_scaling_factor(client_data, client_name):
    """
    Returns the ratio of local to global data size
    """
    client_names = list(client_data.keys())
    model_counts = [
        tf.data.experimental.cardinality(client_data[client]).numpy()
        for client in client_names
    ]
    global_count = sum(model_counts)

    local_count = tf.data.experimental.cardinality(client_data[client_name]).numpy()
    return local_count / global_count


def scale_local_weights(weight, scalar):
    """
    Returns scaled model weights for each client
    """
    return [scalar * w for w in weight]


def sum_scaled_weights(scaled_wts):
    """
    Return the sum of the scaled weights
    """
    return [tf.math.reduce_sum(weight, axis=0) for weight in zip(*scaled_wts)]


def test_model(X_test, Y_test, model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    logits = model.predict(X_test, batch_size=100)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print(f"comm_round: {comm_round} | global_acc: {acc:.3%} | global_loss: {loss}")
    return acc, loss


def euclid_dist(point1, point2):
    """
    Computes the Euclidean distance between two points in 2D space.
    """
    # Calculate the difference between the weights and biases of the two models
    diff = [np.subtract(point1[0][i], point2[0][i]) for i in range(len(point1[0]))]

    # Return Euclidean distance between the two models
    sq_diff = [np.square(d) for d in diff]
    return np.sqrt(np.sum(sq_diff))


def make_edges(coord, threshold):
    """
    Creates edges between coordinates if the Euclidean distance between them is less than the threshold.
    """
    edges = set()
    length = len(coord)
    for i in range(length):
        for j in range(i + 1, length):
            if euclid_dist(coord[i], coord[j]) < threshold:
                edges.add(frozenset({i, j}))
    return edges


def get_max_clique(graph):
    """
    Returns the maximum clique in the graph using bron-kerbosch algorithm.
    """
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
