import numpy as np
import random
import cv2
import os
import networkx as nx
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K

from fl_mnist_implementation_tutorial_utils import *

# declear path to your mnist data folder
img_path = './archive/trainingSet/trainingSet'

# get the path list using the path object
image_paths = list(paths.list_images(img_path))
# apply our function
image_list, label_list = load(image_paths, verbose=10000)

# binarize the labels
lb = preprocessing.LabelBinarizer()
label_list = lb.fit_transform(label_list)

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    image_list, label_list, test_size=0.1, random_state=42)

# create clients
clients = create_clients(X_train, y_train, num_clients=10, initial='client')

# process and batch the training data for each client
clients_batched = dict()
poisioned_batched = dict()
for (client_name, data) in clients.items():
    bd = batch_data(data)
    pd = poision_data(data)
    clients_batched[client_name] = bd
    k = random.randint(0, 1)
    if k == 1:
        poisioned_batched[client_name] = pd
    else:
        poisioned_batched[client_name] = bd

# process and batch the test set
test_batched = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)).batch(len(y_test))

comms_round = 100

# create optimizer
lr = 0.01
loss = 'categorical_crossentropy'
metrics = ['accuracy']
optimizer = tf.keras.optimizers.legacy.SGD(
    lr=lr, momentum=0.9, decay=lr / comms_round)

# initialize global model
smlp_global = SimpleMLP()
global_model = smlp_global.build(784, 10)

# commence global training loop
for comm_round in range(comms_round):
    graph = set()
    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()

    # initial list to collect local model weights after scalling
    scaled_local_weight_list = list()

    # randomize client data - using keys
    # client_names= list(clients_batched.keys())
    client_names = list(poisioned_batched.keys())
    random.shuffle(client_names)

    # loop through each client and create new local model
    for client in client_names:
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(784, 10)
        local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # set local model weight to the weight of the global model
        local_model.set_weights(global_weights)

        # fit local model with client's data
        # local_model.fit(clients_batched[client], epochs=1, verbose=0)
        local_model.fit(poisioned_batched[client], epochs=1, verbose=0)

        # scale the model weights and add to list
        # scaling_factor = weight_scalling_factor(clients_batched, client)
        scaling_factor = weight_scalling_factor(poisioned_batched, client)

        scaled_weights = scale_model_weights(
            local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)

        # clear session to free memory after each communication round
        K.clear_session()

    edges = make_edges(scaled_local_weight_list, 0.2)
    graph = {}
    vertices = set()

    for u, v in edges:
        if u not in graph:
            graph[u] = set()
        if v not in graph:
            graph[v] = set()
        graph[u].add(v)
        graph[v].add(u)
        vertices.add(u)
        vertices.add(v)
    print(graph)
    max_clique = bron_kerbosch(nx.Graph(graph))
    print(max_clique)
    if len(max_clique) <= 5:
        exit()

    modified_scaled_local_weight_list = []
    for x in max_clique:
        modified_scaled_local_weight_list.append(scaled_local_weight_list[x])
    # to get the average over all the local model, we simply take the sum of the scaled weights
    # average_weights = sum_scaled_weights(scaled_local_weight_list)
    average_weights = sum_scaled_weights(modified_scaled_local_weight_list)

    # update global model
    global_model.set_weights(average_weights)

    # test global model and print out metrics after each communications round
    for (X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(
            X_test, Y_test, global_model, comm_round)
        SGD_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)).shuffle(len(y_train)).batch(320)
smlp_SGD = SimpleMLP()
SGD_model = smlp_SGD.build(784, 10)

SGD_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# fit the SGD training data to model
_ = SGD_model.fit(SGD_dataset, epochs=100, verbose=0)

# test the SGD global model and print out metrics
for (X_test, Y_test) in test_batched:
    SGD_acc, SGD_loss = test_model(X_test, Y_test, SGD_model, 1)
