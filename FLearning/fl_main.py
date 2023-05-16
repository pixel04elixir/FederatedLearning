import tensorflow as tf
from imutils import paths
from keras import backend as K
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from FLearning.utils import *


class FLBase:
    def __init__(
        self,
        img_path,
        lr=0.01,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        momentum=0.9,
        optimizer=tf.keras.optimizers.legacy.SGD,
    ):
        self.path = img_path
        self.img_paths = list(paths.list_images(self.path))
        self.img_list, self.lbl_list = load(self.img_paths, verbose=True)
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer(learning_rate=lr, momentum=momentum)

        # Binarize labels
        lb = preprocessing.LabelBinarizer()
        self.lbl_list = lb.fit_transform(self.lbl_list)

    def _get_split(self):
        # split data into training and test set
        return train_test_split(
            self.img_list, self.lbl_list, test_size=0.1, random_state=42
        )

    def _create_clients(self, count=10, namespace="client"):
        return create_clients(
            self.x_train, self.y_train, num_clients=count, namespace=namespace
        )

    def _get_client_data(self):
        return {
            client: client_batch_data(data) for (client, data) in self.clients.items()
        }

    def initialize(self, clients=10):
        self.x_train, self.x_test, self.y_train, self.y_test = self._get_split()
        self.clients = self._create_clients(count=clients)

        # Create data batch
        self.client_batch = self._get_client_data()

        self.test_batch = tf.data.Dataset.from_tensor_slices(
            (self.x_test, self.y_test)
        ).batch(len(self.y_test))

        global_model = MLP()
        self.global_model = global_model.build_model(784, 10)

    def train(self, rounds=100, test=True):
        # Iteratively update global model
        result = []
        for round in range(rounds):
            global_wt = self.global_model.get_weights()
            local_wt = {}

            for client in self.client_batch.keys():
                mlp_local = MLP()
                local_model = mlp_local.build_model(784, 10)
                local_model.compile(
                    optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
                )

                # train local model with global model's weights
                local_model.set_weights(global_wt)
                local_model.fit(self.client_batch[client], epochs=1, verbose=0)

                # scale the local model weights
                local_wt[client] = local_model.get_weights()

                # scaled_local_wt.append(scaled_wt)
                K.clear_session()

            # Post-cleaning step
            filter_keys = self.fl_cleanup(local_wt)
            clean_batch = {
                client: value
                for client, value in self.client_batch.items()
                if client in filter_keys
            }

            scaled_local_wt = [
                scale_local_weights(local_wt[i], local_scaling_factor(clean_batch, i))
                for i in clean_batch.keys()
            ]

            average_weights = sum_scaled_weights(scaled_local_wt)

            # update global model
            self.global_model.set_weights(average_weights)

            if test:
                for X_test, Y_test in self.test_batch:
                    gacc, _ = test_model(X_test, Y_test, self.global_model, round)
                    result.append(gacc * 100)

        return result

    def evaluate(self):
        for X_test, Y_test in self.test_batch:
            global_acc, global_loss = test_model(X_test, Y_test, self.global_model, 0)
        return global_acc, global_loss

    def save(self, save_file="fl_model.h5"):
        self.global_model.save(save_file)

    def fl_cleanup(self, local_wt):
        return list(local_wt.keys())
