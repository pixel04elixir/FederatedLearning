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
        rounds=100,
        lr=0.01,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        momentum=0.9,
        optimizer=tf.keras.optimizers.legacy.SGD,
    ):
        self.path = img_path
        self.img_paths = list(paths.list_images(self.path))
        self.img_list, self.lbl_list = load(self.img_paths, verbose=10000)
        self.rounds = rounds
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer(
            learning_rate=lr, momentum=momentum, decay=lr / self.rounds
        )

        # Binarize labels
        lb = preprocessing.LabelBinarizer()
        self.lbl_list = lb.fit_transform(self.lbl_list)

    def _get_split(self):
        # split data into training and test set
        return train_test_split(
            self.img_list, self.lbl_list, test_size=0.1, random_state=42
        )

    def _get_clients(self, count=10, namespace="client"):
        return create_clients(
            self.x_train, self.y_train, num_clients=count, namespace=namespace
        )

    def _get_client_data(self):
        return {client: batch_data(data) for (client, data) in self.clients.items()}

    def initialize(self):
        self.x_train, self.x_test, self.y_train, self.y_test = self._get_split()
        self.clients = self._get_clients()

        # Create data batch
        self.client_batch = self._get_client_data()

        self.test_batch = tf.data.Dataset.from_tensor_slices(
            (self.x_test, self.y_test)
        ).batch(len(self.y_test))

        smlp_global = SimpleMLP()
        self.global_model = smlp_global.build(784, 10)

    def train(self, debug=True):
        # Global training loop
        for comm_round in range(self.rounds):
            global_weights = self.global_model.get_weights()
            scaled_local_wt = []

            for client in self.client_batch.keys():
                smlp_local = SimpleMLP()
                local_model = smlp_local.build(784, 10)
                local_model.compile(
                    loss=self.loss, optimizer=self.optimizer, metrics=self.metrics
                )

                # train local model with global model's weights
                local_model.set_weights(global_weights)
                local_model.fit(self.client_batch[client], epochs=1, verbose=0)

                # scale the model weights and add to list
                scaling_factor = weight_scalling_factor(self.client_batch, client)

                scaled_wt = scale_model_weights(
                    local_model.get_weights(), scaling_factor
                )
                scaled_local_wt.append(scaled_wt)
                K.clear_session()

            # Post-cleaning step
            scaled_local_wt = self.fl_cleanup(scaled_local_wt)
            average_weights = sum_scaled_weights(scaled_local_wt)

            # update global model
            self.global_model.set_weights(average_weights)

            if debug:
                for X_test, Y_test in self.test_batch:
                    global_acc, global_loss = test_model(
                        X_test, Y_test, self.global_model, comm_round
                    )

    def evaluate(self):
        for X_test, Y_test in self.test_batch:
            global_acc, global_loss = test_model(X_test, Y_test, self.global_model, 0)
        return global_acc, global_loss

    def save(self, save_file="fl_model.h5"):
        self.global_model.save(save_file)

    def fl_cleanup(self, scaled_local_wt):
        return scaled_local_wt
