import tensorflow as tf

from FLearning.fl_main import FLBase
from FLearning.utils import *


class FLPoison(FLBase):
    def __init__(
        self,
        img_path,
        lr=0.01,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        momentum=0.9,
        optimizer=tf.keras.optimizers.legacy.SGD,
    ):
        super().__init__(img_path, lr, loss, metrics, momentum, optimizer)

    def initialize(self, clients=10, poison_clients=0.1):
        self.poison_clients = poison_clients
        super().initialize(clients=clients)

    def _get_client_data(self):
        assert self.poison_clients <= len(self.clients)
        poison_clients = random.sample(
            self.clients.keys(), int(self.poison_clients * len(self.clients))
        )
        return {
            client: poison_data(data)
            if client in poison_clients
            else client_batch_data(data)
            for (client, data) in self.clients.items()
        }
