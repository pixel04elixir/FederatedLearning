import tensorflow as tf

from FLearning.fl_main import FLBase
from FLearning.utils import *


class FLPoison(FLBase):
    def __init__(
        self,
        img_path,
        rounds=100,
        lr=0.01,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        momentum=0.9,
        optimizer=tf.keras.optimizers.legacy.SGD,
        poison_clients=0.1,
    ):
        self.poison_clients = poison_clients
        super().__init__(img_path, rounds, lr, loss, metrics, momentum, optimizer)

    def _get_client_data(self):
        assert self.poison_clients <= len(self.clients)
        poison_clients = random.sample(
            self.clients.keys(), int(self.poison_clients * len(self.clients))
        )
        return {
            client: poision_data(data) if client in poison_clients else batch_data(data)
            for (client, data) in self.clients.items()
        }
