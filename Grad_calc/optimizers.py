# This file contain the following optimizers :
# - SGD
# - Adam
# - RMSprop

import numpy as np
import Layers.Layer as Layer
import Model.model as model


def SGD(model, lr):
    for layer in model.layers:
        if layer.trainable:
            layer.weights -= lr * layer.grads