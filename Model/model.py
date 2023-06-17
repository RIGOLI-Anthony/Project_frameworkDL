# This file contain the class model to build DNN model
import numpy as np
import Func.activation as act
import Func.loss as loss
import Grad_calc.optimizers as opt
import Grad_calc.autograd
import Layers.Layer as Layer


class Model:
    
    def __init__(self):
        self.layers = None # List of layers and not only one layer
        self.loss_func = None
        self.optimizer = None # Optimizer of the model
        self.loss = None
        self.grad_value = None # Value of the gradient for this layer