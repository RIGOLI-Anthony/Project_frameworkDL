# The goal here is to create some functions that can be used to calculate the gradient of a function
# This would be pretty much the autograd package in pytorch but coded from scratch

import numpy as np
import Func.activation as act
import Func.loss as loss

class GradCalc:
    
    def __init__(self, X, y, model, loss_func):
        self.X = X
        self.y = y
        self.model = model
        self.loss_func = loss_func