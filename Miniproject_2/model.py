from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

def Conv2d():
    pass

def TransposeConv2d():
    pass
def NearestUpsampling():
    pass

def ReLU():
    pass

def Sigmoid(input):
    return input.exp().div(input.exp()+1)

def MSE():
    pass

def SGD():
    pass

class Sequential():
    pass

class Model():
    def __init__(self):
        ## instantiate model + optimizer + loss function + any other stuff you need
        pass
    
    def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel .pth into the model
        pass
    
    def train(self, train_input, train_target):
        # train_input : tensor of size (N, C, H, W) containing a noisy version of the images
        # train_target : tensor of size (N, C, H, W) containing another noisy version of the
        # same images , which only differs from the input by their noise .
        pass
    
    def predict(self, test_input):
        # test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network .
        # returns a tensor of the size (N1 , C, H, W)
        pass