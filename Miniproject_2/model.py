from turtle import forward
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

class Module(object):
    def __init__(self):
        pass
    
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, gradwrtoutput):
        raise NotImplementedError
    
    def param(self):
        return []
    
    def __call__(self, input) :
        return self.forward(input)

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # self.parameters = 
        # self.gradient = 
    
    def forward(self, input):
        pass

class TransposeConv2d(Module):
    def __init__(self):
        super().__init__()

class NearestUpsampling(Module):
    def __init__(self):
        super().__init__()

class ReLU(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.mul(input >= 0)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.mul(-1).exp().add(1).pow(-1)

class MSE(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        return input.sub(target).pow(2).mean()
    
    def __call__(self, input, target):
        return self.forward(input, target)

class SGD(object):
    def __init__(self, parameters, lr):
        pass

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for module in args:
            self.modules.append(module)
    
    def forward(self, input):
        for module in self.modules:
            input = module(input)
        return input
    
    def backward(self, gradwrtoutput):
        for module in self.modules:
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput

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