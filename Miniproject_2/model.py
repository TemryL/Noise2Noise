from turtle import forward
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

def padd(input, padd_size):
    pass

def dilate():
    pass
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        
        super().__init__()
        
        if isinstance(in_channels, int):
            self.in_channels = in_channels
        else:
            raise ValueError('Invalid input argument when instantiating Conv2d class: in_channels must be int')
        
        if isinstance(out_channels, int):
            self.out_channels = out_channels
        else:
            raise ValueError('Invalid input argument when instantiating Conv2d class: out_channels must be int')
        
        self.kernel_size = self._check_argmument(kernel_size, "kernel_size")
        self.stride = self._check_argmument(stride, "stride")
        self.padding = self._check_argmument(padding, "padding")
        self.dilation = self._check_argmument(dilation, "dilation")
        
        self.x = empty(out_channels, in_channels, *self.kernel_size)
        if bias is True:
            self.bias = empty(out_channels)
        
        self.input = None
    
    def forward(self, input):
        # input : tensor of size (N, C, H, W) 
        self.input = input
        
        return self._convolve(input, self.out_channels, self.weight, self.bias, self.stride, self.dilation, self.padding)
    
    # def backward(self, gradwrtoutput):
    #     # Gradient of the loss wrt the weights
    #     dw = self._convolve(self.input, self.out_channels, gradwrtoutput, dilation=(self.stride[0], self.stride[1]))
        
    #     # Gradient of the loss wrt the bias
    #     db = gradwrtoutput.sum((0,2,3))
        
    #     # Gradient of the loss wrt the module's input
    #     flip_weight = 
    #     dx = 
    #     return
    
    def param(self):
        return [(self.weight, self.weight.grad), (self.bias, self.bias.grad)]
    
    def _convolve(self, input, out_channels, weight, bias=None, stride=1, dilation=1, padding=0):
        N, C, H, W = list(input.shape)
        K = (weight.shape[-2], weight.shape[-1])
        S = self._check_argmument(stride)
        D = self._check_argmument(dilation)
        P = self._check_argmument(padding)
        
        unfolded = unfold(input, kernel_size=K, dilation=D, padding=P, stride=S)
        if bias is None:
            kxb = weight.view(out_channels, -1).matmul(unfolded)
        else:
            kxb = weight.view(out_channels, -1).matmul(unfolded) + bias.view(1, -1, 1)
        output = kxb.view(N, out_channels , int((H+2*P[0]-D[0]*(K[0]-1)-1)/S[0] + 1) , int((W+2*P[1]-D[1]*(K[1]-1)-1)/S[1] + 1))
        return output
    
    def _check_argmument(self, arg, name=""):
        if isinstance(arg, int):
            arg = (arg, arg)
        
        elif not(isinstance(arg, tuple) and len(arg)==2):
            raise ValueError("Invalid input argument when instantiating Conv2d class: {} must be int or tuple of ints of size 2".format(name))
        
        return arg

class TransposeConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        
        if isinstance(in_channels, int):
                self.in_channels = in_channels
        else:
            raise ValueError('Invalid input argument when instantiating TransposeConv2d class: in_channels must be int')
        
        if isinstance(in_channels, int):
            self.out_channels = out_channels
        else:
            raise ValueError('Invalid input argument when instantiating TransposeConv2d class: out_channels must be int')
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple) and len(kernel_size)==2:
            self.kernel_size = kernel_size
        else:
            raise ValueError('Invalid input argument when instantiating TransposeConv2d class: kernel_size must be int or tuple of ints of size 2')
        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple) and len(stride)==2:
            self.stride = stride
        else:
            raise ValueError('Invalid input argument when instantiating TransposeConv2d class: stride must be int or tuple of ints of size 2')
        
        self.weight = empty(out_channels, in_channels, *self.kernel_size)
        if bias is True:
            self.bias = empty(out_channels)
    
    # def forward(self, input):
    #     # input : tensor of size (N, C, H, W) 
    #     N, C, H, W = list(input.shape)
    #     K = self.kernel_size
    #     S = self.stride
        
    #     unfolded = unfold(input, kernel_size=K, stride=S, padding=(H-K[0], W-K[1]))
    #     wxb = self.weight.view(self.out_channels, -1).matmul(unfolded) + self.bias.view(1, -1, 1)
    #     output = wxb.view(N, self.out_channels , S[0]*(H-1) + K[0], S[1]*(W-1) + K[1])
    #     return output


# class NearestUpsampling(Module):
#     def __init__(self):
#         super().__init__()

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
    
    # def backward(self, gradwrtoutput):
    #     for module in self.modules:
    #         gradwrtoutput = module.backward(gradwrtoutput)
    #     return gradwrtoutput

class Model():
    def __init__(self):
        ## instantiate model + optimizer + loss function + any other stuff you need
        pass
    
    def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel .pth into the model
        pass
    
    def train(self, train_input, train_target):
        # train_input : tensor of size (N, C, H, W) containing a noisy version of the images with values in range 0-255.
        # train_target : tensor of size (N, C, H, W) containing another noisy version of the
        # same images , which only differs from the input by their noise, with values in range 0-255.
        pass
    
    def predict(self, test_input):
        # test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network, with values in range 0-255.
        # returns a tensor of the size (N1 , C, H, W) with values in range 0-255.
        pass