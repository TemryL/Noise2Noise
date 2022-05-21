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
    
    def __repr__(self):
        return "{}".format(self.__class__.__name__)

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
        
        self.kernel_size = self._check_argument(kernel_size, "kernel_size")
        self.stride = self._check_argument(stride, "stride")
        self.padding = self._check_argument(padding, "padding")
        self.dilation = self._check_argument(dilation, "dilation")
        
        self.weight = empty(out_channels, in_channels, *self.kernel_size)
        self.weight.grad = empty(out_channels, in_channels, *self.kernel_size).mul(0.0)
        
        if bias is True:
            self.bias = empty(out_channels)
            self.bias = empty(out_channels).mul(0.0)
        
        self.input = None
    
    def forward(self, input):
        # input : tensor of size (N, C, H, W) 
        self.input = input
        return self._convolve(input, self.out_channels, self.weight, self.bias, self.stride, self.dilation, self.padding)
    
    def backward(self, gradwrtoutput):
        # Gradient of the loss wrt the weights
        dw = empty(self.weight.shape)
        
        for b in range(self.input.shape[0]):
            test_in = self.input[b:b+1,:,:,:]
            test_out = gradwrtoutput[b:b+1,:,:,:].permute(1,0,2,3)
        
            for i in range(self.input.shape[1]):
                dw[:,i:i+1,:,:] += self._convolve(test_in[:,i:i+1,:,:], self.out_channels, test_out, padding=self.padding, stride=self.dilation, dilation=self.stride).permute(1,0,2,3)
        
        self.weight.grad += dw
        
        # Gradient of the loss wrt the bias
        db = gradwrtoutput.sum((0,2,3))
        self.bias.grad += db
        
        # Gradient of the loss wrt the module's input
        H, W = self.weight.shape[-2], self.weight.shape[-1]
        flip_weight = self._rot90(self._rot90(self.weight))
        dilated_output = self._dilate(gradwrtoutput, dilation=self.stride)
        dx =  self._convolve(dilated_output, self.in_channels, flip_weight, padding=(H-1, W-1))
        print(dx.shape)
        return dx
    
    def param(self):
        return [(self.weight, self.weight.grad), (self.bias, self.bias.grad)]
    
    def _convolve(self, input, out_channels, weight, bias=None, stride=1, dilation=1, padding=0):
        N, C, H, W = list(input.shape)
        K = (weight.shape[-2], weight.shape[-1])
        S = self._check_argument(stride)
        D = self._check_argument(dilation)
        P = self._check_argument(padding)
        
        unfolded = unfold(input, kernel_size=K, dilation=D, padding=P, stride=S)
        if bias is None:
            kxb = weight.view(out_channels, -1).matmul(unfolded)
        else:
            kxb = weight.view(out_channels, -1).matmul(unfolded) + bias.view(1, -1, 1)
        output = kxb.view(N, out_channels , int((H+2*P[0]-D[0]*(K[0]-1)-1)/S[0] + 1) , int((W+2*P[1]-D[1]*(K[1]-1)-1)/S[1] + 1))
        return output
    
    def _padd(self, input, padding):
        N, C, H, W = list(input.shape)
        P = self._check_argument(padding)
        padded = unfold(input, kernel_size = 1, padding = P)
        padded = fold(padded, output_size = (H+2*P[0],W+2*P[1]), kernel_size = 1)
        return padded
    
    def _dilate(self, input, dilation):
        N, C, H, W = list(input.shape)
        D = self._check_argument(dilation)
        dilated = empty(N ,C, H + (H-1)*(D[0]-1), W + (W-1)*(D[1]-1)).mul(0.0)
        dilated[:,:,::D[0],::D[1]] = input
        return dilated
    
    def _rot90(self, input):
        input = input.transpose(-1,-2)
        rot = empty(input.shape)
        for i in range(input.shape[-2]):
            rot[:,:,i,:] = input[:,:,-i-1,:]
        return rot
    
    def _check_argument(self, arg, name=""):
        if isinstance(arg, int):
            arg = (arg, arg)
        
        elif not(isinstance(arg, tuple) and len(arg)==2):
            if name != "":
                raise ValueError("Invalid input argument when instantiating Conv2d class: {} must be int or tuple of ints of size 2".format(name))
            else:
                raise ValueError("Invalid argument when calling internal function: check kernel_size, padding, stride or dilation")
        
        return arg
    
    def __repr__(self):
        if self.bias is not None:
            bias = True
        else:
            bias = False
        return "Conv2d(in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, dilation={}, bias={})".format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, bias)

# class TransposeConv2d(Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
#         super().__init__()
        
#         if isinstance(in_channels, int):
#                 self.in_channels = in_channels
#         else:
#             raise ValueError('Invalid input argument when instantiating TransposeConv2d class: in_channels must be int')
        
#         if isinstance(in_channels, int):
#             self.out_channels = out_channels
#         else:
#             raise ValueError('Invalid input argument when instantiating TransposeConv2d class: out_channels must be int')
        
#         self.kernel_size = self._check_argument(kernel_size, "kernel_size")
#         self.stride = self._check_argument(stride, "stride")
#         self.padding = self._check_argument(padding, "padding")
#         self.dilation = self._check_argument(dilation, "dilation")
        
#         self.x = empty(out_channels, in_channels, *self.kernel_size)
#         if bias is True:
#             self.bias = empty(out_channels)
        
#         self.input = None
        
#     def forward(self, input):
#         # input : tensor of size (N, C, H, W) 
#         N, C, H, W = list(input.shape)
#         K = self.kernel_size
#         S = self.stride
        
#         unfolded = unfold(input, kernel_size=K, stride=S, padding=(H-K[0], W-K[1]))
#         wxb = self.weight.view(self.out_channels, -1).matmul(unfolded) + self.bias.view(1, -1, 1)
#         output = wxb.view(N, self.out_channels , S[0]*(H-1) + K[0], S[1]*(W-1) + K[1])
#         return output
    
#     def _check_argument(self, arg, name=""):
#         if isinstance(arg, int):
#             arg = (arg, arg)
        
#         elif not(isinstance(arg, tuple) and len(arg)==2):
#             raise ValueError("Invalid input argument when instantiating TransposeConv2d class: {} must be int or tuple of ints of size 2".format(name))
        
#         return arg

# class NearestUpsampling(Module):
#     def __init__(self):
#         super().__init__()

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.input = None
    
    def forward(self, input):
        self.input = input
        return input.mul(input >= 0)
    
    def backward(self, gradwrtoutput):
        # Gradient of the loss wrt the module's input
        print(gradwrtoutput.shape)
        print(self.input.shape)
        return gradwrtoutput.mul(self._dReLU(self.input))
    
    def _dReLU(self, input):
        output = empty(input.shape)
        output[input>0] = 1
        output[input<=0] = 0
        return output

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.input = None
    
    def forward(self, input):
        self.input = input
        return input.mul(-1).exp().add(1).pow(-1)
    
    def backward(self, gradwrtoutput):
        # Gradient of the loss wrt the module's input
        return gradwrtoutput.mul(self._dSigmoid(self.input))
    
    def _dSigmoid(self, input):
        return input.mul(-1).exp().div(input.mul(-1).exp().add(1).pow(2))

class MSE(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        self.input = input
        self.target = target
        return input.sub(target).pow(2).mean()
    
    def backward(self):
        return self.input.sub(self.target).mul(2).div(self.input.numel())
    
    def __call__(self, input, target):
        return self.forward(input, target)

class SGD(object):
    def __init__(self, parameters, lr):
        # parameters: list of pairs (2-tuple) of parameters returned by param() function from class Module
        self.params = parameters
        self.lr = lr
    
    def zero_grad(self):
        for param in self.params:
            param[1].mul_(0.0)
    
    def step(self):
        for param in self.params:
            param[0].add_(param[1].mul(-self.lr))

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
        for module in reversed(self.modules):
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput
    
    def param(self):
        param = []
        for module in self.modules:
            param.extend(module.param())
        return param
    
    def __repr__(self):
        name = ""
        for module in self.modules:
            name += "   " + module.__repr__() + "\n"
        return "Sequential(\n{})".format(name)

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
    
    def __repr__(self):
        pass