import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
from pathlib import Path

class Module(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
            raise ValueError('Invalid input argument when instantiating {} class: in_channels must be int'.format(self.__class__.__name__))
        
        if isinstance(out_channels, int):
            self.out_channels = out_channels
        else:
            raise ValueError('Invalid input argument when instantiating {} class: out_channels must be int'.format(self.__class__.__name__))
        
        self.kernel_size = self._check_argument(kernel_size, "kernel_size")
        self.stride = self._check_argument(stride, "stride")
        self.padding = self._check_argument(padding, "padding")
        self.dilation = self._check_argument(dilation, "dilation")
        
        k = 1/(self.in_channels*self.kernel_size[0]*self.kernel_size[1])
        self.weight = empty(out_channels, in_channels, *self.kernel_size).to(self.device).uniform_(-k**(1/2), k**(1/2))
        self.weight.grad = empty(out_channels, in_channels, *self.kernel_size).to(self.device).zero_()
        
        if bias is True:
            self.bias = empty(out_channels).to(self.device).uniform_(-k**(1/2), k**(1/2))
            self.bias.grad = empty(out_channels).to(self.device).zero_()
        
        self.input = None
    
    def forward(self, input):
        # input : tensor of size (N, C, H, W) 
        self.input = input
        return self._convolve(input, self.out_channels, self.weight, self.bias, self.stride, self.dilation, self.padding)
    
    def backward(self, gradwrtoutput):
        K = self.weight.shape[-2], self.weight.shape[-1]
        
        # Gradient of the loss wrt the weights
        dw = empty(self.weight.shape).to(self.device).zero_()
        
        unfolded_grad = gradwrtoutput.view(self.input.shape[0], self.out_channels, -1)
        unfolded_input = unfold(self.input, kernel_size = K, stride = self.stride, padding = self.padding)
        dw = unfolded_grad.matmul(unfolded_input.transpose(1,2)).sum(0)
        dw = dw.view(dw.shape[0], dw.shape[1], 1)
        dw = fold(dw, output_size = K, kernel_size = K, stride = self.stride)
        
        self.weight.grad.add_(dw)
        
        # Gradient of the loss wrt the bias
        db = gradwrtoutput.sum((0,2,3))
        self.bias.grad.add_(db)
        
        # Gradient of the loss wrt the module's input
        P = self.padding
        flip_weight = self._rot90(self._rot90(self.weight))
        dilated_output = self._dilate(gradwrtoutput, dilation=self.stride)
        dx =  self._convolve(dilated_output, self.in_channels, flip_weight, padding=(K[0]-P[0]-1, K[1]-P[1]-1))
        
        return dx
    
    def param(self):
        return [[self.weight, self.weight.grad], [self.bias, self.bias.grad]]
    
    def _convolve(self, input, out_channels, weight, bias=None, stride=1, dilation=1, padding=0):
        N, C, H, W = list(input.shape)
        K = (weight.shape[-2], weight.shape[-1])
        S = self._check_argument(stride)
        D = self._check_argument(dilation)
        P = self._check_argument(padding)
        
        unfolded = unfold(input, kernel_size=K, dilation=D, padding=P, stride=S)
        if bias is None:
            kxb = weight.reshape(out_channels, -1).matmul(unfolded)
        else:
            kxb = weight.reshape(out_channels, -1).matmul(unfolded) + bias.reshape(1, -1, 1)
        output = kxb.reshape(N, out_channels , int((H+2*P[0]-D[0]*(K[0]-1)-1)/S[0] + 1) , int((W+2*P[1]-D[1]*(K[1]-1)-1)/S[1] + 1))
        return output
    
    def _padd(self, input, padding):
        N, C, H, W = list(input.shape)
        P = self._check_argument(padding)
        padded = unfold(input, kernel_size = 1, padding = P)
        padded = fold(padded, output_size = (H+2*P[0],W+2*P[1]), kernel_size = 1)
        return padded
    
    def _padd_bottom(self, input, padd_bot):
        if padd_bot <= 0:
            return input
        N, C, H, W = list(input.shape)
        padded = empty(N ,C, H+padd_bot, W).to(self.device).zero_()
        padded[:,:,:-padd_bot,:] = input
        return padded
    
    def _padd_right(self, input, padd_right):
        if padd_right <= 0:
            return input
        N, C, H, W = list(input.shape)
        padded = empty(N ,C, H, W+padd_right).to(self.device).zero_()
        padded[:,:,:,:-padd_right] = input
        return padded
    
    def _dilate(self, input, dilation):
        N, C, H, W = list(input.shape)
        D = self._check_argument(dilation)
        dilated = empty(N ,C, H + (H-1)*(D[0]-1), W + (W-1)*(D[1]-1)).to(self.device).zero_()
        dilated[:,:,::D[0],::D[1]] = input
        return dilated
    
    def _rot90(self, input):
        input = input.transpose(-1,-2)
        rot = empty(input.shape).to(self.device)
        for i in range(input.shape[-2]):
            rot[:,:,i,:] = input[:,:,-i-1,:]
        return rot
    
    def _check_argument(self, arg, name=""):
        if isinstance(arg, int):
            arg = (arg, arg)
        
        elif not(isinstance(arg, tuple) and len(arg)==2):
            if name != "":
                raise ValueError("Invalid input argument when instantiating {} class: {} must be int or tuple of ints of size 2".format(self.__class__.__name__, name))
            else:
                raise ValueError("Invalid argument when calling internal function: check kernel_size, padding, stride or dilation")
        
        return arg
    
    def __repr__(self):
        if self.bias is not None:
            bias = True
        else:
            bias = False
        return "{}(in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, dilation={}, bias={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, bias)

class NearestUpsampling(Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
    
    def forward(self, input):
        # Input: tensor of size N, C, H, W
        # Output: tensor of size N, C, scale_factor x H, scale_factor x W
        return input.repeat_interleave(self.scale_factor,-1).repeat_interleave(self.scale_factor,-2) 
    
    def backward (self, gradwrtoutput):
        scale_factor = self.scale_factor
        
        dx = gradwrtoutput.unfold(2, scale_factor, scale_factor).unfold(3, scale_factor, scale_factor)
        dx = dx.contiguous().view(dx.shape[0], dx.shape[1], -1, dx.shape[4], dx.shape[5])
        dx = dx.view(dx.shape[0], dx.shape[1], dx.shape[2], -1)
        dx = dx.sum(dim=3)
        dx = dx.view(dx.shape[0], dx.shape[1], int(gradwrtoutput.shape[3]/self.scale_factor), int(gradwrtoutput.shape[3]/self.scale_factor))
        
        return dx

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.input = None
    
    def forward(self, input):
        self.input = input
        return input.mul((input >= 0).float())
    
    def backward(self, gradwrtoutput):
        # Gradient of the loss wrt the module's input
        return gradwrtoutput.mul(self._dReLU(self.input))
    
    def _dReLU(self, input):
        output = (input > 0).float()
        return output

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.input = None
    
    def forward(self, input):
        self.input = input
        return input.mul(-1.0).exp().add(1.0).pow(-1.0)
    
    def backward(self, gradwrtoutput):
        # Gradient of the loss wrt the module's input
        y = (-self.input).exp().div((1+(-self.input).exp())**2)
        out = y.mul(gradwrtoutput)
        return out

class MSE(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        self.input = input
        self.target = target
        return input.sub(target).pow(2.0).nanmean()
    
    def backward(self):
        return self.input.sub(self.target).mul(2.0).div(float(self.input.numel()))
    
    def __call__(self, input, target):
        return self.forward(input, target)

class SGD(object):
    def __init__(self, model, lr):
        # parameters: list of pairs (2-list) of parameters returned by param() function from class Module
        self.model = model
        self.lr = lr
    
    def zero_grad(self):
        for i in range(len(self.model.modules)):
            if self.model.modules[i].param() == []:
                continue
            self.model.modules[i].weight.grad.zero_()
            self.model.modules[i].bias.grad.zero_()
    
    def step(self):
        for i in range(len(self.model.modules)):
            if self.model.modules[i].param() == []:
                continue
            self.model.modules[i].weight.sub_(self.lr*self.model.modules[i].weight.grad)
            self.model.modules[i].bias.sub_(self.lr*self.model.modules[i].bias.grad)

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

class Model(Module):
    def __init__(self):
        ## instantiate model + optimizer + loss function + any other stuff you need
        super().__init__()
        
        self.model = Sequential(Conv2d(3, 32, kernel_size=2, stride=2), 
                                ReLU(),
                                Conv2d(32, 32, kernel_size=2, stride=2),
                                ReLU(),
                                NearestUpsampling(scale_factor=2),
                                Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                ReLU(),
                                NearestUpsampling(scale_factor=2),
                                Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
                                Sigmoid())
        
        self.optimizer = SGD(self.model, lr=1)
        self.criterion = MSE()
    
    def load_pretrained_model(self):
        # This loads the parameters saved in bestmodel .pth into the model
        model_path = Path(__file__).parent/"bestmodel.pth"
        parameters = torch.load(model_path, map_location=torch.device(self.device))
            
        for i in range(len(self.model.modules)):
            if self.model.modules[i].param() == []:
                continue
            self.model.modules[i].weight.copy_(parameters.pop(0)[0])
            self.model.modules[i].bias.copy_(parameters.pop(0)[0])
    
    def train(self, train_input, train_target, num_epochs):
        # train_input : tensor of size (N, C, H, W) containing a noisy version of the images with values in range 0-255.
        # train_target : tensor of size (N, C, H, W) containing another noisy version of the
        # same images , which only differs from the input by their noise, with values in range 0-255.
        
        # Normalize data
        train_input = train_input.div(255.0)
        train_target = train_target.div(255.0)
        
        mini_batch_size = 50
        for e in range(num_epochs):
            epoch_loss = 0
            
            for b in range(0, train_input.size(0), mini_batch_size):
                output = self(train_input.narrow(0, b, mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, mini_batch_size))
                epoch_loss += loss
                self.optimizer.zero_grad()
                self.backward(self.criterion.backward())
                self.optimizer.step()
                
            print("Epoch {}: Loss {}".format(e, epoch_loss))
    
    def predict(self, test_input):
        # test_input : tensor of size (N1 , C, H, W) with values in range 0-255 that has to be denoised by the trained
        # or the loaded network .
        # returns a tensor of the size (N1 , C, H, W) with values in range 0-255.
        
        test_input = test_input.div(255.0)
        test_output = self(test_input).mul(255.0)
        return test_output
    
    def forward(self, input):
        return self.model(input)
    
    def backward(self, gradwrtoutput):
        return self.model.backward(gradwrtoutput)
    
    def param(self):
        return self.model.param()
    
    def __repr__(self):
        name = self.model.__repr__().replace("Sequential", self.__class__.__name__)
        name = name[:-1] + "   (criterion): {}\n)".format(self.criterion.__class__.__name__)
        return name