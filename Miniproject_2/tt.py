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
        
        self.weight = empty(out_channels, in_channels, *self.kernel_size).uniform_(-1,1)
        self.weight.grad = empty(out_channels, in_channels, *self.kernel_size).mul(0.0)
        
        if bias is True:
            self.bias = empty(out_channels).uniform_(-1,1)
            self.bias = empty(out_channels).mul(0.0)
        
        self.input = None
    
    def forward(self, input):
        # input : tensor of size (N, C, H, W) 
        self.input = input
        return self._convolve(input, self.out_channels, self.weight, self.bias, self.stride, self.dilation, self.padding)
    
    def backward(self, gradwrtoutput):
        # Gradient of the loss wrt the weights
        dw = empty(self.weight.shape).mul(0.0)
        
        for b in range(self.input.shape[0]):
            permuted_output = gradwrtoutput[b:b+1,:,:,:].permute(1,0,2,3)
            for i in range(self.input.shape[1]):
                # Problem with stride
                dw[:,i:i+1,:,:] += self._convolve(self.input[b:b+1,i:i+1,:,:], self.out_channels, permuted_output, padding=self.padding, stride=self.dilation, dilation=self.stride).permute(1,0,2,3)
        
        self.weight.grad += dw
        
        # Gradient of the loss wrt the bias
        db = gradwrtoutput.sum((0,2,3))
        self.bias.grad += db
        
        # Gradient of the loss wrt the module's input
        H, W = self.weight.shape[-2], self.weight.shape[-1]
        flip_weight = self._rot90(self._rot90(self.weight))
        dilated_output = self._dilate(gradwrtoutput, dilation=self.stride)
        
        dx =  self._convolve(dilated_output, self.in_channels, flip_weight, padding=(H-1,W-1))
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
            kxb = weight.reshape(out_channels, -1).matmul(unfolded)
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