from tkinter import SE
import torch
from model import ReLU, Sequential
from torch import nn

if __name__ == "__main__":
    r1 = ReLU()
    tt = torch.randn(3,3,7)
    seq = Sequential(ReLU(), ReLU())
    print(torch.eq(seq(tt), r1(r1(tt))))
    
