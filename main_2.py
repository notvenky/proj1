import torch

class Oscilator(torch.nn.Module):
     def __init__(self):
           super().__init__()
           self.myparam = torch.nn.Parameter(torch.zeros(1))


class Oscilator(torch.nn.Module):
     def __init__(self):
         super().__init__()
         self.myparam = torch.nn.Parameter(torch.zeros(1))
     def forward(self, x):
         return self.myparam * x

model = Oscilator()

print(list(model.parameters()))