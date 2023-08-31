import torch

class MLP(torch.nn.Module):
  def __init__(self, in_dim, hid_dim , out_dim):
    """ This is the constructor. Initialize your network
    layers here.
    Inputs
    --------
    in_dim: int. Dimension of your input. It was 2 in the example
    hid_dim: int. Dimension of the intermediate level. Also 2 in the example
    out_dim: int. Dimension of your output. It was 1 in the example
    """
    super().__init__()        # Invoke the constructor of the parent class
    self.w = torch.nn.Linear(in_dim, hid_dim)
    self.y = torch.nn.ReLU()
    self.w3 = torch.nn.Linear(hid_dim, out_dim)

  def forward(self, inp):
    """ Method for forward pass
    Input
    -------
    inp: torch.Tensor. The dimension of the input (i.e, the last number in shape)
        should match `in_dim`. The shape should be (n,in_dim) where n is the batch size

    Output
    ------
    z: torch.Tensor. The result of the forward pass. The dimension of the output
     (i.e, the last number in shape) should match `out_dim`. The shape should be
     (n, out_dim) where n is the batch size.
    """
    z = self.w3(self.y(self.w(inp)))
    return z