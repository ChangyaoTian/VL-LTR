import torch
import torch.distributed as dist
from torch.autograd import Function

class GatherLayer(Function):
    '''
        Gather tensors from all process, support backward propagation.
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)
    
    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()] * dist.get_world_size()
        return grad_out