from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

# update functions

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2, step, gas, s, c = 0):
    # _prepare
    # if step % gas != 0:
    #     beta1 = 1.
    
    # this is really weird NaN circumvention
    is_nan = grad.isnan().any()
    # ok so now it's weird
    # if not is_nan:
    #     # exp_avg = beta1 * exp_avg + beta2 * grad
    #     exp_avg.mul_(beta1).add_(grad, alpha = beta2)
    # else:
    #     beta1 = 1.
    #     # why is this like that?
    #     grad = torch.zeros_like(grad)
    #     # In the end exp_avg doesn't get changed
    #     # beta1 is set to 1, exp_avg the same
    #     # beta2*grad, grad is 0, exp_avg the same
    
    if is_nan:
        # Can this be simplified?
        # I am not doing it because fp16
        p.sub_(c)
        p.mul_(s)
        p.add_(c)
    else:
        # we are done accumulating gradient
        if step % gas == 0:
            exp_avg.mul_(beta1)
        exp_avg.add_(grad, alpha = beta2)
        # u = (exp_avg.sign() + wd * p) * lr
        # p.sub_(u)
        # p - p * lr * wd
        # it is really weird that we are updating gradients even when doing accum
        p.data.mul_(1 - lr * wd)
        p.add(exp_avg.sign(), alpha = -lr)

# class

class Tiger(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.965,
        grad_accum_steps: int = 1,
        weight_decay: float = 0.01,
        shrink_ratio: float = 0.99,
        use_triton: bool = False,
        use_cuda: bool = False,
    ):
        assert lr > 0.
        assert 0. <= beta1 <= 1.
        assert grad_accum_steps >= 1

        # TODO: figure out how to change grad accum when reloading training

        beta2 = (1 - beta1) / grad_accum_steps
        defaults = dict(
            lr = lr,
            betas = (beta1, beta2),
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

        self.update_fn = update_fn
        self.grad_accum_steps = grad_accum_steps
        self.shrink_ratio = shrink_ratio

        # Do pytorch optimisers have that?
        # self.step_i = 0

        # if use_triton:
        #     from lion_pytorch.triton import update_fn as triton_update_fn
        #     self.update_fn = triton_update_fn

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):
                # TODO: figure out if step is called when rem grad accum is not zero!
                #       this is very important for this optimiser!
                #       maybe fake grad accum?

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['step'] = 0
                    # should be different per parameter
                    # perhaps put in a group?
                    state['c'] = group.get('c', 0)

                state['step'] += 1

                exp_avg = state['exp_avg']
                step = state['step']
                c = state['c']

                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2,
                    step,
                    self.grad_accum_steps,
                    c,
                )

        return loss