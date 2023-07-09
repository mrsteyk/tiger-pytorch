import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    print(
        "triton is not installed, please install by running"
        + "`pip install triton -U --pre`"
    )
    exit()

# clone param and exp_avg before autotuning takes place
# as those are updated in-place


def clone_inplace_updated_params(nargs):
    nargs["p_ptr"] = nargs["p_ptr"].clone()
    nargs["exp_avg_ptr"] = nargs["exp_avg_ptr"].clone()


# triton cuda kernel


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE": 128}, num_warps=4, pre_hook=clone_inplace_updated_params
        ),
        triton.Config(
            {"BLOCK_SIZE": 1024}, num_warps=8, pre_hook=clone_inplace_updated_params
        ),
    ],
    key=["n_elements"],
)
@triton.jit
def update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    lr,
    wd,
    beta1,
    beta2,
    n_elements,
    is_nan,
    grad_done,
    shrink_ratio,
    c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # offsetted pointers
    offset_p_ptr = p_ptr + offsets

    # load
    p = tl.load(offset_p_ptr, mask=mask)

    # TODO: is it okay to have branches in triton?
    #       I also do not think I can remove it...
    #       I can change values based on those bools using where but that's it
    if is_nan:
        p = p - c
        p = p * shrink_ratio
        p = p + c
        tl.store(offset_p_ptr, p, mask=mask)
    else:
        offset_grad_ptr = grad_ptr + offsets
        offset_exp_avg_ptr = exp_avg_ptr + offsets
        
        grad = tl.load(offset_grad_ptr, mask=mask)
        exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)

        if grad_done:
            exp_avg = exp_avg * beta1
        exp_avg = exp_avg + grad * beta2

        # torch.sign
        can_update = exp_avg != 0
        update_sign = tl.where(exp_avg > 0, -lr, lr)

        # stepweight decay
        p = p * (1 - lr * wd)

        # perform update
        p = p + update_sign * can_update

        tl.store(offset_p_ptr, p, mask=mask)
        tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)


def update_fn(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
    grad_done: bool,
    shrink_ratio: float,
    c=0.0,
):
    assert all([t.is_cuda for t in (p, grad, exp_avg)])
    n_elements = p.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    update_fn_kernel[grid](
        p,
        grad,
        exp_avg,
        lr,
        wd,
        beta1,
        beta2,
        n_elements,
        grad.isnan().any(),
        grad_done,
        shrink_ratio,
        c,
    )
