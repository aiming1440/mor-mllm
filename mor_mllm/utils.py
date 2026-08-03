import numpy as np

def order_pick_k(lst, k):
    if len(lst) <= k:
        return lst
    rng = np.random.random(len(lst))
    index = np.argsort(rng)[:k]
    index_sort = sorted(index)
    new_lst = [lst[i] for i in index_sort]
    print(
        f"WARNING: total file: {len(lst)}, random pick: {k}."
        f" (ignored)"
    )
    return new_lst

class HookTool:
    def __init__(self):
        self.fea = None
    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out.detach().cpu()

def get_gating_logit_by_hook(model):
    fea_hooks = []
    for n, m in model.named_modules():
        if 'wg' in n and isinstance(m, nn.Linear):
            print(n, m, 'match!!!!!!!!!!!!!!!!!!!!!!!!!')
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)
    return fea_hooks

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
