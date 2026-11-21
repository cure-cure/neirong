import torch
import torch.nn.functional as F

def attack(logit_fn,x,y,epsilon,steps=10,restarts=1,random_start=True,lower=0.,upper=1.,seed=202430,alpha=None):
    """Untargeted L-inf PGD; optimize only input; select worst restart per example."""
    generator=torch.Generator(device=x.device).manual_seed(seed)
    step=alpha if alpha is not None else (epsilon if steps==1 else epsilon/4)
    target=torch.full((len(x),),float(y),device=x.device)
    best=x.clone();best_loss=torch.full((len(x),),-float('inf'),device=x.device)
    for restart in range(restarts):
        z=x.clone()
        if random_start:z=(z+torch.empty_like(z).uniform_(-epsilon,epsilon,generator=generator)).clamp(lower,upper)
        for _ in range(steps):
            z=z.detach().requires_grad_(True)
            loss=F.binary_cross_entropy_with_logits(logit_fn(z),target)
            grad=torch.autograd.grad(loss,z,only_inputs=True)[0]
            z=z.detach()+step*grad.sign();z=torch.maximum(torch.minimum(z,x+epsilon),x-epsilon).clamp(lower,upper)
        with torch.no_grad():
            loss=F.binary_cross_entropy_with_logits(logit_fn(z),target,reduction='none');mask=loss>best_loss
            best[mask]=z[mask];best_loss[mask]=loss[mask]
    assert float((best-x).abs().max())<=epsilon+1e-6
    return best.detach()
