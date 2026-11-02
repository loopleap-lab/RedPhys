import torch
from .utils import project_perturbation, normalize_grad
from ... import trace


def pgd_veattack(
        forward,
        loss_fn,
        data_clean,
        norm,
        eps,
        iterations,
        stepsize,
        output_normalize,
        perturbation=None,
        mode='min',
        momentum=0.9,
        verbose=False
):
    """Momentum PGD attack on vision encoder tokens."""
    assert torch.max(data_clean) < 1. + 1e-6 and torch.min(data_clean) > -1e-6

    if perturbation is None:
        perturbation = torch.zeros_like(data_clean, requires_grad=True)
    velocity = torch.zeros_like(data_clean)
    for i in range(iterations):
        perturbation.requires_grad = True
        with torch.enable_grad():
            embedding, tokens = forward(data_clean + perturbation,
                                        output_normalize=output_normalize, tokens=True)
            loss = loss_fn(embedding, tokens)
            if verbose:
                print(f'[{i}] {loss.item():.5f}')
            if i % trace.EVERY == 0 or i == iterations - 1:
                trace.log(i, loss=loss)

        with torch.no_grad():
            gradient = torch.autograd.grad(loss, perturbation)[0]
            gradient = torch.where(gradient.isnan(), 0., gradient)  # no per-step sync
            gradient = normalize_grad(gradient, p=norm)
            velocity = momentum * velocity + gradient
            velocity = normalize_grad(velocity, p=norm)
            if mode == 'min':
                perturbation = perturbation - stepsize * velocity
            elif mode == 'max':
                perturbation = perturbation + stepsize * velocity
            else:
                raise ValueError(f'Unknown mode: {mode}')
            perturbation = project_perturbation(perturbation, eps, norm)
            perturbation = torch.clamp(
                data_clean + perturbation, 0, 1
            ) - data_clean

    assert not perturbation.isnan().any()  # NaN persists through clamp, so checking once suffices
    assert torch.max(data_clean + perturbation) < 1. + 1e-6 and torch.min(data_clean + perturbation) > -1e-6
    return data_clean + perturbation.detach()
