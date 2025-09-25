import torch as th
from fancy_einsum import einsum
from tqdm import tqdm

def MO_MHIS(
        model,
        orig_dists: list[Discrete],
        target: int,
        *,
        temp: float,
        n_samples: int,
        burn_in: int,
        batch_size: int = 32,
        lam: float = 0.5,
        window_size: int = 2, 
        show_progress: bool = False

) -> float:
    """
    Momentum-based MHIS includes a pseudo-momentum-like smoothing term for the score function. This term smooths across token positions.
    The smoothing coefficient, lam, can be specified (between range [0, 1]), as well as the window size, which takes a local average across nearby tokens.
    """

    # the first part, aka calculating the log probabilities of input dist p(x) is the same as original MHIS
    d_vocab = model.embed.d_vocab 
    ctx_len = len(orig_dists) 
    scores = th.zeros((ctx_len, d_vocab), device=model.device) 

    #* we're not training the model so freeze the parameters
    for param in model.parameters():
        param.requires_grad_(False)

    # here, calculate the orig log probabilities (un-upweighted distribution) over the entire vocabulary
    orig_log_probs = []
    for pos in range(ctx_len): 
        mask = -th.inf * th.ones(d_vocab, device=model.device) 
        mask[orig_dists[pos].values] = th.log(orig_dists[pos].probs) 
        orig_log_probs.append(mask) 
    orig_log_probs = th.stack(orig_log_probs)

    results = []
    scores = []

    # same sampling process as original MHIS - sample batch_size parallel runs 
    with th.enable_grad():
        current_samples = th.stack([dist.sample((batch_size,)) for dist in orig_dists], dim=1) # shape: [batch_size, ctx_len]

        acceptance_rate = 0
        total_proposals = 0

        # here, we can start by initializing the scores, which will be used for smoothing
        # ema stands for exponentially moving average, which is the approximation we'll use for momentum
        ema_scores = th.zeros((batch_size, ctx_len), device = model.device)

        for step in tqdm(range((n_samples // batch_size + burn_in)), disable=not show_progress):

            # same as normal MHIS - go through forward pass and then compute gradients
            onehot = th.nn.functional.one_hot(current_samples,)
            onehot.requires_grad_(True)

            x = onehot @ model.embed.W_E
            x = x + model.pos_embed(current_samples)
            for block in model.blocks:
                x = block(x)
            x = model.ln_final(x[:, -1].unsqueeze(1))
            y = model.unembed(x).squeeze(1)
            current_scores = y[:, target]
            current_scores.sum().backward()
            current_grads = onehot.grad.detach().clone()

            # we want to apply the EMA to the scores without affecting backprop, so we can also detach the scores
            current_scores = current_scores.detach().clone()

            # then, implement smoothing across token positions
            # first, calculate the local neighbor average 
            # current_scores has shape [batch_size, ctx_len]
            # pad_scores uses replicate to copy the edge values, after padding, pad_scores has shape [batch_size, ctx_len + 2*w] (where w is the num neighbors we're using)
            # the first w columns replicate the first tok, and the last w columns represent the last tok 
            pad_scores = th.nn.functional.pad(current_scores, (window_size, window_size), mode = 'replicate') # replicate edge values, after padding, pad_scores has shape [batch_size, ctx_len + 2*w]
            avg_neighbors = th.stack([pad_scores[:, i:i+2*window_size+1].mean(dim=1) for i in range(ctx_len)], dim=1)
            # apply EMA
            ema_scores = (1 - lam) * current_scores + lam * avg_neighbors
            smoothed_scores = ema_scores

            # pick a random token position for proposal
            pos = th.randint(0, ctx_len, (batch_size,), device=current_samples.device)

            # compute proposal probabilities using smoothed scores
            proposal_logits = orig_log_probs[pos] + current_grads[th.arange(batch_size), pos] / temp
            proposal_probs = th.softmax(proposal_logits, dim=-1)
            proposed_tokens = th.multinomial(proposal_probs, 1).squeeze(-1)

            proposed_samples = current_samples.clone()
            proposed_samples[th.arange(batch_size), pos] = proposed_tokens

            # recompute scores and grads for proposed samples
            onehot = th.nn.functional.one_hot(proposed_samples, num_classes=d_vocab).float()
            onehot.requires_grad_(True)
            x = onehot @ model.embed.W_E
            x = x + model.pos_embed(proposed_samples)
            for block in model.blocks:
                x = block(x)
            x = model.ln_final(x[:, -1].unsqueeze(1))
            y = model.unembed(x).squeeze(1)
            proposed_scores = y[:, target]
            proposed_scores.sum().backward()
            proposed_grads = onehot.grad.clone()
            proposed_results = (y.argmax(dim=-1) == target).float()
            onehot.grad = None

            reverse_proposal_logits = orig_log_probs[pos] + proposed_grads[th.arange(batch_size), pos] / temp
            reverse_proposal_probs = th.softmax(reverse_proposal_logits, dim=-1)

            log_accept_probs = (proposed_scores - smoothed_scores) / temp + \
                               orig_log_probs[pos, proposed_tokens] - orig_log_probs[pos, current_samples[th.arange(batch_size), pos]] + \
                               th.log(reverse_proposal_probs[th.arange(batch_size), current_samples[th.arange(batch_size), pos]]) - \
                               th.log(proposal_probs[th.arange(batch_size), proposed_tokens])

            accept_mask = th.log(th.rand(batch_size, device=log_accept_probs.device)) < log_accept_probs
            current_samples[accept_mask] = proposed_samples[accept_mask]
            smoothed_scores[accept_mask] = proposed_scores[accept_mask]
            current_grads[accept_mask] = proposed_grads[accept_mask]
            current_results[accept_mask] = proposed_results[accept_mask]

            # detach & clone
            smoothed_scores = smoothed_scores.detach().clone()
            current_grads = current_grads.detach().clone()
            current_results = current_results.detach().clone()
            current_samples = current_samples.detach().clone()

            if step >= burn_in:
                results.append(current_results.detach().clone())
                scores.append(smoothed_scores.detach().clone())

            acceptance_rate += accept_mask.float().mean().item()
            total_proposals += 1

    acceptance_rate /= total_proposals

    results = th.cat(results)
    scores = th.cat(scores)
    exp_scores = th.exp(scores / temp)
    normalizing_constant = 1 / (1 / exp_scores).mean().item()
    unbiased_estimates = results * normalizing_constant / exp_scores

    return unbiased_estimates.mean().item()

