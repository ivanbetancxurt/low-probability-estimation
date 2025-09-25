import torch as th
from fancy_einsum import einsum
from tqdm import tqdm

from .utils import Discrete

def MT_MHIS(
        model,
        orig_dists: list[Discrete],
        target: int,
        *,
        temp: float,
        n_samples: int,
        burn_in: int,
        batch_size: int = 32,
        n_candidates: int = 4,
        show_progress: bool = False
) -> float:
    """
    Multi-Try Metropolis-Hastings Importance Sampling (MT-MHIS).
    Stationary: q(x) ∝ exp(s(x)/temp) * p(x), with MH proposal modified to propose
    n_candidates per step and select one via weighted sampling.
    """
    d_vocab = model.embed.d_vocab
    ctx_len = len(orig_dists)

    # Freeze model params
    for p in model.parameters():
        p.requires_grad_(False)

    # Base prior log-probs with -inf mask (same as MHIS)
    orig_log_probs = []
    for pos in range(ctx_len):
        mask = -th.inf * th.ones(d_vocab, device=model.device)
        mask[orig_dists[pos].values] = th.log(orig_dists[pos].probs)
        orig_log_probs.append(mask)
    orig_log_probs = th.stack(orig_log_probs)  # (ctx_len, d_vocab)

    results = []
    scores  = []

    with th.enable_grad():
        # Initialize chains
        current_samples = th.stack([dist.sample((batch_size,)) for dist in orig_dists], dim=1)  # (B, L)

        acceptance_rate = 0.0
        total_proposals = 0

        for step in tqdm(range((n_samples // batch_size + burn_in)), disable=not show_progress):
            if step == 0:
                # Forward/backward at current state
                onehot = th.nn.functional.one_hot(current_samples, num_classes=d_vocab).float()
                onehot.requires_grad_(True)

                x = onehot @ model.embed.W_E
                x = x + model.pos_embed(current_samples)
                for block in model.blocks:
                    x = block(x)
                x = model.ln_final(x[:, -1].unsqueeze(1))
                y = model.unembed(x).squeeze(1)

                current_scores = y[:, target]
                current_scores.sum().backward()
                current_scores = current_scores.detach().clone()
                current_grads  = onehot.grad.detach().clone()  # (B, L, V)

                current_results = (y.argmax(dim=-1) == target).float()

            # Pick positions per chain
            pos = th.randint(0, ctx_len, (batch_size,), device=current_samples.device)  # (B,)

            # ----- Forward multi-candidate proposal -----
            # logits/probs (B,V), then sample (B, n_candidates)
            grad_term        = current_grads[th.arange(batch_size), pos] / temp          # (B, V)
            candidate_logits = orig_log_probs[pos] + grad_term                           # (B, V)
            candidate_probs  = th.softmax(candidate_logits, dim=-1)                      # (B, V)
            candidate_tokens = th.multinomial(candidate_probs, n_candidates, replacement=True)  # (B, n_candidates)
            candidate_token_probs = candidate_probs.gather(1, candidate_tokens)          # (B, n_candidates)

            # Weighted selection of one candidate per chain
            weights = candidate_token_probs / candidate_token_probs.sum(dim=1, keepdim=True)    # (B, n_candidates)
            selected_indices   = Categorical(weights).sample()                                   # (B,)
            selected_candidates = candidate_tokens[th.arange(batch_size), selected_indices]      # (B,)
            selected_candidate_probs = candidate_token_probs[th.arange(batch_size), selected_indices]  # (B,)

            # Proposed samples
            proposed_samples = current_samples.clone()
            proposed_samples[th.arange(batch_size), pos] = selected_candidates

            # Scores/gradients at proposed
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
            proposed_grads  = onehot.grad.clone()
            proposed_results = (y.argmax(dim=-1) == target).float()

            onehot.grad = None

            # ----- Reverse multi-candidate proposal at x' (same candidates) -----
            rev_grad_term  = proposed_grads[th.arange(batch_size), pos] / temp           # (B, V)
            rev_logits     = orig_log_probs[pos] + rev_grad_term                         # (B, V)
            rev_probs      = th.softmax(rev_logits, dim=-1)                              # (B, V)
            reverse_candidate_probs = rev_probs.gather(1, candidate_tokens)              # (B, n_candidates)
            reverse_selected_probs  = reverse_candidate_probs[th.arange(batch_size), selected_indices]  # (B,)

            # Prior ratio for selected candidate
            log_prior_ratio = orig_log_probs[pos, selected_candidates] - \
                              orig_log_probs[pos, current_samples[th.arange(batch_size), pos]]        # (B,)

            # Acceptance (one per chain)
            log_accept_probs = (proposed_scores - current_scores) / temp + \
                               log_prior_ratio + \
                               th.log(reverse_selected_probs) - th.log(selected_candidate_probs)      # (B,)

            u = th.rand(batch_size, device=log_accept_probs.device)
            accept_mask = th.log(u) <= log_accept_probs                                                # (B,)

            # Commit accepted
            current_samples[accept_mask] = proposed_samples[accept_mask]
            current_scores[accept_mask]  = proposed_scores[accept_mask]
            current_grads[accept_mask]   = proposed_grads[accept_mask]
            current_results[accept_mask] = proposed_results[accept_mask]

            # Detach
            current_scores  = current_scores.detach().clone()
            current_grads   = current_grads.detach().clone()
            current_results = current_results.detach().clone()
            current_samples = current_samples.detach().clone()

            if step >= burn_in:
                results.append(current_results.detach().clone())
                scores.append(current_scores.detach().clone())

            acceptance_rate += accept_mask.float().mean().item()
            total_proposals += 1

    acceptance_rate /= max(total_proposals, 1)

    results = th.cat(results)
    scores  = th.cat(scores)
    exp_scores = th.exp(scores / temp)
    normalizing_constant = 1 / (1 / exp_scores).mean().item()
    unbiased_estimates = results * normalizing_constant / exp_scores

    return unbiased_estimates.mean().item()