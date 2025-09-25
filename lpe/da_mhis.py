import torch as th
from fancy_einsum import einsum
from tqdm import tqdm

from .utils import Discrete

def DA_MHIS(
    model,
    orig_dists: list[Discrete],
    target: int,
    *,
    temp: float,
    n_samples: int,
    burn_in: int,
    batch_size: int = 32,
    show_progress: bool = False
) -> float:
    """
    Delayed-Acceptance Metropolis-Hastings Importance Sampling (DA-MHIS).
    Stationary: q(x) ∝ exp(M_t(x)/temp) * p(x), with MH proposal identical to MHIS.
    Stage-1: cheap surrogate screen using Δ~M (linearized via current grads).
    Stage-2: exact correction using full forward at x' and reverse proposal at x'.
    """
    import torch as th
    d_vocab = model.embed.d_vocab
    ctx_len = len(orig_dists)

    # Freeze model params
    for p in model.parameters():
        p.requires_grad_(False)

    # Build per-position base prior log-probs with -inf mask (same as your MHIS)
    orig_log_probs = []
    for pos in range(ctx_len):
        mask = -th.inf * th.ones(d_vocab, device=model.device)
        mask[orig_dists[pos].values] = th.log(orig_dists[pos].probs)
        orig_log_probs.append(mask)
    orig_log_probs = th.stack(orig_log_probs)  # (ctx_len, d_vocab)

    results = []
    scores  = []

    stage1_pass_total = 0.0
    stage1_trials     = 0.0
    acceptance_rate   = 0.0
    total_proposals   = 0

    with th.enable_grad():
        # ---- init batch of chains from p(x)
        current_samples = th.stack([dist.sample((batch_size,)) for dist in orig_dists], dim=1)  # (B, L)

        # forward/backward at current state (collect scores, grads, results)
        onehot = th.nn.functional.one_hot(current_samples, num_classes=d_vocab).float()
        onehot.requires_grad_(True)

        x = onehot @ model.embed.W_E
        x = x + model.pos_embed(current_samples)
        for block in model.blocks:
            x = block(x)
        x = model.ln_final(x[:, -1].unsqueeze(1))
        y = model.unembed(x).squeeze(1)
        current_scores = y[:, target]  # (B,)
        current_scores.sum().backward()
        current_grads   = onehot.grad.detach().clone()      # (B, L, V)
        current_results = (y.argmax(dim=-1) == target).float()

        onehot.grad = None
        current_scores = current_scores.detach().clone()
        current_samples = current_samples.detach().clone()
        current_results = current_results.detach().clone()

        # ---- main loop
        for step in tqdm(range((n_samples // batch_size + burn_in)), disable=not show_progress):
            # 2a. pick positions per chain
            pos = th.randint(0, ctx_len, (batch_size,), device=current_samples.device)  # (B,)

            # 2b–2e. forward proposal logits (same as MHIS)
            # current_grads[:, pos, :] acts like d_v = g_i^T e(v)
            proposal_logits = orig_log_probs[pos] + current_grads[th.arange(batch_size), pos] / temp  # (B, V)
            proposal_log_probs = th.log_softmax(proposal_logits, dim=-1)  # use log-softmax for stability
            proposed_tokens = th.multinomial(proposal_log_probs.exp(), 1).squeeze(-1)               # (B,)

            # Prepare candidate x' (do NOT commit)
            proposed_samples = current_samples.clone()
            proposed_samples[th.arange(batch_size), pos] = proposed_tokens

            # -------- Stage-1: cheap DA screen (no forward at x') --------
            log_phi_forward = proposal_log_probs[th.arange(batch_size), proposed_tokens]  # log φ_i(v|x)

            d_v   = current_grads[th.arange(batch_size), pos, proposed_tokens]
            d_x_i = current_grads[th.arange(batch_size), pos, current_samples[th.arange(batch_size), pos]]
            delta_tildeM = d_v - d_x_i  # surrogate logit change

            log_prior_ratio = orig_log_probs[pos, proposed_tokens] - \
                              orig_log_probs[pos, current_samples[th.arange(batch_size), pos]]

            log_r1 = log_prior_ratio + (delta_tildeM / temp)

            # Accept Stage-1 in log-space
            u1 = th.rand(batch_size, device=log_r1.device)
            stage1_mask = (log_r1 >= 0) | (th.log(u1) <= log_r1)   # (B,) bool

            stage1_pass_total += stage1_mask.float().mean().item()
            stage1_trials     += 1.0

            accept_mask = th.zeros(batch_size, dtype=th.bool, device=current_samples.device)  # init

            if stage1_mask.any():
                # ----- Heavy work ONLY for those that passed Stage-1 -----
                idx = stage1_mask.nonzero(as_tuple=False).squeeze(-1)       # indices of passers
                pos_pass = pos[idx]

                # A) exact target change at x' (1 forward)
                onehot = th.nn.functional.one_hot(proposed_samples[idx], num_classes=d_vocab).float()
                onehot.requires_grad_(True)

                x = onehot @ model.embed.W_E
                x = x + model.pos_embed(proposed_samples[idx])
                for block in model.blocks:
                    x = block(x)
                x = model.ln_final(x[:, -1].unsqueeze(1))
                y = model.unembed(x).squeeze(1)

                proposed_scores_pass = y[:, target]                     # (P,)
                proposed_results_pass = (y.argmax(dim=-1) == target).float()

                # backprop at x' to build reverse proposal
                proposed_scores_pass.sum().backward()
                proposed_grads_pass = onehot.grad.clone()               # (P, L, V)
                onehot.grad = None

                # reverse proposal log-probs at same position i
                rev_logits_pass = orig_log_probs[pos_pass] + \
                                  proposed_grads_pass[th.arange(idx.numel(), device=pos.device), pos_pass] / temp  # (P, V)
                rev_log_probs_pass = th.log_softmax(rev_logits_pass, dim=-1)
                old_tokens_pass = current_samples[idx, pos_pass]
                log_phi_reverse_pass = rev_log_probs_pass[th.arange(idx.numel(), device=pos.device), old_tokens_pass]

                # B) DA correction acceptance (Stage-2)
                delta_M_pass = proposed_scores_pass - current_scores[idx]
                log_r_full_pass = log_prior_ratio[idx] + (delta_M_pass / temp) + \
                                  (log_phi_reverse_pass - log_phi_forward[idx])
                log_r2_pass = log_r_full_pass - log_r1[idx]

                u2 = th.rand(idx.numel(), device=log_r2_pass.device)
                accept_pass = (log_r2_pass >= 0) | (th.log(u2) <= log_r2_pass)

                # commit accepted passers
                if accept_pass.any():
                    idx_acc = idx[accept_pass]

                    accept_mask[idx_acc] = True
                    current_samples[idx_acc] = proposed_samples[idx_acc]
                    current_scores[idx_acc]  = proposed_scores_pass[accept_pass]
                    current_grads[idx_acc]   = proposed_grads_pass[accept_pass]
                    current_results[idx_acc] = proposed_results_pass[accept_pass]

            # detach/clamp state tensors (like your MHIS)
            current_scores  = current_scores.detach().clone()
            current_grads   = current_grads.detach().clone()
            current_results = current_results.detach().clone()
            current_samples = current_samples.detach().clone()

            # collect samples after burn-in
            if step >= burn_in:
                results.append(current_results.detach().clone())
                scores.append(current_scores.detach().clone())

            acceptance_rate += accept_mask.float().mean().item()  # Stage-2 accept rate
            total_proposals += 1

    acceptance_rate /= max(total_proposals, 1)
    # (Optional) stage1_pass_rate = stage1_pass_total / max(stage1_trials, 1)

    # IS estimator (unchanged from your MHIS)
    results = th.cat(results)  # (n_samples,)
    scores  = th.cat(scores)   # (n_samples,)
    exp_scores = th.exp(scores / temp)
    normalizing_constant = 1 / (1 / exp_scores).mean().item()
    unbiased_estimates = results * normalizing_constant / exp_scores

    return unbiased_estimates.mean().item()