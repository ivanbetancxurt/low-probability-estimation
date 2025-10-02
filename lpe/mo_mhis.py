import torch as th
from fancy_einsum import einsum
from tqdm import tqdm

from .utils import Discrete

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
        window_size: int = 1, 
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
            onehot = th.nn.functional.one_hot(current_samples, num_classes=d_vocab).float()
            onehot.requires_grad_(True)

            x = onehot @ model.embed.W_E
            x = x + model.pos_embed(current_samples)
            for block in model.blocks:
                x = block(x)
            x = model.ln_final(x[:, -1].unsqueeze(1))
            y = model.unembed(x).squeeze(1)
            # this has dim [batch_size, ], which is the model's score for the target token for each batch 
            # however, the current_scores vector actually has no positional influence - this is one scalar per sequence
            # we want a value with a positional influence that can be smoothed over different positions 
            current_scores = y[:, target]
            current_scores.sum().backward()
            current_grads = onehot.grad.detach().clone() # this has same shape as the onehot vector, which is [batch_size, ctx_len, d_vocab]
            current_results = (y.argmax(dim=-1) == target).float()

            # then, implement smoothing across token positions
            # 1) first, we wanwt to use a function that has a per-position scalar influence (i.e. some function that varies by position) 
            # the current_scores is not appropriate for this since it's one scalar value per sequence
            # the gradient !!! has a per-position breakdown - for each token position i, the gradient tells us how sensitive the model's output is to changes in the one-hot encoding at that position
            # hence, we can use a function of the gradient, such as the L2 norm, to denote our per-position influence
            g = current_grads.norm(dim=-1) # dim: [batch_size, ctx_len], because we "squish" along the last axis, or d_vocab

            # 2) then, we can compute the local neighbor average
            # pad_scores uses replicate to copy the edge values, after padding, pad_scores has shape [batch_size, ctx_len + 2*w] (where w is the num neighbors we're using)
            # the first w columns replicate the first tok, and the last w columns represent the last tok 
            pad_g = th.nn.functional.pad(g, (window_size, window_size), mode = 'replicate') # replicate edge values, after padding, pad_scores has shape [batch_size, ctx_len + 2*w]
            avg_neighbors = th.stack([pad_g[:, i:i+2*window_size+1].mean(dim=1) for i in range(ctx_len)], dim=1) # FIG OUT DIMS OF THIS: [BATCH_SIZE, CTX_LEN]
            
            # 3) apply EMA / smoothing per-position
            ema_scores = (1 - lam) * g + lam * avg_neighbors # dim = [batch_size, ctx_len]
            smoothed_scores = ema_scores

            # 4) pick a random token position for proposal
            # we wnat to access the smoothed value (that we calculated earlier) at that position
            pos = th.randint(0, ctx_len, (batch_size,), device=current_samples.device)
            current_smoothed_at_pos = smoothed_scores[th.arange(batch_size), pos] # dim = [batch_size, ]

            # compute proposal probabilities using smoothed scores
            # for the proposal logits, we want to use the smoothed score at the position, instead of the original score
            proposal_logits = orig_log_probs[pos] + current_smoothed_at_pos.unsqueeze(-1) / temp
            proposal_probs = th.softmax(proposal_logits, dim=-1)
            proposed_tokens = th.multinomial(proposal_probs, 1).squeeze(-1)

            proposed_samples = current_samples.clone()
            proposed_samples[th.arange(batch_size), pos] = proposed_tokens

            # recompute scores and grads for proposed samples
            # this is the true model score for the proposed samples, which we will use in the acceptance probability 
            # in the acceptance probability, this can be thought of as pi(x)
            onehot = th.nn.functional.one_hot(proposed_samples, num_classes=d_vocab).float()
            onehot.requires_grad_(True)
            x = onehot @ model.embed.W_E
            x = x + model.pos_embed(proposed_samples)
            for block in model.blocks:
                x = block(x)
            x = model.ln_final(x[:, -1].unsqueeze(1))
            y = model.unembed(x).squeeze(1)

            # proposed scores is the model's score for the target token for each batch
            proposed_scores = y[:, target]
            proposed_scores.sum().backward()
            proposed_grads = onehot.grad.clone()
            # we also have to calculate a per-position proposed g
            proposed_g = proposed_grads.norm(dim = -1)
            proposed_results = (y.argmax(dim=-1) == target).float()


            # here, we build the reverse proposal distribution q(x_current | x_proposed)
            # at the proposed states (x_current), we need to use the smoothed scores 
            pad_proposed_g = th.nn.functional.pad(proposed_g, (window_size, window_size), mode='replicate')
            avg_neighbors_prop = th.stack([pad_proposed_g[:, i:i+2*window_size+1].mean(dim=1) for i in range(ctx_len)], dim=1)
            proposed_smoothed = (1 - lam) * proposed_g + lam * avg_neighbors_prop
            proposed_smoothed_at_pos = proposed_smoothed[th.arange(batch_size), pos]  # [B,]

            # calculate reverse proposal logits
            reverse_proposal_logits = orig_log_probs[pos] + proposed_smoothed_at_pos.unsqueeze(-1) / temp
            reverse_proposal_probs = th.softmax(reverse_proposal_logits, dim=-1)

            # calculating the acceptance probability
            log_pi_current = current_scores + orig_log_probs[pos, current_samples[th.arange(batch_size), pos]]
            log_pi_proposed = proposed_scores + orig_log_probs[pos, proposed_tokens]    

            log_accept_probs = (log_pi_proposed - log_pi_current) 
            + (reverse_proposal_probs[th.arange(batch_size), current_samples[th.arange(batch_size), pos]].log() 
               - proposal_probs[th.arange(batch_size), proposed_tokens].log())


            accept_mask = th.log(th.rand(batch_size, device=log_accept_probs.device)) < log_accept_probs
            
            # safe indexing to update per-position arrays: get accepted row indices
            accepted_idx = accept_mask.nonzero(as_tuple=True)[0]   # 1D tensor of accepted batch indices

            if accept_mask.any():
                rows = accept_mask.nonzero(as_tuple=True)[0]
                cols = pos[rows]

                # Update token sequences
                current_samples[rows, cols] = proposed_samples[rows, cols]

                # Update scalar sequence-level results and scores
                current_results[rows] = proposed_results[rows]
                current_scores[rows] = proposed_scores[rows]

                # Update per-position gradients
                current_grads[rows] = proposed_grads[rows]

                # Update EMA recursively for accepted positions
                ema_scores[rows, cols] = (1 - lam) * ema_scores[rows, cols] + lam * proposed_g[rows, cols]

            # Detach to break graph
            current_samples = current_samples.detach().clone()
            current_grads = current_grads.detach().clone()
            current_results = current_results.detach().clone()
            ema_scores = ema_scores.detach().clone()


            if step >= burn_in:
                results.append(current_results.detach().clone())
                scores.append(smoothed_scores.detach().clone())

            acceptance_rate += accept_mask.float().mean().item()
            total_proposals += 1

    acceptance_rate /= total_proposals

    results = th.cat(results)
    scores = th.cat(scores)
    # we want to compute sequence-levell estimates, so we average over the token positions
    exp_scores_seq = th.exp(scores.mean(dim=1) / temp)  # shape: [n_samples] 

    normalizing_constant = 1 / (1 / exp_scores_seq).mean().item()
    unbiased_estimates = results * normalizing_constant / exp_scores_seq # shape: [n_samples]

    return unbiased_estimates.mean().item()

