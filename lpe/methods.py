import torch as th
from fancy_einsum import einsum
from tqdm import tqdm

from .utils import Discrete


def random_constraint_projection(
    P: th.tensor, biases: th.tensor, n_reps=1000, tol=1e-9, factor=0.99
):
    """
    Finds shortest point in the polytope defined by P @ x + biases >= 0.
    This method isn't very airtight, but it was what we used in our experiments.
    """
    n, d = P.shape
    device = P.device

    x = th.zeros(d, device=device)

    for _ in range(n_reps * 100):
        constraints = P @ x + biases
        # This should be be -tol, but we used tol in the official experiments
        unsatisfied = constraints < tol

        # Randomly choose an unsatisfied constraint
        # This will crash if there are no unsatisfied constraints?
        unsatisfied_indices = th.where(unsatisfied)[0]
        random_index = unsatisfied_indices[th.randint(len(unsatisfied_indices), (1,))]

        # Project onto the chosen constraint
        p_i = P[random_index]

        projection = ((-constraints[random_index]) / th.sum(p_i**2)) * p_i
        x += projection.squeeze()

        # Check if we found a feasible point
        if th.all(P @ x + biases >= 0):
            if _ < n_reps:
                x *= factor
            else:
                return x

    assert False, "Did not find a feasible point"


def QLD(
    W_U: th.Tensor, act_samps: th.Tensor, target: int, *, batch_size: int = 512
) -> float:
    """
    Quadratic Logit Decomposition. Estimates the probability of outputing token `target` by using the Quadratic Logit Decomposition with the Shortest Accepting Vector as `d'.
    Inputs:
    - W_U: the unembedding matrix of the model (d_model x d_vocab).
    - act_samps: the samples of the activations right before unembedding (n_samples x d_model).
    - target: the target token (in range [0...d_model)).
    Returns:
    - The estimated probability of outputing token `target`.
    """

    d_model = W_U.shape[0]
    d_vocab = W_U.shape[1]
    n = act_samps.shape[0]
    assert target < d_vocab, "Target token out of range"
    assert act_samps.shape == (n, d_model), "act_samps has incorrect shape"

    # whiten the activations.
    n = act_samps.shape[0]
    mean = act_samps.mean(dim=0)
    cov = (act_samps - mean).T @ (act_samps - mean) / n
    EPS = 1e-5
    A = th.linalg.cholesky(
        cov + EPS * th.eye(d_model, device=cov.device)
    )  # (d_model, d_model). We have cov + EPS*I == A @ A.T.
    u_samps = act_samps - mean @ th.inverse(A.T)  # (n, d_model), the whitened samples.

    # Given whitened samples z of shape (n, d_model), the logits are given by z @ pullbacks.T + biases. Every row of the logits is a sample.
    pullbacks = W_U.T @ A  # (d_vocab, d_model)
    biases = W_U.T @ mean  # (d_vocab, ).

    # find the shortest accepting vector.
    pullbacks_diff = pullbacks[target].unsqueeze(0) - pullbacks
    biases_diff = biases[target].unsqueeze(0) - biases
    biases_diff[target] = 100
    d = random_constraint_projection(
        pullbacks_diff, biases_diff, n_reps=200, factor=0.95
    )  # (d_model, )
    d = d / th.norm(d)

    a_samps = (u_samps @ d).sort().values
    all_probs = []

    assert n % batch_size == 0
    for y in th.split(u_samps, batch_size):
        b_samps = y - y @ th.outer(d, d)  # (batch_size, d_model)

        # figure out the lower and upper bounds on the last direction.
        # Let z be b_samps. For a particular sample i, we need that for all j != t=token, we have:
        #  z_i @ pullbacks[t].T + biases[t] > z_i @ pullbacks[j].T + biases[j].
        # If we let z_i = a_i + r_i d, where r_i is a scalar, then we need that:
        #  a_i @ (pullbacks[t] - pullbacks[j]).T + biases[t] - biases[j] > -r_i d @ (pullbacks[t] - pullbacks[j]).T
        #  {a_i @ (pullbacks[t] - pullbacks[j]).T + biases[t] - biases[j]} / -{d @ (pullbacks[t] - pullbacks[j]).T} > r_i   (possibly with a sign flip)

        pullbacks_diff = (pullbacks[target].unsqueeze(0) - pullbacks).mT
        numerator = y @ pullbacks_diff + biases[target] - biases
        denominator = -d @ pullbacks_diff
        lower = (
            th.where(denominator < 0, numerator / denominator, -th.inf).max(-1).values
        )  # (batch_size, )
        upper = (
            th.where(denominator > 0, numerator / denominator, th.inf).min(-1).values
        )  # (batch_size, )

        # find how many latents were between upper and lower
        all_probs.append(
            th.maximum(
                th.searchsorted(a_samps, upper) - th.searchsorted(a_samps, lower),
                th.tensor(0),
            )
            / n  # (batch_size, )
        )

    all_probs = th.cat(all_probs)  # (n, )
    return all_probs.mean().item()


def GLD(
    W_U: th.Tensor, act_samps: th.Tensor, target: int, *, batch_size: int = 512
) -> float:
    """
    Gaussian Logit Difference. Finds parameters of the normal distribution fit to the target logit minus the maximum logit.
    Inputs:
    - W_U: the unembedding matrix of the model (d_model x d_vocab).
    - act_samps: the samples of the activations right before unembedding (n_samples x d_model).
    - target: the target token (in range [0...d_model)).
    Returns:
    - mu, sigma: The mean and variance of the logit differnce. Note mu <= 0.
    """

    argmax = []
    # Use batches to avoid OOM
    for batch in act_samps.split(batch_size, dim=0):
        logits = batch @ W_U
        argmax.append(logits.argmax(dim=1))
    argmax = th.cat(argmax)

    max_samps = einsum("b x, x b -> b", act_samps, W_U[:, argmax])
    target_samps = act_samps @ W_U[:, target]

    mu = (target_samps - max_samps).mean().item()
    sigma = (target_samps - max_samps).std().item()

    return mu, sigma


def ITGIS(
    model,
    orig_dists: list[Discrete],
    target: int,
    *,
    temp: float,
    n_samples: int,
    batch_size: int = 256,
    decay_rate: float = 0.9,
    show_progress: bool = False
) -> float:
    """
    Independent Token Gradient Importance Sampling. Uses the gradient of the logit with respect to the token embedding to define a new importance sampling distribution (with all tokens still being independent). Adaptively updates the importance sampling distribution based on samples from the previous.
    Inputs:
    - model: the transformer.
    - orig_dists: list of Discrete distributions for each token position.
    - target: the target token (in range [0...d_vocab)).
    - temp: the temperature.
    - n_samples: the number of samples to be drawn.
    - batch_size: the batch size.
    - decay_rate: the decay rate in the exponentially-weighted moving average of gradients.
    Returns:
    - The estimated probability of outputing token `target`.
    """

    d_vocab = model.embed.d_vocab
    ctx_len = len(orig_dists)
    scores = th.zeros((ctx_len, d_vocab), device=model.device)

    for param in model.parameters():
        param.requires_grad_(False)

    imp_samp_probs = []
    assert n_samples % batch_size == 0
    for i in tqdm(list(range(n_samples // batch_size)), disable=not show_progress):
        target_samples = []
        target_logratios = []
        adj_temp = (
            temp * (1 - decay_rate**i) / (1 - decay_rate) if i > 0 else 1
        )  # adjust temperature for exponential moving average
        for dist, scores_at_pos in zip(orig_dists, scores):
            samples_at_pos = dist.boltzmann_distribution(
                scores=scores_at_pos[dist.values], temperature=adj_temp
            ).sample((batch_size,))
            target_samples.append(samples_at_pos)
            target_logratios.append(
                th.logsumexp(
                    scores_at_pos[dist.values] / adj_temp + th.log(dist.probs), dim=0
                )
                - scores_at_pos[samples_at_pos] / adj_temp
            )

        samples = th.stack(target_samples, dim=1)  # (batch_size, ctx_len)
        logratios = th.stack(target_logratios, dim=1)  # (batch_size, ctx_len)

        with th.enable_grad():
            onehot = th.nn.functional.one_hot(
                samples, num_classes=d_vocab
            ).float()  # (batch_size, ctx_len, d_vocab)
            onehot.requires_grad_(True)

            x = (
                onehot @ model.embed.W_E
            )  # (batch_size, context_len, d_model) @ (d_model, d_vocab) -> (batch_size, context_len, d_vocab)
            x = x + model.pos_embed(samples)
            for block in model.blocks:
                x = block(x)
            x = model.ln_final(x)[:, -1, :]
            y = model.unembed(x[:, None, :])  # (batch_size, 1, d_vocab)

            probs = (y[:, -1, :].argmax(-1) == target).float().detach()
            imp_samp_probs.append((th.exp(logratios.sum(-1)) * probs))

            (
                x @ model.unembed.W_U[:, target]
            ).sum().backward()  # for some reason, this is faster than y[:,-1,target].sum().backward():
            scores *= decay_rate
            scores += onehot.grad.sum(0) / batch_size

    imp_samp_probs = th.cat(imp_samp_probs, dim=0)
    return imp_samp_probs.mean().item()

def MHIS(
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
    Metropolis-Hastings Importance Sampling. Takes batch_size independent random walks in token space, with q(x) \propto exp(logit / temp) * p(x) as the stationary distribution.
    We use the proposal function phi(x'|x) defined by:
    - Choose a random token position i
    - Take the gradient of s(x) with respect to the embedding of token i. Dot that with the different tokens y you could replace token i with, and take probabilities proportional to p(y) exp(grad \cdot y / temp).
    Inputs:
    - model: the transformer.
    - orig_dists: list of Discrete distributions for each token position.
    - target: the target token (in range [0...d_vocab)).
    - temp: the temperature (for both the Boltzmann stationary distribution and proposal distribution)
    - n_samples: the total number of samples to be drawn, ignoring burn-in.
    - batch_size: the number of parallel random walks to run (the total number of samples drawn is n_samples + burn_in * batch_size)
    Returns:
    - The estimated probability of outputing token `target`.
    """
    d_vocab = model.embed.d_vocab #* vocabulary size
    ctx_len = len(orig_dists) #* length of the input sequence (context length)
    #scores = th.zeros((ctx_len, d_vocab), device=model.device) 

    #* we're not training the model so freeze the parameters
    for param in model.parameters():
        param.requires_grad_(False)

    ## compute the log probabilities of the original input distribution, which is our baseline p(x) 
    ## the mask ensures that only valid tokens have a nonnegative probabiity, while setting invalid tokens to -inf 
    orig_log_probs = []
    for pos in range(ctx_len): #* for every distribution position in the context...
        mask = -th.inf * th.ones(d_vocab, device=model.device) #* make a vector of length vocabulary-size of -inf (disallowed tokens)
        ## the valid tokens = orig_dists[pos].values
        ## valid token probabilities = orig_dists[pos].probs. here, we take the log-prob
        mask[orig_dists[pos].values] = th.log(orig_dists[pos].probs) #* at the positions of the allowed token, fill in that vector with that token's log prob
        orig_log_probs.append(mask) #* add that vector to orig_log_probs list
    orig_log_probs = th.stack(orig_log_probs) #* convert the orig_log_probs list into a (ctx_len, d_vocab) matrix

    #for param in model.parameters():
    #    param.requires_grad_(False)

    results = []
    scores = []

    with th.enable_grad():
        # Initialize the first batch of samples
        current_samples = th.stack([dist.sample((batch_size,)) for dist in orig_dists], dim=1) #* sample batch_size input sequences as a matrix of shape (batch_size, ctx_len), so each value is a token ID

        acceptance_rate = 0
        total_proposals = 0

        for step in tqdm(range((n_samples // batch_size + burn_in)), disable=not show_progress):

            if step == 0: #* on the first step...
                
                #* turn the input sequences matrix into 3D onehot tensor of shape (batch_size, ctx_len, d_vocab)
                ## turning the input sequence into a one-hot encoding allows us to treat these input tokens as continuous, and hence take the gradient with respect to them
                onehot = th.nn.functional.one_hot(current_samples, num_classes=d_vocab).float()
                onehot.requires_grad_(True)

                #* Perform forward pass through the model for the current samples
                x = onehot @ model.embed.W_E                #* multiply one-hot encoded tokens by the embedding matrix to get token embeddings                  
                x = x + model.pos_embed(current_samples)    #* add positional encoding to embeddings (pos_embed is definied in components.py within utils)

                #* Then, pass token embeddings through transformer layers to capture context from the entire sequence.
                for block in model.blocks:
                    x = block(x)                         #* Each block processes embeddings and captures the relationships between tokens.
                x = model.ln_final(x[:,-1].unsqueeze(1)) #* apply final layer normalization to last token
                ## the resulting x is indexed by [:, -1], and can be thought of as the contextual information of the last token position
                
                #* logits for each token (our raw predictions)
                y = model.unembed(x).squeeze(1) #* convert embeddings back to raw logits for token predictions
                current_scores = y[:, target]   #* extract the logit that corresponds to the target token

                #* Computes gradients of the target logit with respect to our input
                current_scores.sum().backward()                     #* 1) perform backpropogation
                current_scores = current_scores.detach().clone()    #* 2) Detatch gradients from the computation graph
                current_grads = onehot.grad.detach().clone()        #* 3) store gradients so we cna use them later.

                #* Checks if the model correctly predicted target token
                current_results = (y.argmax(dim=-1) == target).float()

            #* pick a random position to make a token proposal
            pos = th.randint(0, ctx_len, (batch_size,), device=current_samples.device)
            
            # Compute proposal probabilities
            proposal_logits = orig_log_probs[pos] + current_grads[th.arange(batch_size), pos] / temp
            proposal_probs = th.softmax(proposal_logits, dim=-1) #* Adding gradients to logits and converting to probabilities
            
            # Propose new tokens #* (sample new tokens from the proposal distribution)
            proposed_tokens = th.multinomial(proposal_probs, 1).squeeze(-1)

            # Create proposed samples
            proposed_samples = current_samples.clone()
            proposed_samples[th.arange(batch_size), pos] = proposed_tokens #* Replace selected positions with the new proposed tokens

            # Recompute scores and gradients for proposed samples. No difference in the following section than in the first step
            onehot = th.nn.functional.one_hot(proposed_samples, num_classes=d_vocab).float()
            onehot.requires_grad_(True)                 
            x = onehot @ model.embed.W_E                
            x = x + model.pos_embed(proposed_samples)  

            for block in model.blocks:
                x = block(x)
            x = model.ln_final(x[:,-1].unsqueeze(1)) 
            y = model.unembed(x).squeeze(1)
            proposed_scores = y[:, target]
            proposed_scores.sum().backward()
            proposed_grads = onehot.grad.clone()
            proposed_results = (y.argmax(dim=-1) == target).float()

            # Clear gradients
            onehot.grad = None

            # Compute reverse proposal probabilities
            reverse_proposal_logits = orig_log_probs[pos] + proposed_grads[th.arange(batch_size), pos] / temp
            reverse_proposal_probs = th.softmax(reverse_proposal_logits, dim=-1) #* reverse proposal probabilities = adjusting logits w/ gradients

            # Compute log acceptance probabilities using proposed and current scores, and proposal probabilities
            log_accept_probs = (proposed_scores - current_scores) / temp + \
                               orig_log_probs[pos, proposed_tokens] - orig_log_probs[pos, current_samples[th.arange(batch_size), pos]] + \
                               th.log(reverse_proposal_probs[th.arange(batch_size), current_samples[th.arange(batch_size), pos]]) - \
                               th.log(proposal_probs[th.arange(batch_size), proposed_tokens])

            # Accept or reject proposals
            accept_mask = th.log(th.rand(batch_size, device=log_accept_probs.device)) < log_accept_probs #* Randomly accept/reject proposals based on log acceptance probabilities
            current_samples[accept_mask] = proposed_samples[accept_mask] #* Accept proposed samples if acceptance test passed

            #* Accept proposed scores, gradients, and results if proposal is accepted.
            current_scores[accept_mask] = proposed_scores[accept_mask]                                   
            current_grads[accept_mask] = proposed_grads[accept_mask]
            current_results[accept_mask] = proposed_results[accept_mask]

            #* Detach current values so we don't do furthur gradient tracking. i.e. detach & clone scores, gradients, results, and current samples.
            current_scores = current_scores.detach().clone()
            current_grads = current_grads.detach().clone()
            current_results = current_results.detach().clone()
            current_samples = current_samples.detach().clone()
            
            #* record results after "burn-in period", basically during the initial burn in period, the model is exploring the token space but we don't consider those samples to be reliable for statistics or estimations. 
            #* We're discarding samples that could have been influenced by random initializaiton or transient states, and we're ensuring the Markov chain is stable. 
            if step >= burn_in:
                results.append(current_results.detach().clone())
                scores.append(current_scores.detach().clone())
            
            #* Keep tracking the acceptance rate and & of proposals
            acceptance_rate += accept_mask.float().mean().item()
            total_proposals += 1

    acceptance_rate /= total_proposals

    results = th.cat(results)       # list of results (stored in batches) concatenated into a single tensor
    scores = th.cat(scores)         # list of scores ""
    exp_scores = th.exp(scores / temp) # Exponentiate the scores to transform them into probabilities
    normalizing_constant = 1 / (1 / exp_scores).mean().item()  # E_p[exp(s(x)/temp)] -> just the normalizing constant
    unbiased_estimates = results * normalizing_constant / exp_scores    # use normalizing constant ti give us our final unbiased estimates.

    return unbiased_estimates.mean().item()

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

def MTM_MHIS(
        
)