### Overview of the folders and content within this repo

LPE > Utils > behavior.py
- Main dataclass is @behavior, which encodes the probability that a model outputs a token in 'output_tokens' given the input tokens?
- Input distributions are a list of discrete distributions, which determine which input distribution to sample from
  - Input tokens are drawn independently from these distributions, but we can sample from repeated input distributions
  - rep_range is a (start, end) tuple that specifies a **slice of the input distribution that can be repeated**
  - n_reps is how many repetitions of that particular distribution we want. It takes the repeatable slice from rep_range **up to n_reps elements**
  - For example, if we have input_dists_distinct = [D0, D1, D2, D3, D4, D5, D6, D7], and rep_range (2, 6), then the repeatable distribution slice would be [D2, D3, D4, D5]. We can observe three cases:
    - n_reps is within rep_range, ex. n_reps = 3. then, we take the first three elements of rep_range, and the final input distribution is as such: [D0, D1] + [D2, D3, D4] + [D6, D7]
    - n_reps is exactly rep_range, if, n_reps = 4. then, the final input distribution is exactly [D0, D1] + [D2, D3, D4, D5] + [D6, D7]
    - n_reps is greater than rep_range. in this case, the input distribution is **padded by the last distribution in rep_range**. for example, if n_reps = 6, then the final input distribution is given by [D0, D1] + [D2, D3, D4, D5, D5, D5] + [D6, D7] 
- Properties: rep_start, rep_end normalize rep_range to be within positive indices, n_nonreps defines the number of distributions that are not repeated
- Function **input_dists** creates this final list of input distributions, based on the fixed distributions + number of repeated distributions
- Function **sample** samples a given number of tokens for each input distribution from the sequence, returning a tensor with shape [n_samples, sequence_length]
  - This effectively simulates an input sequence that a model might receive
- Function **constant_dist_maker** creates deterministic token distributions
  - This is useful because each input sequence is usually comprised of a partially-fixed distribution, which is combined with the repetitive, randomly sampled segments that we generate with the **sample** function
  - The resulting input sequences are families that have systematically varying lengths and tokens (because they are sampled from different distributions), but with certain fixed prefixes/suffixes that are consistent across all input sequences       
- Class **NoRepsBehavior** restricts class **Behavior** to fixed-length input sequences 

  
LPE > Utils > components.py

This file defines and allows us to edit neural network components (re-implementing standard parts of a transformer model)  

1. Embed: maps token IDs (integers) into embedding vectors
- d_model is the dimension of the model, which dictates the size of each token's vector representation
- W_E is the embedding matrix that has shape [vocab_size, d_model]
- vocab_size is the number of distinct tokens in the vocabulary
- Each row of W_E is a vector representation of a token in the vocab
2. Unembed: takes model states back into logits for prediction of next token
3. FinalSoftmax: converting logits to probabilities based on temperature
- If temperature is 0.0, return a one-hot encoding vector of the maximum (this is a deterministic prediction)
- Otherwise, return a probability over logits


LPE > Utils > datasets.py
- Code for loading in datasets, as well as caching function outputs to JSON (in order to prevent having to recompute)
- Saves function outputs locally
- get_token_dataset gets the dataset and tokenizes it, and then wraps it with beginning/end tokens, then caches the dataset for future access



LPE > Utils > Distributions

This paper uses 8 input distributions: 
- hex (hexadecimal)
- camel (camelcase Python tokens)
- colon (Python tokens ending with :)
- if (Python tokens starting with 'if')
- caps ("He/She screamed:" + caps and punctuation)
- english (English words)
- spanish (Spanish tokens)
- icl (in-context learning prompt) 
