### Overview of the folders and content within this repo

LPE > Utils > behavior.py
- Main dataclass is @behavior, which encodes the probability that a model outputs a token in 'output_tokens' given the input tokens?
- Input distributions are a list of discrete distributions, which determine which input distribution to sample from
  - Input tokens are drawn independently from these distributions, but we can sample from repeated input distributions
  - rep_range is a (start, end) tuple that specifies a **slice of the input distribution that can be repeated**
  - n_reps is how many repetitions of that particular distribution we want. It takes the repeatable slice from rep_range **up to n_reps elements**
  - For example, if we have input_dists_distinct = [D0, D1, D2, D3, D4, D5, D6, D7], and rep_range (2, 6), then the repeatable distribution slice would be [D2, D3, D4, D5]. We can observe three cases:
    - n_reps is within rep_range, ex. n_reps = 3. then, we take the first three elements of rep_range, and the final input distribution is as such: [D0, D1] + [D2, D3, D4] + [D6, D7]
    - n_reps is exactly rep_range, here, n_reps = 4. then, the final input distribution is exactly [D0, D1] + [D2, D3, D4, D5]  
- Properties: rep_start, rep_end normalize rep_range to be within positive indices, n_nonreps defines the number of distributions that are not repeated
- Function **def input_dists** 

  


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
