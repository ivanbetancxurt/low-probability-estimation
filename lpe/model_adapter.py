class ModelAdapter:
    def __init__(self, hf_model, tokenizer, device=None):
        self.hf = hf_model.eval()
        self.tokenizer = tokenizer
        self.device = device or next(hf_model.parameters()).device
        # embed
        class _Embed:
            def __init__(self, w): self.W_E = w
            @property
            def d_vocab(self): return self.W_E.shape[0]
            
        self.embed = _Embed(self.hf.transformer.wte.weight)           # [V, D]
        # pos_embed: lookup positions 0..L-1 from wpe
        def pos_embed(tokens):
            B, L = tokens.shape
            pos = th.arange(L, device=tokens.device)[None, :].expand(B, L)
            return self.hf.transformer.wpe(pos)                        # [B, L, D]

        self.pos_embed = pos_embed
        # blocks: wrap full transformer forward on inputs_embeds
        class _BlockWrapper:
            def __init__(self, hf): self.hf = hf
            def __call__(self, x):  # x: [B,L,D]
                out = self.hf.transformer(inputs_embeds=x, use_cache=False).last_hidden_state
                return out
        
        self.blocks = [_BlockWrapper(self.hf)]                        # single “block” does full stack
        # ln_final: GPT-2 already applies ln_f; use identity
        self.ln_final = lambda x: x
        # unembed
        class _Unembed:
            def __init__(self, w): self.W_U = w.T                     # [D, V]
            def __call__(self, x): return x @ self.W_U[None]          # [B,L,V]
        self.unembed = _Unembed(self.hf.lm_head.weight)
        # parameters() for freezing
        def _params(): 
            for p in self.hf.parameters(): yield p
        self.parameters = _params