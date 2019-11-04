import torch


class UnifiedGPT2MediumConfig:
    vocab_size = 50265
    n_positions = 1024
    n_ctx = 1024
    n_embd = 1024
    n_layer = 24
    n_head = 16
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02
    gradient_checkpointing = True


class UnifiedGPT2SmallConfig:
    vocab_size = 50265
    n_positions = 1024
    n_ctx = 1024
    n_embd = 768
    n_layer = 12
    n_head = 12
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02
    gradient_checkpointing = False