import torch
from torch import nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        head_size: int,
        dropout: float = 0.1,
        causal: bool = False,
        max_context: int = 256,
    ) -> None:
        """
        Class for a single self-attention head.

        Args:
            n_inputs (int): number of input features, typically the dimension of
                the input embeddings.
            head_size (int): number of output features for this attention head.
                Typically the same as `n_inputs` in models like GPT and BERT.
            dropout (float, optional): dropout probability applied to the
                attention weights for regularization. Defaults to 0.1.
            causal (bool, optional): if True, the attention mechanism will be
                causal, meaning each token will only attend to previous tokens
                and itself. This is typically used for autoregressive models.
                Defaults to False.
            max_context (int, optional): maximum sequence length supported by
                the model. Defaults to 256.
        """
        super().__init__()

        self.query = nn.Linear(n_inputs, head_size, bias=False)
        self.key = nn.Linear(n_inputs, head_size, bias=False)
        self.value = nn.Linear(n_inputs, head_size, bias=False)
        self.causal = causal
        if causal:
            self.register_buffer(
                "mask", ~torch.tril(torch.ones(max_context, max_context)).bool()
            )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, C = x.shape

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Compute the scaled dot product between the query of each token and
        # the keys of all tokens
        q_k = query @ key.transpose(-1, -2) * C**-0.5
        if self.causal:
            q_k = q_k.masked_fill(self.mask[:T, :T], float("-inf"))
        q_k_weights = torch.softmax(q_k, dim=-1)
        q_k_weights = self.attn_dropout(q_k_weights)
        output = q_k_weights @ value
        return output


class MultiHeadAttention(nn.Module):
    def __init__(
        self, n_inputs, n_heads, dropout=0.1, causal=False, max_context=256
    ) -> None:
        """
        Class for multi-head attention.

        Args:
            n_inputs (int): number of input features, typically the dimension of
                the input embeddings.
            n_heads (int): number of attention heads. The size of each head is
                calculated as `n_inputs // n_heads`. Ensure that `n_inputs` is
                divisible by `n_heads`.
            dropout (float, optional): dropout probability applied to the
                output of the multi-head attention for regularization.
                Defaults to 0.1.
            causal (bool, optional): if True, the attention mechanism will be
                causal, meaning each token will only attend to previous tokens
                and itself. This is typically used for autoregressive models.
                Defaults to False.
            max_context (int, optional): maximum sequence length supported by
                the model. Defaults to 256.
        """
        super().__init__()
        head_size = n_inputs // n_heads
        assert head_size * n_heads == n_inputs, "Wrong dimensions!"

        self.att_heads = nn.ModuleList(
            [
                AttentionHead(
                    n_inputs,
                    head_size,
                    dropout,
                    causal,
                    max_context,
                )
                for _ in range(n_heads)
            ]
        )
        self.lin = nn.Linear(n_inputs, n_inputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.att_heads], dim=-1)
        x = self.lin(x)
        output = self.dropout(x)
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self, n_inputs, n_heads, dropout=0.1, causal=False, max_context=256
    ) -> None:
        """
        Class for a transformer block with a multi-head attention followed by a
        feed-forward network. Both components are preceded by pre-layer
        normalization, and dropout is applied to the outputs of each.

        Args:
            n_inputs (int): number of input features, typically the dimension of
                the input embeddings.
            n_heads (int): number of attention heads. The size of each head is
                calculated as `n_inputs // n_heads`. Ensure that `n_inputs` is
                divisible by `n_heads`.
            dropout (float, optional): dropout probability applied to the output
                of the multi-head attention and the feed-forward network for
                regularization. Defaults to 0.1.
            causal (bool, optional): if True, the attention mechanism will be
                causal, meaning each token will only attend to previous tokens
                and itself. This is typically used for autoregressive models.
                Defaults to False.
            max_context (int, optional): maximum sequence length supported by
                the model. Defaults to 256.

        """
        super().__init__()

        self.mha = MultiHeadAttention(
            n_inputs, n_heads, dropout, causal, max_context
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(n_inputs, 4 * n_inputs),
            nn.GELU(),
            nn.Linear(4 * n_inputs, n_inputs),
            nn.Dropout(dropout),
        )
        self.layer_norm_1 = nn.LayerNorm(n_inputs)
        self.layer_norm_2 = nn.LayerNorm(n_inputs)

    def forward(self, x):
        # pre-layer normalization
        x = self.mha(self.layer_norm_1(x)) + x
        output = self.feed_forward(self.layer_norm_2(x)) + x
        return output


class GPT(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_heads,
        n_tokens,
        n_embed,
        dropout=0.1,
        max_context=256,
    ) -> None:
        """
        A class that implements a simplified version of GPT.

        Args:
            n_blocks (int): number of transformer blocks in the model.
            n_heads (int): number of attention heads. The size of each head is
                calculated as `n_embed // n_heads`. Ensure that `n_embed` is
                divisible by `n_heads`.
            n_tokens (int): vocabulary size, i.e., the total number of unique
                tokens that the model should expect.
            n_embed (int): size of embeddings for token and positional
                embeddings.
            dropout (float, optional): dropout probability applied to the output
                of the multi-head attention and the feed-forward network in the
                transformer blocks. Defaults to 0.1.
            max_context (int, optional): maximum sequence length supported by
                the model. Defaults to 256.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(n_tokens, n_embed)
        self.position_embedding = nn.Embedding(max_context, n_embed)

        # We want to always set causal=True for GPT
        self.blocks = nn.Sequential(
            *(
                TransformerBlock(
                    n_embed,
                    n_heads,
                    dropout,
                    causal=True,
                    max_context=max_context,
                )
                for _ in range(n_blocks)
            )
        )
        self.layer_norm = nn.LayerNorm(n_embed)
        self.output = nn.Linear(n_embed, n_tokens)

    def forward(self, x):
        _, T = x.shape
        token_emb = self.token_embedding(x)
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.output(x)
        return logits
