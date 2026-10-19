import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10, n_layers: int = 4, max_len: int = 1024):
        super().__init__()
        self.n_tokens = n_tokens
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, max_len, d_latent))
        torch.nn.init.normal_(self.pos_embedding, std=0.02)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=4 * d_latent,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = torch.nn.LayerNorm(d_latent)
        self.head = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        x: (B, h, w) integer tokens
        return: logits (B, h, w, n_tokens) over the next token (apply softmax for probabilities;
                use cross_entropy on the logits for training), and an empty dict of extra terms.
        """
        B, h, w = x.shape
        L = h * w
        z = self.embedding(x.reshape(B, L).long())
        # shift right by one: position i sees only tokens < i, position 0 sees a zero vector
        z = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], dim=1)
        z = z + self.pos_embedding[:, :L]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        z = self.transformer(z, mask=mask, is_causal=True)
        logits = self.head(self.norm(z))
        return logits.reshape(B, h, w, self.n_tokens), {}

    @torch.no_grad()
    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Sample B token images of shape (B, h, w), one token at a time in raster order.
        """
        if device is None:
            device = next(self.parameters()).device
        L = h * w
        tokens = torch.zeros(B, L, dtype=torch.long, device=device)
        for i in range(L):
            logits, _ = self.forward(tokens.reshape(B, h, w))
            probs = torch.softmax(logits.reshape(B, L, -1)[:, i], dim=-1)
            tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
        return tokens.reshape(B, h, w)
