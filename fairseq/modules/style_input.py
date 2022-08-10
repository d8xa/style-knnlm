import torch
from torch import nn

import logging
logger = logging.getLogger(__name__)

class StyleInput(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.dim = input_dim
        self.embedding_dim = output_dim

        self.embeddings = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False)
        )

        def init_weights(m):
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def forward(self, input: torch.Tensor):
        return self.embeddings(input)