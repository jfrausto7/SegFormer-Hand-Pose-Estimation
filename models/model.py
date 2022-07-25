import torch
from torch import Tensor, nn
from einops import repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(
        self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768,
    ):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break down each image into patches and flatten
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange("b e (h) (w) -> b (h w) e"),  # pylint: disable=syntax-error
        )
        # class token
        self.class_token = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x: Tensor):
        (b,) = x.shape
        x = self.projection(x)
        # utilize class tokens and prepend to input
        class_tokens = repeat(self.class_token, "() n e -> b n e", b=b)
        x = torch.cat([class_tokens, x], dim=1)
        return x


class ViT(nn.Sequential):
    """
    Implementation of Vision Transformer (ViT) proposed in 
    `An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale`.

    This model architecture is repurposed/redesigned here for the purpose of 2D
    Hand Pose Estimation.
    
    Link to original paper: https://arxiv.org/pdf/2010.11929.pdf_
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        emb_size: int = 768,
        img_size: int = 224,
        depth: int = 12,
        out_channels: int = 50,
        **kwargs,
    ):
        super().__init__()
