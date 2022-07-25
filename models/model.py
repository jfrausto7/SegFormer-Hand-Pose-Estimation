import torch
from torch import Tensor, nn
from einops import repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        emb_size: int = 768,
        img_size: int = 224,
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
        self.positions = nn.Parameter(
            torch.randn((img_size // patch_size) ** 2 + 1, emb_size)
        )

    def forward(self, x: Tensor):
        (b,) = x.shape
        x = self.projection(x)
        # utilize class tokens and prepend to input
        class_tokens = repeat(self.class_token, "() n e -> b n e", b=b)
        x = torch.cat([class_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x


class EncoderBlock(nn.Sequential):
    """
    Transformer Encoder block proposed in `Attention Is All You Need`_
    
    Link to original paper: https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        emb_size: int = 768,
        drop_p: float = 0.0,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.0,
        **kwargs,
    ):
        super().__init__()


class MultiHeadAttention(nn.Module):
    """
    Multi head attention proposed in `Attention Is All You Need`

    Link to original paper: https://arxiv.org/abs/1706.03762
    """
    def __init__(self, emb_size: int = 512, num_heads: int = 8, dropout: float = 0):
        super().__init__()


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__()

class TransformerEncoder(nn.Sequential):
    """
    Transformer Encoder proposed in `Attention Is All You Need`. The ViT architecture
    only uses the Encoder, which is why the decoder is omitted. 

    Link to original paper: https://arxiv.org/abs/1706.03762
    """

    def __init__(self, depth: int = 12, **kwargs):
        super().__init__()


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
