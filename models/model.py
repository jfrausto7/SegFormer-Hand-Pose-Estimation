from torch import nn


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
        out_channels: int = 50 ** kwargs,
    ):
        super().__init__()
