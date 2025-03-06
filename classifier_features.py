import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from tsai.models.InceptionTime import InceptionTime, InceptionBlock
from tsai.models.MLP import MLP
from tsai.models.FCN import FCN
from tsai.models.ResNet import ResNet


class InceptionBlockModule(nn.Module):
    def __init__(self, c_in, nf=32, depth=6, **kwargs):
        super().__init__()
        # Instantiate the inception block using the same parameters
        self.inception_block = InceptionBlock(c_in, nf=nf, depth=depth, **kwargs)

        #             (convs): ModuleList(
        #           (0): Conv1d(32, 32, kernel_size=(39,), stride=(1,), padding=(19,), bias=False)
        #           (1): Conv1d(32, 32, kernel_size=(19,), stride=(1,), padding=(9,), bias=False)
        #           (2): Conv1d(32, 32, kernel_size=(9,), stride=(1,), padding=(4,), bias=False)
        #         )
        self.depth = depth
        self.kernel_sizes = [39, 3] * 6
        self.strides = [1, 1] * 6
        self.paddings = [19, 1] * 6

    def forward(self, x):
        return self.inception_block(x)

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings


def incpetionTime_base_architecture_generator(pretrained=False, weight_path=None, in_channels=1, n_pred_classes=2, device='cuda',
                                              **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        weight_path (str): where to load the pretrained weights
        in_channels (int): number of input channels in the time series
        n_pred_classes (int): number of output class
    """

    model = InceptionTime(in_channels, n_pred_classes)
    # print(weight_path)
    if pretrained:
        state_dict = torch.load(weight_path, map_location=torch.device(device))
        model.load_state_dict(state_dict, strict=False)

    # extracted_block = model.inceptionblock

    # Option 2: Create a new module and load the weights from the pretrained inceptionBlock
    extracted_block = InceptionBlockModule(c_in=in_channels, nf=32, **kwargs)
    extracted_block.inception_block.load_state_dict(model.inceptionblock.state_dict())

    return extracted_block
