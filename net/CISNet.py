import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# time model
from net.utils.subject_attention import SubjectAttentionLayer

# pre-trained backbone
import torchvision.models as models

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conv_block(c_in, c_out, ks=3, sd=1, batch_norm=True):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(c_in,
                      c_out,
                      kernel_size=ks,
                      stride=sd,
                      padding=(ks - 1) // 2,
                      bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.MaxPool2d(2))
    else:
        return nn.Sequential(
            nn.Conv2d(c_in,
                      c_out,
                      kernel_size=ks,
                      stride=sd,
                      padding=(ks - 1) // 2,
                      bias=True), nn.ReLU(), nn.MaxPool2d(2))


class Model(nn.Module):
    r"""Baseline

    Args:
        num_class (int): Number of classes for the classification task
        temporal_model (str): choose from 'single', 'lstm' and 'tcn'
        backbone (str): choose from 'simple', 'resnet50', 'resnet101', 'vgg16', 'alexnet'
        hidden_size (int): hidden_size for lstm
        num_layers (int): num_layers for lstm
        num_channels (list): num_channel for tcn
        kernel_size (int): kernel_size for tcn
        batch_norm (bool): for backbone: 'simple' 
        dropout (int): for all the model
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, (T_{in}), C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, num_class)` 
          
    """
    def __init__(self,
                 num_class,
                 backbone='simple',
                 temporal_model='single',
                 hidden_size=256,
                 num_layers=2,
                 num_channels=[512, 256, 256],
                 kernel_size=2,
                 batch_norm=True,
                 dropout=0.3,
                 subject=False,
                 pooling=False,
                 d_in=512,
                 d_m=256,
                 d_out=512,
                 **kwargs):
        super().__init__()
        assert d_in == d_out

        self.backbone = backbone
        self.temporal_model = temporal_model

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.num_channels = num_channels
        self.kernel_size = kernel_size

        self.batch_norm = batch_norm
        self.dropout = dropout

        self.subject = subject
        self.pooling = pooling

        self.d_in = d_in
        self.d_m = d_m
        self.d_out = d_out

        if self.backbone == 'alexnet':
            self.encoder = nn.Sequential(
                *list(models.alexnet(
                    pretrained=False).children())[0],  # [N, 256, 6, 6]
            )
            self.output_channel = 256
            self.output_size = 6

        elif self.backbone == 'vgg16':
            self.encoder = nn.Sequential(
                *list(models.vgg16(
                    pretrained=False).children())[0],  # [N, 512, 8, 8]
            )
            self.output_channel = 512
            self.output_size = 8

        elif self.backbone == 'squeezenet':
            self.encoder = nn.Sequential(
                *list(models.squeezenet1_0(
                    pretrained=False).children())[0],  # [N, 512, 15, 15]
            )
            self.output_channel = 512
            self.output_size = 8

        elif self.backbone == 'resnet18':
            self.encoder = nn.Sequential(
                *list(models.resnet18(pretrained=False).children())
                [:-1],  # [N, 512, image_size // (2^4), _]
            )
            self.output_channel = 512
            self.output_size = 16

        elif self.backbone == 'resnet34':
            self.encoder = nn.Sequential(
                *list(models.resnet34(pretrained=False).children())
                [:-1],  # [N, 512, image_size // (2^4), _]
            )
            self.output_channel = 512
            self.output_size = 16

        elif self.backbone == 'resnet50':
            self.encoder = nn.Sequential(
                *list(models.resnet50(pretrained=False).children())
                [:-1],  # [N, 1024, image_size // (2^4), _]
            )
            self.output_channel = 2048
            self.output_size = 16

        if self.temporal_model == 'single':
            pass
        elif self.temporal_model == 'lstm':
            self.lstm = nn.LSTM(input_size=self.output_channel,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True,
                                dropout=self.dropout)
            self.final = nn.Sequential(
                nn.Linear(self.hidden_size, num_class),
                nn.Sigmoid(),
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # subject related
        if self.subject:
            if self.pooling == False:
                self.projector = nn.Linear(
                    self.output_channel * self.output_size * self.output_size,
                    self.d_in)
                self.subject_attention = SubjectAttentionLayer(
                    self.d_in, self.d_m, self.d_out)
                self.final = nn.Sequential(
                    nn.Linear(self.d_out, 64),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(64, num_class),
                    nn.Sigmoid(),
                )
            else:
                self.projector = nn.Linear(self.output_channel, self.d_in)
                self.subject_attention = SubjectAttentionLayer(
                    self.d_in, self.d_m, self.d_out)
                self.final = nn.Sequential(
                    nn.Linear(self.d_out, 64),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(64, num_class),
                    nn.Sigmoid(),
                )
        else:
            if self.pooling == False:
                self.final = nn.Sequential(
                    nn.Linear(
                        self.output_channel * self.output_size *
                        self.output_size, 64),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(64, num_class),
                    nn.Sigmoid(),
                )
            else:
                self.final = nn.Sequential(
                    nn.Linear(self.output_channel, 64),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(64, num_class),
                    nn.Sigmoid(),
                )

    def forward(self, image, subject_infos=None):
        '''
        image for cnn: [N, C, H, W] if single
                        [N, T, C, H, W] if sequential model (time_model is set)
        '''
        N, T, C, H, W = image.shape
        x = image.view(-1, image.shape[2], image.shape[3], image.shape[4])

        x = self.encoder(x)
        if self.pooling == False:
            x = x.view(x.shape[0], -1)
        else:
            x = self.avgpool(x)
            x = x.view(x.shape[0], -1)

        if self.subject:
            x = self.projector(x)
            feature = x
            if subject_infos:
                x = self.subject_attention(x, subject_infos)
        else:
            feature = x

        if self.temporal_model == 'single':
            pass
        elif self.temporal_model == 'lstm':
            x = x.view(N, -1, self.output_channel)
            h0 = torch.zeros(self.num_layers, x.size(0),
                             self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0),
                             self.hidden_size).to(device)
            self.lstm.flatten_parameters()
            x, _ = self.lstm(x, (h0, c0))  # [N, T, self.hidden_size]
            x = x[:, -1, :]  # N x self.hidden_size

        output = self.final(x)

        return feature, output