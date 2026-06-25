import torch.nn as nn

# Residual block
class ResidualBlock1dConv(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        kernelsize,
        stride,
        padding,
        dilation,
        downsample,
        a=1.0,
        b=1.0,
    ):
        super(ResidualBlock1dConv, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in)
        self.conv1 = nn.Conv1d(
            channels_in, channels_in, kernel_size=1, stride=1, padding=0
        )
        self.dropout1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(channels_in)
        self.conv2 = nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size=kernelsize,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.dropout2 = nn.Dropout(p=0.5)
        self.downsample = downsample
        self.a = a
        self.b = b

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = self.a * residual + self.b * out
        return out


def make_res_block_encoder_feature_extractor(
    in_channels,
    out_channels,
    kernelsize,
    stride,
    padding,
    dilation,
    a_val=1.0,
    b_val=1.0,
):
    downsample = None
    if (stride != 1) or (in_channels != out_channels) or dilation != 1:
        downsample = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernelsize,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
        )
    layers = []
    layers.append(
        ResidualBlock1dConv(
            in_channels,
            out_channels,
            kernelsize,
            stride,
            padding,
            dilation,
            downsample,
            a=a_val,
            b=b_val,
        )
    )
    return nn.Sequential(*layers)


class FeatureExtractorText(nn.Module):
    def __init__(self, cfg, a, b):
        super(FeatureExtractorText, self).__init__()
        self.a = a
        self.b = b
        self.conv1 = nn.Conv1d(
            cfg.dataset.num_features,
            cfg.dataset.filter_dim_text,
            kernel_size=4,
            stride=2,
            padding=1,
            dilation=1,
        )
        self.resblock_1 = make_res_block_encoder_feature_extractor(
            cfg.dataset.filter_dim_text,
            2 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=a,
            b_val=b,
        )
        self.resblock_2 = make_res_block_encoder_feature_extractor(
            2 * cfg.dataset.filter_dim_text,
            3 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=a,
            b_val=b,
        )
        self.resblock_3 = make_res_block_encoder_feature_extractor(
            3 * cfg.dataset.filter_dim_text,
            4 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=a,
            b_val=b,
        )
        self.resblock_4 = make_res_block_encoder_feature_extractor(
            4 * cfg.dataset.filter_dim_text,
            5 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=a,
            b_val=b,
        )
        self.resblock_5 = make_res_block_encoder_feature_extractor(
            5 * cfg.dataset.filter_dim_text,
            5 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=a,
            b_val=b,
        )
        self.resblock_6 = make_res_block_encoder_feature_extractor(
            5 * cfg.dataset.filter_dim_text,
            5 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=0,
            dilation=1,
            a_val=a,
            b_val=b,
        )

    def forward(self, x):
        x = x.transpose(-2, -1)
        out = self.conv1(x)
        out = self.resblock_1(out)
        out = self.resblock_2(out)
        out = self.resblock_3(out)
        out = self.resblock_4(out)
        out = self.resblock_5(out)
        out = self.resblock_6(out)
        return out





# used in the encoder for celeba - this is a simple linear layer that compresses the features extracted by the feature extractor to the latent dimension, without any residual connections
class LinearFeatureCompressor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearFeatureCompressor, self).__init__()
        self.mu = nn.Linear(in_channels, out_channels, bias=False)
        self.logvar = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, feats):
        feats = feats.reshape(feats.size(0), -1)
        mu, logvar = self.mu(feats), self.logvar(feats)
        return mu, logvar