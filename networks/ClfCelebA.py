import torch.nn as nn

from networks.CelebAEncoderImg import FeatureExtractorImg
from networks.CelebaEncoderTxt import make_res_block_encoder_feature_extractor

class ClfImg(nn.Module):
    def __init__(self, cfg):
        super(ClfImg, self).__init__()
        self.feature_extractor = FeatureExtractorImg(cfg, a=2.0, b=0.3)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(in_features=cfg.dataset.num_layers_img*cfg.dataset.filter_dim_img, out_features=40)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_img):
        h = self.feature_extractor(x_img)
        h = self.dropout(h)
        h = h.reshape(h.size(0), -1)
        h = self.linear(h)
        out = self.sigmoid(h)
        return out

    def get_activations(self, x_img):
        h = self.feature_extractor(x_img)
        return h


class ClfText(nn.Module):
    def __init__(self, cfg):
        super(ClfText, self).__init__()
        self.conv1 = nn.Conv1d(
            cfg.dataset.num_features,
            cfg.dataset.filter_dim_text,
            kernel_size=3,
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
        )
        self.resblock_2 = make_res_block_encoder_feature_extractor(
            2 * cfg.dataset.filter_dim_text,
            3 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
        )
        self.resblock_3 = make_res_block_encoder_feature_extractor(
            3 * cfg.dataset.filter_dim_text,
            4 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
        )
        self.resblock_4 = make_res_block_encoder_feature_extractor(
            4 * cfg.dataset.filter_dim_text,
            5 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
        )
        self.resblock_5 = make_res_block_encoder_feature_extractor(
            5 * cfg.dataset.filter_dim_text,
            6 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
        )
        self.resblock_6 = make_res_block_encoder_feature_extractor(
            6 * cfg.dataset.filter_dim_text,
            7 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=0,
            dilation=1,
        )
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(
            in_features=cfg.dataset.num_layers_text * cfg.dataset.filter_dim_text,
            out_features=40,
            bias=True,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_text):
        x_text = x_text.transpose(-2, -1)
        out = self.conv1(x_text)
        out = self.resblock_1(out)
        out = self.resblock_2(out)
        out = self.resblock_3(out)
        out = self.resblock_4(out)
        out = self.resblock_5(out)
        out = self.resblock_6(out)
        h = self.dropout(out)
        h = h.view(h.size(0), -1)
        h = self.linear(h)
        # out = self.sigmoid(h)
        return h
