import torch.nn as nn
import torch 

from networks.CelebaEncoderTxt import FeatureExtractorText
from networks.CelebADecoderTxt import DataGeneratorText
from networks.CelebAEncoderImg import FeatureExtractorImg
from networks.CelebADecoderImg import DataGeneratorImg

# used in the encoder for celeba - this is a simple linear layer that compresses the features extracted by the feature extractor to the latent dimension, without any residual connections
class LinearFeatureCompressor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearFeatureCompressor, self).__init__()
        self.mu = nn.Linear(in_channels, out_channels, bias=False)
        self.logvar = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, feats):
        feats = feats.view(feats.size(0), -1)
        mu, logvar = self.mu(feats), self.logvar(feats)
        return mu, logvar


class EncoderText(nn.Module):
    def __init__(self, cfg):
        super(EncoderText, self).__init__()
        self.feature_extractor = FeatureExtractorText(
            cfg,
            a=cfg.dataset.skip_connections_text_weight_a,
            b=cfg.dataset.skip_connections_text_weight_b,
        )
        self.feature_compressor = LinearFeatureCompressor(
            5 * cfg.dataset.filter_dim_text, cfg.model.latent_dim
        )

    def forward(self, x_text):
        h_text = self.feature_extractor(x_text)
        mu, logvar = self.feature_compressor(h_text)
        # return mu, logvar, h_text;
        return mu, logvar


class DecoderText(nn.Module):
    def __init__(self, cfg):
        super(DecoderText, self).__init__()
        self.feature_generator = nn.Linear(
            cfg.model.latent_dim, 5 * cfg.dataset.filter_dim_text, bias=True
        )
        self.text_generator = DataGeneratorText(
            cfg,
            a=cfg.dataset.skip_connections_text_weight_a,
            b=cfg.dataset.skip_connections_text_weight_b,
        )

    def forward(self, z):
        text_feat_hat = self.feature_generator(z)
        text_feat_hat = text_feat_hat.unsqueeze(-1)
        text_hat = self.text_generator(text_feat_hat)
        text_hat = text_hat.transpose(-2, -1)
        return [text_hat]


class EncoderImg(nn.Module):
    def __init__(self, cfg):
        super(EncoderImg, self).__init__()
        self.feature_extractor = FeatureExtractorImg(
            cfg,
            a=cfg.dataset.skip_connections_img_weight_a,
            b=cfg.dataset.skip_connections_img_weight_b,
        )
        # the feature compressor takes the features extracted by the feature extractor and compresses them to the latent dimension - this is a simple linear layer without any residual connections
        self.feature_compressor = LinearFeatureCompressor(
            cfg.dataset.num_layers_img * cfg.dataset.filter_dim_img,
            cfg.model.latent_dim,
        )

    def forward(self, x_img):
        h_img = self.feature_extractor(x_img)
        h_img = h_img.view(h_img.shape[0], h_img.shape[1], h_img.shape[2])
        mu, logvar = self.feature_compressor(h_img)
        return mu, logvar


class DecoderImg(nn.Module):
    def __init__(self, cfg):
        super(DecoderImg, self).__init__()
        self.feature_generator = nn.Linear(
            cfg.model.latent_dim,
            cfg.dataset.num_layers_img * cfg.dataset.filter_dim_img,
            bias=True,
        )
        self.img_generator = DataGeneratorImg(
            cfg,
            a=cfg.dataset.skip_connections_img_weight_a,
            b=cfg.dataset.skip_connections_img_weight_b,
        )

    def forward(self, z):
        img_feat_hat = self.feature_generator(z)
        img_feat_hat = img_feat_hat.view(
            img_feat_hat.size(0), img_feat_hat.size(1), 1, 1
        )
        img_hat = self.img_generator(img_feat_hat)
        return img_hat, 0.75