import sys
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score

import torch

# PyTorch >= 2.6 flips torch.load's default to weights_only=True. The
# pretrained classifier checkpoints loaded below (ClfPolyMNIST/ClfCelebA/
# ClfscMNC) were all saved via LightningModule.save_hyperparameters(cfg),
# which pickles the whole Hydra cfg -- an omegaconf.DictConfig -- into the
# checkpoint. DictConfig isn't in torch's default weights_only-safe allowlist,
# so loading fails on newer torch even though this repo pins torch==2.2.0
# (where weights_only defaulted to False). pytorch_lightning==2.1.4's own
# checkpoint loader doesn't expose a weights_only kwarg through the public
# load_from_checkpoint API, so we patch the default here instead. Safe as
# long as the checkpoints being loaded are ones you trust (here: your own
# training runs), matching option (1) in torch's own error message.
_torch_load = torch.load


def _torch_load_full(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_full

from clfs.polymnist_clf import ClfPolyMNIST
from clfs.celeba_clf import ClfCelebA
from clfs.scMNC_clf import ClfscMNC


def _to_np(tensor):
    # Under precision="bf16-mixed" (see run_experiment.py), encoder/decoder/
    # classifier outputs are bfloat16 tensors. Neither NumPy nor scikit-learn
    # support that dtype, so anything crossing that boundary needs an explicit
    # float32 cast first -- letting sklearn's own numpy.asarray(tensor) bridge
    # do it raises "Got unsupported ScalarType BFloat16".
    return tensor.detach().float().cpu().numpy()


def train_clf_lr_PM(encodings, labels):
    clf = LogisticRegression(max_iter=10000).fit(_to_np(encodings), labels.cpu())
    return clf


def eval_clf_lr_PM(clf, encodings, labels):
    y_pred = clf.predict(_to_np(encodings))
    acc = accuracy_score(labels.cpu(), y_pred)
    return np.array(acc)


def train_clf_lr_scMNC(encodings, labels):
    clf = LogisticRegression(max_iter=10000).fit(_to_np(encodings), labels.cpu())
    return clf


def eval_clf_lr_scMNC(clf, encodings, labels):
    y_pred = clf.predict(_to_np(encodings))
    acc = accuracy_score(labels.cpu(), y_pred)
    return np.array(acc)


def train_clf_lr_celeba(encodings, labels):
    n_labels = labels.shape[1]
    clfs = []
    encodings_np = _to_np(encodings)
    for k in range(0, n_labels):
        clf = LogisticRegression(max_iter=10000).fit(
            encodings_np, labels[:, k].cpu()
        )
        clfs.append(clf)
    return clfs


def eval_clf_lr_celeba(clfs, encodings, labels):
    n_labels = labels.shape[1]
    scores = torch.zeros(n_labels)
    encodings_np = _to_np(encodings)
    for k in range(0, n_labels):
        clf_k = clfs[k]
        y_pred_k = clf_k.predict(encodings_np)
        ap = average_precision_score(labels[:, k].cpu(), y_pred_k)
        scores[k] = ap
    return scores

def generate_samples(decoders, rep):
    imgs_gen = []
    for dec in decoders:
        img_gen = dec(rep)
        imgs_gen.append(img_gen[0])
    return imgs_gen


def conditional_generation(mvvae, dists):
    imgs_gen = []
    for idx, dist in enumerate(dists):
        mu, lv = dist
        imgs_gen_dist = []
        for m in range(len(mvvae.decoders)):
            z_out = mvvae.reparametrize(mu, lv)
            # cond_gen_m = mvvae.cond_generate_samples(m, z_out)[0]
            cond_gen_m = mvvae.decoders[m](z_out)[0]
            imgs_gen_dist.append(cond_gen_m)
        imgs_gen.append(imgs_gen_dist)
    return imgs_gen
  
def conditional_generation_cov(mvvae, dists):
    imgs_gen = []
    for idx, dist in enumerate(dists):
        mu, lv = dist
        imgs_gen_dist = []
        for m in range(len(mvvae.decoders)):
            z_out = mvvae.reparametrize(mu, lv)
            cond_gen_m = mvvae.cond_generate_samples_cov(idx, m, z_out)[0]
            imgs_gen_dist.append(cond_gen_m)
        imgs_gen.append(imgs_gen_dist)
    return imgs_gen


def load_modality_clfs(cfg):
    if cfg.dataset.name.startswith("PM"):
        model = load_modality_clfs_PM(cfg)
    elif cfg.dataset.name.startswith("celeba"):
        model = load_modality_clfs_celeba(cfg)
    elif cfg.dataset.name.startswith("sc"):
        model = load_modality_clfs_scMNC(cfg)
    else:
        print("dataset does not exist..exit")
        sys.exit()
    return model


def load_modality_clfs_PM(cfg):
    fp_clf = os.path.join(
        cfg.dataset.dir_clfs_base, cfg.dataset.suffix_clfs, "last.ckpt"
    )
    model = ClfPolyMNIST.load_from_checkpoint(fp_clf)
    return model


def load_modality_clfs_celeba(cfg):
    fp_clf = os.path.join(cfg.dataset.dir_clf, "last.ckpt")
    model = ClfCelebA.load_from_checkpoint(fp_clf)
    return model
  
def load_modality_clfs_scMNC(cfg):
    fp_clf = os.path.join(cfg.dataset.dir_clf, "last.ckpt")
    model = ClfscMNC.load_from_checkpoint(fp_clf)
    return model


def calc_coherence_acc(cfg, clf, imgs, labels):
    out_clf = clf(cfg, [imgs, labels])
    preds = out_clf[0]
    return preds


def from_preds_to_acc(preds, labels, modality_names):
    n_views = len(modality_names)
    accs = torch.zeros((n_views, n_views, 1))
    for m, m_key in enumerate(modality_names):
        for m_tilde, m_tilde_key in enumerate(modality_names):
            preds_m_mtilde = preds[:, m, m_tilde, :]
            acc_m_mtilde = accuracy_score(
                labels.cpu(),
                np.argmax(_to_np(preds_m_mtilde), axis=1).astype(int),
            )
            accs[m, m_tilde, 0] = acc_m_mtilde
    return accs


def from_preds_to_ap(preds, labels, modality_names):
    n_views = len(modality_names)
    n_labels = labels.shape[1]
    aps = torch.zeros((n_views, n_views, n_labels))
    for m, m_key in enumerate(modality_names):
        for m_tilde, m_tilde_key in enumerate(modality_names):
            preds_m_mtilde = preds[:, m, m_tilde, :]
            for k in range(0, n_labels):
                ap_m_mtilde_k = average_precision_score(
                    labels[:, k].cpu(), _to_np(preds_m_mtilde[:, k])
                )
                aps[m, m_tilde, k] = ap_m_mtilde_k
    return aps


def calc_coherence_ap(cfg, clf, mods, labels):
    out_clf = clf(cfg, [mods, labels])
    preds = out_clf[0]
    return preds
