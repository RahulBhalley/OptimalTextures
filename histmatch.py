import numpy as np
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps')


def hist_match(target, source, mode="pca", eps=1e-2):
    target = target.permute(0, 3, 1, 2)  # -> b, c, h, w
    source = source.permute(0, 3, 1, 2)

    if mode == "cdf":
        b, c, h, w = target.shape
        matched = cdf_match(target.reshape(c, -1), source.reshape(c, -1)).reshape(b, c, h, w)

    else:
        # based on https://github.com/ProGamerGov/Neural-Tools/blob/master/linear-color-transfer.py#L36

        mu_t = target.mean((2, 3), keepdim=True)
        hist_t = (target - mu_t).view(target.size(1), -1)  # [c, b * h * w]
        cov_t = hist_t @ hist_t.T / hist_t.shape[1] + eps * torch.eye(hist_t.shape[0], device=device)

        mu_s = source.mean((2, 3), keepdim=True)
        hist_s = (source - mu_s).view(source.size(1), -1)
        cov_s = hist_s @ hist_s.T / hist_s.shape[1] + eps * torch.eye(hist_s.shape[0], device=device)

        if mode == "chol":
            chol_t = torch.linalg.cholesky(cov_t)
            chol_s = torch.linalg.cholesky(cov_s)
            matched = chol_s @ torch.inverse(chol_t) @ hist_t

        elif mode == "pca":
            # eva_t, eve_t = torch.symeig(cov_t, eigenvectors=True, upper=True)
            eva_t, eve_t = torch.linalg.eigh(cov_t, UPLO='U')
            Qt = eve_t @ torch.sqrt(torch.diag(eva_t)) @ eve_t.T
            # eva_s, eve_s = torch.symeig(cov_s, eigenvectors=True, upper=True)
            eva_s, eve_s = torch.linalg.eigh(cov_s, UPLO='U')
            Qs = eve_s @ torch.sqrt(torch.diag(eva_s)) @ eve_s.T
            matched = Qs @ torch.inverse(Qt) @ hist_t

        elif mode == "sym":
            # eva_t, eve_t = torch.symeig(cov_t, eigenvectors=True, upper=True)
            eva_t, eve_t = torch.linalg.eigh(cov_t, UPLO='U')
            Qt = eve_t @ torch.sqrt(torch.diag(eva_t)) @ eve_t.T
            Qt_Cs_Qt = Qt @ cov_s @ Qt
            # eva_QtCsQt, eve_QtCsQt = torch.symeig(Qt_Cs_Qt, eigenvectors=True, upper=True)
            eva_QtCsQt, eve_QtCsQt = torch.linalg.eigh(Qt_Cs_Qt, UPLO='U')
            QtCsQt = eve_QtCsQt @ torch.sqrt(torch.diag(eva_QtCsQt)) @ eve_QtCsQt.T
            matched = torch.inverse(Qt) @ QtCsQt @ torch.inverse(Qt) @ hist_t

        matched = matched.view(*target.shape) + mu_s

    return matched.permute(0, 2, 3, 1)  # -> b, h, w, c


# def hist_match(target, source, mode="pca", eps=1e-2):
#     target = target.permute(0, 3, 1, 2)  # -> b, c, h, w
#     source = source.permute(0, 3, 1, 2)

#     if mode == "cdf":
#         b, c, h, w = target.shape
#         matched = cdf_match(target.reshape(c, -1), source.reshape(c, -1)).reshape(b, c, h, w)

#     else:
#         # based on https://github.com/ProGamerGov/Neural-Tools/blob/master/linear-color-transfer.py#L36

#         mu_t = target.mean((2, 3), keepdim=True)
#         hist_t = (target - mu_t).view(target.size(1), -1)  # [c, b * h * w]
#         cov_t = hist_t @ hist_t.T / hist_t.shape[1] + eps * torch.eye(hist_t.shape[0], device=target.device)

#         mu_s = source.mean((2, 3), keepdim=True)
#         hist_s = (source - mu_s).view(source.size(1), -1)
#         cov_s = hist_s @ hist_s.T / hist_s.shape[1] + eps * torch.eye(hist_s.shape[0], device=target.device)

#         if mode == "chol":
#             chol_t = torch.linalg.cholesky(cov_t)
#             chol_s = torch.linalg.cholesky(cov_s)
#             matched = torch.linalg.solve(chol_t, chol_s @ hist_t)

#         elif mode == "pca":
#             eva_t, eve_t = torch.linalg.eig(cov_t)
#             eva_t = eva_t.real
#             eve_t = eve_t.real
#             Qt = eve_t @ torch.sqrt(torch.diag(eva_t)) @ eve_t.T
#             eva_s, eve_s = torch.linalg.eig(cov_s)
#             eva_s = eva_s.real
#             eve_s = eve_s.real
#             Qs = eve_s @ torch.sqrt(torch.diag(eva_s)) @ eve_s.T
#             Qt_inv = torch.inverse(Qt)
#             matched = Qs @ Qt_inv @ hist_t

#         elif mode == "sym":
#             # NOT OPTIMIZED AFTER.
            
#             # # eva_t, eve_t = torch.symeig(cov_t, eigenvectors=True, upper=True)
#             # eva_t, eve_t = torch.linalg.eigh(cov_t, UPLO='U')
#             # Qt = eve_t @ torch.sqrt(torch.diag(eva_t)) @ eve_t.T
#             # Qt_Cs_Qt = Qt @ cov_s @ Qt
#             # # eva_QtCsQt, eve_QtCsQt = torch.symeig(Qt_Cs_Qt, eigenvectors=True, upper=True)
#             # eva_QtCsQt, eve_QtCsQt = torch.linalg.eigh(Qt_Cs_Qt, UPLO='U')
#             # QtCsQt = eve_QtCsQt @ torch.sqrt(torch.diag(eva_QtCsQt)) @ eve_QtCsQt.T
#             # matched = torch.inverse(Qt) @ QtCsQt @ torch.inverse(Qt) @ hist_t

#             # Tried to optimize.
#             eva_t, eve_t = torch.linalg.eigh(cov_t, UPLO='U')
#             Qt = torch.bmm(eve_t, torch.sqrt(torch.diag_embed(eva_t)))
#             Qt = torch.bmm(Qt, eve_t.transpose(1,2))
#             Qt_Cs_Qt = torch.bmm(Qt, cov_s)
#             Qt_Cs_Qt = torch.bmm(Qt_Cs_Qt, Qt.transpose(1,2))
#             eva_QtCsQt, eve_QtCsQt = torch.linalg.eigh(Qt_Cs_Qt, UPLO='U')
#             QtCsQt = torch.bmm(eve_QtCsQt, torch.sqrt(torch.diag_embed(eva_QtCsQt)))
#             QtCsQt = torch.bmm(QtCsQt, eve_QtCsQt.transpose(1,2))
#             inv_Qt = torch.inverse(Qt)
#             matched = torch.bmm(inv_Qt, hist_t)
#             matched = torch.bmm(QtCsQt, matched)
#             matched = torch.bmm(inv_Qt, matched)

#         matched = matched.view(*target.shape) + mu_s

#     return matched.permute(0, 2, 3, 1)  # -> b, h, w, c

def cdf_match(target, source, bins=256):
    matched = torch.empty_like(target)
    for i, (target_channel, source_channel) in enumerate(zip(target.contiguous(), source)):
        lo = torch.min(target_channel.min(), source_channel.min())
        hi = torch.max(target_channel.max(), source_channel.max())

        # TODO find batched method of getting histogram? maybe based on numpy's impl?
        # https://github.com/numpy/numpy/blob/v1.20.0/numpy/lib/histograms.py#L678
        target_hist = torch.histc(target_channel, bins, lo, hi)
        source_hist = torch.histc(source_channel, bins, lo, hi)
        bin_edges = torch.linspace(lo, hi, bins + 1, device=device)[1:]

        target_cdf = target_hist.cumsum(0)
        target_cdf = target_cdf / target_cdf[-1]

        source_cdf = source_hist.cumsum(0)
        source_cdf = source_cdf / source_cdf[-1]

        remapped_cdf = interp(target_cdf, source_cdf, bin_edges)
        matched[i] = interp(target_channel, bin_edges, remapped_cdf)
    return matched


def interp(x, xp, fp):
    # based on https://github.com/numpy/numpy/blob/main/numpy/core/src/multiarray/compiled_base.c#L489

    f = torch.zeros_like(x)

    idxs = torch.searchsorted(xp, x)
    idxs_next = (idxs + 1).clamp(0, len(xp) - 1)

    slopes = (fp[idxs_next] - fp[idxs]) / (xp[idxs_next] - xp[idxs])
    f = slopes * (x - xp[idxs]) + fp[idxs]

    infinite = ~torch.isfinite(f)
    if infinite.any():
        inf_idxs_next = idxs_next[infinite]
        f[infinite] = slopes[infinite] * (x[infinite] - xp[inf_idxs_next]) + fp[inf_idxs_next]

        still_infinite = ~torch.isfinite(f)
        if still_infinite.any():
            f[still_infinite] = fp[idxs[still_infinite]]

    return f