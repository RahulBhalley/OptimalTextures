import argparse
from typing import List, Optional

import torch
from kornia.color.hls import hls_to_rgb, rgb_to_hls
from torch import Tensor
from torch.nn.functional import interpolate

from histmatch import hist_match
from util import get_iters_and_sizes, get_size, load_styles, maybe_load_content, resize, save_image, to_nchw, to_nhwc
from vgg import Decoder, Encoder


class OptimalTexture(torch.nn.Module):
    def __init__(
        self,
        size: int = 512,
        iters: int = 500,
        passes: int = 5,
        hist_mode: str = "chol",
        color_transfer: Optional[str] = None,
        content_strength: float = 0.1,
        style_scale: float = 1,
        mixing_alpha: float = 0.5,
        no_pca: bool = False,
        no_multires: bool = False,
    ):
        super().__init__()

        self.hist_mode = hist_mode
        self.color_transfer = color_transfer
        self.content_strength = content_strength
        self.style_scale = style_scale
        self.mixing_alpha = mixing_alpha
        self.use_pca = not no_pca

        # get number of iterations and sizes for optization
        self.passes = passes
        self.iters_per_pass_and_layer, self.sizes = get_iters_and_sizes(size, iters, passes, not no_multires)

        self.encoders = torch.nn.ModuleList([Encoder(l) for l in range(5, 0, -1)])
        self.decoders = torch.nn.ModuleList([Decoder(l) for l in range(5, 0, -1)])

    def encode_inputs(self, pastiche: Tensor, styles: List[Tensor], content: Optional[Tensor], size: int):
        # ensure pastiche, styles, and content are the correct size
        if pastiche.shape[-2] != size and pastiche.shape[-1] != size:
            style_tens = [resize(s, size=get_size(size, self.style_scale, s.shape[2], s.shape[3])) for s in styles]
            if content is not None:
                cont_size = get_size(size, 1.0, content.shape[2], content.shape[3], oversize=True)
                cont_tens = resize(content, size=cont_size)
            else:
                cont_size = (size, size)
                cont_tens = None
            pastiche = resize(pastiche, size=cont_size)
        else:
            style_tens = styles
            cont_tens = content

        # encode inputs to VGG feature space
        style_features, style_eigvs, content_features = [], [], []
        for l, encoder in enumerate(self.encoders):
            style_features.append(torch.cat([encoder(style) for style in style_tens]))  # encode styles

            if self.use_pca:
                style_features[l], eigvecs = fit_pca(style_features[l])  # PCA
                style_eigvs.append(eigvecs)
            else:
                eigvecs = torch.empty(())  # please torch.jit

            if cont_tens is not None:
                content_feature = encoder(cont_tens)
                if self.use_pca:  # project into style PC space
                    content_feature = content_feature @ eigvecs
                # center features at mean of style features
                content_feature = content_feature - content_feature.mean() + torch.mean(style_features[l])
                content_features.append(content_feature)

        return pastiche, style_features, style_eigvs, content_features

    def forward(
        self,
        pastiche: Tensor,
        styles: List[Tensor],
        content: Optional[Tensor] = None,
        verbose: bool = False,
    ):
        for p in range(self.passes):
            if verbose:
                print(f"Pass {p}, size {self.sizes[p]}")

            # get style and content target features
            pastiche, style_features, style_eigvs, content_features = self.encode_inputs(
                pastiche, styles, content, self.sizes[p]
            )

            if len(styles) > 1:
                mixing_mask = torch.ceil(
                    torch.rand(style_features[1].shape[1:3], device=pastiche.device) - self.mixing_alpha
                )[None, None, ...]
                style_features = mix_style_features(style_features, mixing_mask, self.mixing_alpha, self.hist_mode)

            for l, (encoder, decoder) in enumerate(zip(self.encoders, self.decoders)):
                if verbose:
                    print(f"Layer: relu{(4 - l) + 1}_1")

                pastiche_feature = encoder(pastiche)  # encode layer to VGG feature space

                if self.use_pca:
                    pastiche_feature = pastiche_feature @ style_eigvs[l]  # project onto principal components

                rotations = random_rotations(
                    self.iters_per_pass_and_layer[p][l - 1],
                    pastiche_feature.shape[-1],
                    device=pastiche_feature.device,
                    dtype=pastiche_feature.dtype,
                )
                for rotation in rotations:
                    pastiche_feature = optimal_transport(pastiche_feature, style_features[l], self.hist_mode, rotation)

                    if len(content_features) > 0 and l <= 2:  # apply content matching step
                        strength = self.content_strength / 2 ** (4 - l)  # 1, 2, or 4 depending on feature depth
                        pastiche_feature += strength * (content_features[l] - pastiche_feature)

                if self.use_pca:
                    pastiche_feature = pastiche_feature @ style_eigvs[l].T  # reverse principal component projection

                pastiche = decoder(pastiche_feature)  # decode back to image space

        if self.color_transfer is not None:
            assert content is not None, "Color transfer requires content image"
            target_hls = rgb_to_hls(content)
            target_hls[:, 1] = rgb_to_hls(pastiche)[:, 1]  # swap lightness channel
            target = hls_to_rgb(target_hls)

            if self.color_transfer == "opt":
                pastiche, target = to_nhwc(pastiche), to_nhwc(target)
                for _ in range(3):
                    pastiche = optimal_transport(pastiche, target, "cdf")
                pastiche = to_nchw(pastiche)

            elif self.color_transfer == "lum":
                pastiche = target  # return pastiche with hue and saturation from content

        return pastiche


def resolve_device(requested: Optional[str] = None, size: int = 512) -> str:
    if requested is not None:
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return requested

    if torch.backends.mps.is_available() and size >= 256:
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset is None:
        return args

    presets = {
        "fast": {"passes": 1, "iters": 80, "no_multires": True, "hist_mode": "chol"},
        "balanced": {"passes": 3, "iters": 250, "no_multires": False, "hist_mode": "chol"},
        "quality": {"passes": 5, "iters": 500, "no_multires": False, "hist_mode": "chol"},
    }
    for key, value in presets[args.preset].items():
        setattr(args, key, value)
    return args


def resolve_memory_format(memory_format: str, device: str, size: int):
    if memory_format == "auto":
        memory_format = "channels_last" if device == "cpu" or (device == "mps" and size >= 256) else "contiguous"
    return torch.contiguous_format if memory_format == "contiguous" else torch.channels_last


def random_rotation(N: int, device: torch.device, dtype: torch.dtype):
    """
    Draws a random N-dimensional rotation matrix on the target torch device.
    """

    target_device = device
    if device.type == "mps":
        device = torch.device("cpu")

    q, r = torch.linalg.qr(torch.randn((N, N), device=device, dtype=dtype))
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(device=target_device, dtype=dtype)


def random_rotations(count: int, N: int, device: torch.device, dtype: torch.dtype):
    if count == 0:
        return ()

    generation_device = torch.device("cpu") if device.type == "mps" else device
    rotations = torch.stack([random_rotation(N, generation_device, dtype) for _ in range(count)])
    return rotations.to(device=device, dtype=dtype)


def optimal_transport(pastiche_feature: Tensor, style_feature: Tensor, hist_mode: str, rotation: Optional[Tensor] = None):
    if rotation is None:
        rotation = random_rotation(
            pastiche_feature.shape[-1],
            device=pastiche_feature.device,
            dtype=pastiche_feature.dtype,
        )
    rotated_pastiche = pastiche_feature @ rotation
    rotated_style = style_feature @ rotation

    matched_pastiche = hist_match(rotated_pastiche, rotated_style, mode=hist_mode)

    pastiche_feature = matched_pastiche @ rotation.T  # rotate back to normal

    return pastiche_feature


def fit_pca(tensor: Tensor):
    # fit pca
    A = tensor.reshape(-1, tensor.shape[-1]) - tensor.mean()
    _, eigvals, eigvecs = torch.svd(A)
    k = (torch.cumsum(eigvals / torch.sum(eigvals), dim=0) > 0.9).max(0).indices.squeeze()
    eigvecs = eigvecs[:, :k]  # the vectors for 90% of variance will be kept

    # apply to input
    features = tensor @ eigvecs

    return features, eigvecs


def mix_style_features(style_features: List[Tensor], mixing_mask: Tensor, mixing_alpha: float, hist_mode: str):
    i = mixing_alpha

    for l, sf in enumerate(style_features):
        mix = to_nhwc(interpolate(mixing_mask, size=sf.shape[1:3], mode="nearest"))

        A, B = sf[[0]], sf[[1]]
        AtoB = hist_match(A, B, mode=hist_mode)
        BtoA = hist_match(B, A, mode=hist_mode)

        style_target = (A * (1 - i) + AtoB * i) * mix + (BtoA * (1 - i) + B * i) * (1 - mix)

        style_features[l] = style_target
    return style_features


if __name__ == "__main__":

    def required_length(nmin, nmax):
        class RequiredLength(argparse.Action):
            def __call__(self, parser, args, values, option_string=None):
                if not nmin <= len(values) <= nmax:
                    msg = f'argument "{self.dest}" requires between {nmin} and {nmax} arguments'
                    raise argparse.ArgumentTypeError(msg)
                setattr(args, self.dest, values)

        return RequiredLength

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--style", type=str, nargs="+", action=required_length(1, 2), default=["style/graffiti.jpg"], help="Example(s) of the style your texture should take")
    parser.add_argument("-c", "--content", type=str, default=None, help="The structure/shape you want your image to take")
    parser.add_argument("--batch", type=int, default=1, help="Batch size of images to generate")
    parser.add_argument("--size", type=int, default=512, help="The output size of the image (larger output = more memory/time required)")
    parser.add_argument("--passes", type=int, default=5, help="Number of times to loop over each of the 5 layers in VGG-19")
    parser.add_argument("--iters", type=int, default=500, help="Total number of iterations to optimize.")
    parser.add_argument("--hist_mode", type=str, choices=["sym", "pca", "chol", "cdf"], default="chol", help="Histogram matching strategy. CDF is slower than the others, but may use less memory. Each gives slightly different results.")
    parser.add_argument("--color_transfer", type=str, default=None, choices=["lum", "opt"], help="Strategy to employ to keep original color of content image.")
    parser.add_argument("--content_strength", type=float, default=0.01, help="Strength with which to focus on the structure in your content image.")
    parser.add_argument("--style_scale", type=float, default=1.0, help="Scale the style relative to the generated image. Will affect the scale of details generated.")
    parser.add_argument("--mixing_alpha", type=float, default=0.5, help="Value between 0 and 1 for interpolation between 2 textures")
    parser.add_argument("--no_pca", action="store_true", help="Disable PCA of features (slower).")
    parser.add_argument("--no_multires", action="store_true", help="Disable multi-scale rendering (slower, less long-range texture qualities).")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random number generator.")
    parser.add_argument("--no_tf32", action="store_true", help="Disable tf32 format (probably slower).")
    parser.add_argument("--cudnn_benchmark", action="store_true", help="Enable CUDNN benchmarking (probably slower unless doing a high number of iterations).")
    parser.add_argument("--compile", action="store_true", help="Use PyTorch 2.0 compile function to optimize the model.")
    parser.add_argument("--script", action="store_true", help="Use PyTorch JIT script function to optimize the model.")
    parser.add_argument("--preset", type=str, choices=["fast", "balanced", "quality"], default=None, help="Apply benchmark-backed speed/quality settings.")
    parser.add_argument("--device", type=str, default=None, help="Which device to run on.")
    parser.add_argument("--memory_format", type=str, default="auto", choices=["auto", "contiguous", "channels_last"], help="Which memory format to use for optimization.")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to output results.")
    args = parser.parse_args()
    # fmt: on
    args = apply_preset(args)

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = not args.no_tf32
    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
    device = resolve_device(args.device, args.size)
    memory_format = resolve_memory_format(args.memory_format, device, args.size)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    with torch.inference_mode():
        styles = load_styles(
            args.style, size=args.size, scale=args.style_scale, device=device, memory_format=memory_format
        )
        if len(styles) > 1:
            assert styles[0].shape == styles[1].shape, "Style images must have the same shape"
        content = maybe_load_content(args.content, size=args.size, device=device, memory_format=memory_format)
        pastiche = torch.rand(content.shape if content is not None else (args.batch, 3, args.size, args.size)).to(
            device=device, memory_format=memory_format
        )

        texturizer = OptimalTexture(
            size=args.size,
            iters=args.iters,
            passes=args.passes,
            hist_mode=args.hist_mode,
            color_transfer=args.color_transfer,
            content_strength=args.content_strength,
            style_scale=args.style_scale,
            mixing_alpha=args.mixing_alpha,
            no_pca=args.no_pca,
            no_multires=args.no_multires,
        ).to(pastiche)

        if args.script:
            texturizer = torch.jit.optimize_for_inference(torch.jit.script(texturizer))
        if args.compile:
            texturizer = torch.compile(texturizer)

        from time import time

        t = time()
        pastiche = texturizer.forward(pastiche, styles, content, verbose=True)
        print("Took:", time() - t)

    save_image(pastiche, args)
