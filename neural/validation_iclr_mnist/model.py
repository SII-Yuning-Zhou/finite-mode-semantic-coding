"""ICLR-style MNIST codec, entropy model, dither, GAN, and task heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.metadata import version

import torch
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN as CompressAIGDN
from torch import nn
from torch.nn import functional as F

from .protocol import COMPRESSAI_ENTROPY_PROTOCOL, COMPRESSAI_PACKAGE_VERSION

COMPRESSAI_VERSION = version("compressai")
if COMPRESSAI_VERSION != COMPRESSAI_PACKAGE_VERSION:
    raise RuntimeError(
        f"Frozen protocol requires compressai=={COMPRESSAI_PACKAGE_VERSION}, "
        f"found {COMPRESSAI_VERSION}."
    )


class LegacySoftplusGDN(nn.Module):
    """Small positive-parameter GDN/IGDN layer without a CompressAI dependency."""

    def __init__(self, channels: int, inverse: bool = False):
        super().__init__()
        self.inverse = inverse
        inverse_softplus = lambda value: math.log(math.expm1(value))
        self.beta_reparam = nn.Parameter(
            torch.full((channels,), inverse_softplus(1.0))
        )
        initial = torch.full((channels, channels), inverse_softplus(1e-4))
        initial.fill_diagonal_(inverse_softplus(0.1))
        self.gamma_reparam = nn.Parameter(initial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GDN is a numerical precision island.  In particular, squaring a
        # half-precision activation can overflow before the following
        # convolution has a chance to accumulate in FP32.  Returning FP32 is
        # intentional: an enclosing autocast context may still run the next
        # convolution in reduced precision, while the normalization itself
        # remains finite and differentiable.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_float = x.float()
            beta = F.softplus(self.beta_reparam.float()) + 1e-6
            gamma = F.softplus(self.gamma_reparam.float()).view(
                x.shape[1], x.shape[1], 1, 1
            )
            norm = F.conv2d(x_float.square(), gamma, beta).clamp_min(1e-12)
            if self.inverse:
                return x_float * torch.sqrt(norm)
            return x_float * torch.rsqrt(norm)


def make_gdn(channels: int, *, inverse: bool, implementation: str) -> nn.Module:
    """Build either the archived local layer or CompressAI's official GDN."""

    if implementation == "compressai":
        return CompressAIGDN(channels, inverse=inverse)
    if implementation == "legacy_softplus":
        return LegacySoftplusGDN(channels, inverse=inverse)
    raise ValueError(f"Unknown GDN implementation: {implementation}")


class DitherQuantizer(nn.Module):
    """Universal continuous dither or a frozen finite shared-dither codebook."""

    def __init__(
        self,
        latent_channels: int,
        *,
        kind: str,
        mode_slots: int = 1,
        codebook_seed: int = 1729,
        spatial_shape: tuple[int, int] = (4, 4),
    ):
        super().__init__()
        if kind not in {"continuous", "finite"}:
            raise ValueError("kind must be continuous or finite.")
        if mode_slots < 1:
            raise ValueError("mode_slots must be positive.")
        self.kind = kind
        self.mode_slots = mode_slots
        self.codebook_seed = codebook_seed
        generator = torch.Generator(device="cpu").manual_seed(codebook_seed)
        if kind == "finite":
            codebook = torch.rand(
                mode_slots,
                latent_channels,
                *spatial_shape,
                generator=generator,
            ) - 0.5
        else:
            codebook = torch.zeros(1, latent_channels, *spatial_shape)
        self.register_buffer("codebook", codebook)

    def sample(
        self,
        latent: torch.Tensor,
        modes: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = latent.shape[0]
        if self.kind == "continuous":
            dither = torch.rand(
                latent.shape,
                dtype=latent.dtype,
                device=latent.device,
                generator=generator,
            ) - 0.5
            used_modes = torch.zeros(batch, dtype=torch.long, device=latent.device)
        else:
            if latent.shape[1:] != self.codebook.shape[1:]:
                raise ValueError("Latent shape does not match the frozen dither codebook.")
            if modes is None or modes.shape != (batch,):
                raise ValueError("Finite dither requires one mode index per sample.")
            used_modes = modes.to(device=latent.device, dtype=torch.long)
            if torch.any(used_modes < 0) or torch.any(used_modes >= self.mode_slots):
                raise ValueError("Mode index lies outside the frozen codebook.")
            dither = self.codebook[used_modes].to(dtype=latent.dtype)
        return {"dither": dither, "modes": used_modes}

    def forward(
        self,
        latent: torch.Tensor,
        modes: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Legacy hard-quantizer view retained for focused dither tests.

        The formal codec uses :class:`EntropyBottleneck` for quantization and
        entropy coding.  This method remains useful for checking that the
        shared dither codebook itself is deterministic and mode conditioned.
        """

        state = self.sample(latent, modes, generator=generator)
        dither = state["dither"]
        integer_symbols = torch.round(latent + dither)
        quantized_hard = integer_symbols - dither
        quantized = latent + (quantized_hard - latent).detach() if self.training else quantized_hard
        return {
            "quantized": quantized,
            "symbols": integer_symbols.to(torch.int32),
            "dither": dither,
            "modes": state["modes"],
        }


class ModeConditionedEntropyBottleneck(nn.Module):
    """One official entropy bottleneck per receiver-known codec mode.

    Conditioning the probability table on the shared mode makes the
    operational payload correspond to a cross-entropy upper bound on
    ``H(Y|V)``.  The continuous-dither baseline has exactly one model.
    """

    def __init__(self, channels: int, mode_slots: int):
        super().__init__()
        if mode_slots < 1:
            raise ValueError("Entropy mode_slots must be positive.")
        self.mode_slots = mode_slots
        self.models = nn.ModuleList(
            [
                EntropyBottleneck(
                    channels,
                    tail_mass=1e-9,
                    init_scale=10,
                    entropy_coder="ans",
                    entropy_coder_precision=16,
                )
                for _ in range(mode_slots)
            ]
        )

    def _validate_modes(self, modes: torch.Tensor, batch: int) -> torch.Tensor:
        if modes.shape != (batch,):
            raise ValueError("Entropy coding requires one mode per sample.")
        modes = modes.to(dtype=torch.long)
        if torch.any(modes < 0) or torch.any(modes >= self.mode_slots):
            raise ValueError("Entropy mode index lies outside the deployed models.")
        return modes

    def forward(
        self,
        shifted_latent: torch.Tensor,
        modes: torch.Tensor,
        *,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modes = self._validate_modes(modes, len(shifted_latent))
        outputs = torch.zeros_like(shifted_latent)
        likelihoods = torch.zeros_like(shifted_latent)
        for mode_index, entropy_model in enumerate(self.models):
            indices = torch.nonzero(modes == mode_index, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            mode_output, mode_likelihoods = entropy_model(
                shifted_latent.index_select(0, indices), training=training
            )
            outputs = outputs.index_copy(0, indices, mode_output)
            likelihoods = likelihoods.index_copy(0, indices, mode_likelihoods)
        return outputs, likelihoods

    def symbols(
        self, shifted_latent: torch.Tensor, modes: torch.Tensor
    ) -> torch.Tensor:
        modes = self._validate_modes(modes, len(shifted_latent))
        symbols = torch.zeros_like(shifted_latent, dtype=torch.int32)
        for mode_index, entropy_model in enumerate(self.models):
            indices = torch.nonzero(modes == mode_index, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            values = shifted_latent.index_select(0, indices)
            medians = entropy_model.quantiles[:, 0, 1]
            medians = medians.view(1, -1, *([1] * (values.ndim - 2)))
            mode_symbols = entropy_model.quantize(values, "symbols", medians)
            symbols.index_copy_(0, indices, mode_symbols.to(torch.int32))
        return symbols

    @torch.no_grad()
    def compress_decompress(
        self, shifted_latent: torch.Tensor, modes: torch.Tensor
    ) -> tuple[list[bytes], torch.Tensor]:
        """Code one block per active receiver-known mode.

        ``EntropyBottleneck.compress`` otherwise creates one independently
        terminated ANS stream per image.  For a 28x28 image, its fixed stream
        termination overhead alone was 56 bytes = 0.571429 bpp, masking the
        learned low-rate regime.  Packing the latent tiles of a validation
        minibatch into one spatial block per active mode retains exact symbol
        factorization and round-trip reconstruction while amortizing framing
        overhead, as a practical block entropy coder does.
        """
        modes = self._validate_modes(modes, len(shifted_latent))
        strings: list[bytes] = []
        decoded = torch.empty_like(shifted_latent)
        for mode_index, entropy_model in enumerate(self.models):
            indices = torch.nonzero(modes == mode_index, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            values = shifted_latent.index_select(0, indices)
            count, channels, height, width = values.shape
            packed = (
                values.permute(1, 2, 0, 3)
                .reshape(1, channels, height, count * width)
                .contiguous()
            )
            mode_strings = entropy_model.compress(packed)
            if len(mode_strings) != 1:
                raise RuntimeError("Block entropy coding must emit one stream per mode.")
            packed_decoded = entropy_model.decompress(
                mode_strings, (height, count * width)
            )
            mode_decoded = (
                packed_decoded[0]
                .reshape(channels, height, count, width)
                .permute(2, 0, 1, 3)
                .contiguous()
            )
            decoded.index_copy_(0, indices, mode_decoded)
            strings.append(mode_strings[0])
        if not strings:
            raise RuntimeError("At least one entropy payload was not produced.")
        return strings, decoded

    def main_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.named_parameters()
            if not name.endswith(".quantiles")
        ]

    def aux_parameters(self) -> list[nn.Parameter]:
        return [model.quantiles for model in self.models]

    def aux_loss(self) -> torch.Tensor:
        return sum(model.loss() for model in self.models)

    def update(self, *, force: bool = False) -> bool:
        updated = [model.update(force=force) for model in self.models]
        return any(updated)

    def is_updated(self) -> bool:
        return all(model._quantized_cdf.numel() > 0 for model in self.models)

    def cdf_table_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for model in self.models
            for tensor in (model._offset, model._quantized_cdf, model._cdf_length)
        )

    @property
    def entropy_coder_name(self) -> str:
        names = {model.entropy_coder.name for model in self.models}
        if len(names) != 1:
            raise RuntimeError("Deployed entropy modes use inconsistent coders.")
        return next(iter(names))

    @property
    def entropy_coder_precision(self) -> int:
        precisions = {model.entropy_coder_precision for model in self.models}
        if len(precisions) != 1:
            raise RuntimeError("Deployed entropy modes use inconsistent CDF precision.")
        return next(iter(precisions))

    @property
    def tail_mass(self) -> float:
        masses = {model.tail_mass for model in self.models}
        if len(masses) != 1:
            raise RuntimeError("Deployed entropy modes use inconsistent tail mass.")
        return next(iter(masses))


class AnalysisTransform(nn.Module):
    def __init__(
        self,
        base_channels: int,
        latent_channels: int,
        *,
        gdn_implementation: str = "legacy_softplus",
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, base_channels, 5, stride=2, padding=2),
            make_gdn(
                base_channels, inverse=False, implementation=gdn_implementation
            ),
            nn.Conv2d(base_channels, base_channels, 5, stride=2, padding=2),
            make_gdn(
                base_channels, inverse=False, implementation=gdn_implementation
            ),
            nn.Conv2d(base_channels, latent_channels, 5, stride=2, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SynthesisTransform(nn.Module):
    def __init__(
        self,
        base_channels: int,
        latent_channels: int,
        *,
        output_activation: str = "sigmoid",
        gdn_implementation: str = "legacy_softplus",
    ):
        super().__init__()
        if output_activation not in {"sigmoid", "identity"}:
            raise ValueError("output_activation must be sigmoid or identity.")
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(latent_channels, base_channels, 5, stride=2, padding=2),
            make_gdn(
                base_channels, inverse=True, implementation=gdn_implementation
            ),
            nn.ConvTranspose2d(
                base_channels, base_channels, 5, stride=2, padding=2, output_padding=1
            ),
            make_gdn(
                base_channels, inverse=True, implementation=gdn_implementation
            ),
            nn.ConvTranspose2d(
                base_channels, 1, 5, stride=2, padding=2, output_padding=1
            ),
        ]
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        self.output_activation = output_activation
        self.layers = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.layers(latent)


class ICLRMNISTCodec(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = 48,
        latent_channels: int = 12,
        dither_kind: str = "continuous",
        mode_slots: int = 1,
        codebook_seed: int = 1729,
        decoder_output_activation: str = "sigmoid",
        gdn_implementation: str = "legacy_softplus",
        decoder_output_postprocess: str = "identity",
    ):
        super().__init__()
        if decoder_output_postprocess not in {"identity", "clamp_0_1"}:
            raise ValueError(
                "decoder_output_postprocess must be identity or clamp_0_1."
            )
        self.decoder_output_postprocess = decoder_output_postprocess
        self.encoder = AnalysisTransform(
            base_channels,
            latent_channels,
            gdn_implementation=gdn_implementation,
        )
        self.quantizer = DitherQuantizer(
            latent_channels,
            kind=dither_kind,
            mode_slots=mode_slots,
            codebook_seed=codebook_seed,
        )
        entropy_mode_slots = mode_slots if dither_kind == "finite" else 1
        self.entropy_model = ModeConditionedEntropyBottleneck(
            latent_channels, entropy_mode_slots
        )
        self.decoder = SynthesisTransform(
            base_channels,
            latent_channels,
            output_activation=decoder_output_activation,
            gdn_implementation=gdn_implementation,
        )

    def postprocess_decoder_output(
        self, reconstruction: torch.Tensor
    ) -> torch.Tensor:
        """Apply the operational receiver range constraint after neural synthesis."""

        if self.decoder_output_postprocess == "clamp_0_1":
            return reconstruction.clamp(0.0, 1.0)
        return reconstruction

    def entropy_main_parameters(self) -> list[nn.Parameter]:
        return self.entropy_model.main_parameters()

    def entropy_aux_parameters(self) -> list[nn.Parameter]:
        return self.entropy_model.aux_parameters()

    def entropy_aux_loss(self) -> torch.Tensor:
        return self.entropy_model.aux_loss()

    def update_entropy_model(self, *, force: bool = False) -> bool:
        return bool(self.entropy_model.update(force=force))

    def forward(
        self,
        x: torch.Tensor,
        modes: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        latent = self.encoder(x)
        dither_state = self.quantizer.sample(latent, modes, generator=generator)
        # The entropy model is deliberately a full-precision island.  During
        # training EntropyBottleneck adds the standard uniform relaxation;
        # during evaluation it performs median-centred hard quantization.  The
        # shared dither is added before and subtracted after that operation.
        with torch.autocast(device_type=latent.device.type, enabled=False):
            shifted_latent = latent.float() + dither_state["dither"].float()
            quantized_shifted, likelihoods = self.entropy_model(
                shifted_latent, dither_state["modes"]
            )
            quantized = quantized_shifted - dither_state["dither"].float()
            bits = -torch.log2(likelihoods)
            symbols = self.entropy_model.symbols(
                shifted_latent, dither_state["modes"]
            )
        # Keep the linear synthesis output unconstrained while optimizing the
        # ICLR B.1 Lagrangian.  CompressAI likewise clamps only its operational
        # decompress path, not the differentiable training forward path.
        reconstruction = self.decoder(quantized)
        return {
            "latent": latent,
            "reconstruction": reconstruction,
            "rate_bits": bits,
            "likelihoods": likelihoods,
            "quantized": quantized,
            "quantized_shifted": quantized_shifted,
            "symbols": symbols,
            **dither_state,
        }

    @torch.no_grad()
    def compress(
        self,
        x: torch.Tensor,
        modes: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, object]:
        """Encode a real ANS payload and reproduce its receiver reconstruction."""

        if self.training:
            raise RuntimeError("compress() requires codec.eval().")
        if not self.entropy_model.is_updated():
            raise RuntimeError("Entropy CDF is uninitialized; call update_entropy_model().")
        latent = self.encoder(x)
        dither_state = self.quantizer.sample(latent, modes, generator=generator)
        shifted_latent = latent.float() + dither_state["dither"].float()
        quantized_shifted, likelihoods = self.entropy_model(
            shifted_latent, dither_state["modes"], training=False
        )
        strings, decoded_shifted = self.entropy_model.compress_decompress(
            shifted_latent, dither_state["modes"]
        )
        if not torch.equal(decoded_shifted, quantized_shifted):
            raise RuntimeError("CompressAI entropy decode did not reproduce the latent.")
        quantized = quantized_shifted - dither_state["dither"].float()
        decoded_quantized = decoded_shifted - dither_state["dither"].float()
        raw_reconstruction = self.decoder(quantized)
        raw_decoded_reconstruction = self.decoder(decoded_quantized)
        reconstruction = self.postprocess_decoder_output(raw_reconstruction)
        decoded_reconstruction = self.postprocess_decoder_output(
            raw_decoded_reconstruction
        )
        return {
            "latent": latent,
            "quantized": quantized,
            "decoded_quantized": decoded_quantized,
            "reconstruction": reconstruction,
            "decoded_reconstruction": decoded_reconstruction,
            "raw_reconstruction": raw_reconstruction,
            "raw_decoded_reconstruction": raw_decoded_reconstruction,
            "rate_bits": -torch.log2(likelihoods),
            "likelihoods": likelihoods,
            "symbols": self.entropy_model.symbols(
                shifted_latent, dither_state["modes"]
            ),
            "strings": strings,
            **dither_state,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class Discriminator(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        blocks = []
        channels = 1
        for output_channels in (width, 2 * width, 4 * width):
            blocks.extend(
                [
                    nn.Conv2d(channels, output_channels, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).mean(dim=(-2, -1))
        return self.score(features).squeeze(1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class GrayResNet18Small(nn.Module):
    """Four-stage grayscale ResNet-18 with a configurable small width."""

    def __init__(self, width: int = 32, classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        stages = []
        current = width
        for stage, output in enumerate((width, 2 * width, 4 * width, 8 * width)):
            stride = 1 if stage == 0 else 2
            stages.append(nn.Sequential(BasicBlock(current, output, stride), BasicBlock(output, output)))
            current = output
        self.stages = nn.Sequential(*stages)
        self.feature_dimension = current
        self.classifier = nn.Linear(current, classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return x.mean(dim=(-2, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class EqualCapacityModeHead(nn.Module):
    """Equal-shape M/O task head; O receives a forced-zero mode embedding."""

    def __init__(
        self,
        feature_dimension: int,
        *,
        mode_slots: int,
        embedding_dimension: int = 16,
        classes: int = 10,
        mode_aware: bool,
    ):
        super().__init__()
        self.mode_aware = mode_aware
        self.embedding = nn.Embedding(mode_slots, embedding_dimension)
        self.classifier = nn.Linear(feature_dimension + embedding_dimension, classes)

    def forward(self, features: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(modes)
        if not self.mode_aware:
            embedded = torch.zeros_like(embedded)
        return self.classifier(torch.cat([features, embedded], dim=1))


def gradient_penalty(
    discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor
) -> torch.Tensor:
    batch = real.shape[0]
    alpha = torch.rand(batch, 1, 1, 1, device=real.device, dtype=real.dtype)
    interpolated = alpha * real + (1.0 - alpha) * fake
    interpolated.requires_grad_(True)
    score = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return (gradients.flatten(1).norm(2, dim=1) - 1.0).square().mean()


@dataclass(frozen=True)
class ModelResources:
    parameters: int
    parameter_bytes: int
    encoder_parameters: int
    encoder_parameter_bytes: int
    decoder_parameters: int
    decoder_parameter_bytes: int
    entropy_model_parameters: int
    entropy_model_parameter_bytes: int
    model_state_bytes: int
    codebook_values: int
    codebook_bytes: int


def _module_parameter_bytes(module: nn.Module) -> int:
    return sum(value.numel() * value.element_size() for value in module.parameters())


def _module_state_bytes(module: nn.Module) -> int:
    return sum(value.numel() * value.element_size() for value in module.state_dict().values())


def model_resources(codec: ICLRMNISTCodec) -> ModelResources:
    values = codec.quantizer.codebook.numel() if codec.quantizer.kind == "finite" else 0
    return ModelResources(
        parameters=codec.parameter_count(),
        parameter_bytes=sum(value.numel() * value.element_size() for value in codec.parameters()),
        encoder_parameters=sum(value.numel() for value in codec.encoder.parameters()),
        encoder_parameter_bytes=_module_parameter_bytes(codec.encoder),
        decoder_parameters=sum(value.numel() for value in codec.decoder.parameters()),
        decoder_parameter_bytes=_module_parameter_bytes(codec.decoder),
        entropy_model_parameters=sum(
            value.numel() for value in codec.entropy_model.parameters()
        ),
        entropy_model_parameter_bytes=_module_parameter_bytes(codec.entropy_model),
        model_state_bytes=_module_state_bytes(codec),
        codebook_values=values,
        codebook_bytes=values * codec.quantizer.codebook.element_size(),
    )
