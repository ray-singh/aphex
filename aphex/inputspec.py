"""Input specification — describes the tensor(s) a model consumes.

Historically aphex assumed a single input tensor named ``"input"``. Many real
models take several (e.g. a transformer's ``input_ids`` + ``attention_mask``).
``InputSpec`` captures one *or more* named tensors so the benchmark/convert
paths can build, trace, and feed multi-input models.

Backward compatibility is the contract: a spec built from a plain shape behaves
exactly as the old single-tensor code did — one input named ``"input"`` whose
batch axis is dynamic — so existing single-input runs are byte-for-byte
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_DEFAULT_INPUT_NAME = "input"


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]  # without batch dim
    dtype: str = "float"  # "float" | "long"

    def random(self, batch_size: int, torch_dtype: Any, device: Any, vocab: int | None) -> Any:
        """A random tensor of this spec at the given batch size."""
        import torch

        if self.dtype == "long":
            # Index tensors must stay < embedding size. Use the known vocab when
            # available; otherwise 2 is safe for any embedding (timing doesn't
            # depend on the exact indices, only on a valid forward pass).
            high = max(2, vocab or 2)
            return torch.randint(0, high, (batch_size, *self.shape), dtype=torch.long, device=device)
        return torch.randn(batch_size, *self.shape, dtype=torch_dtype, device=device)


@dataclass(frozen=True)
class InputSpec:
    tensors: tuple[TensorSpec, ...]

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def single(cls, shape: list[int] | tuple[int, ...], dtype: str = "float") -> InputSpec:
        return cls((TensorSpec(_DEFAULT_INPUT_NAME, tuple(shape), dtype),))

    @classmethod
    def from_shape(cls, shape: list[int], vocab: int | None = None) -> InputSpec:
        """Single-input spec; ``long`` dtype when the model has an embedding vocab."""
        return cls.single(shape, "long" if vocab else "float")

    def serialize(self) -> str:
        """Inverse of :meth:`parse`: produce the CLI ``--input-shape`` string.

        Single-input default specs (one ``"input"`` tensor of dtype ``float``)
        round-trip to the legacy comma form ``"3,224,224"``; everything else
        uses the explicit ``name:dims[:dtype];...`` syntax. Round-trip property:
        ``InputSpec.parse(spec.serialize()) == spec``.
        """
        if self.is_single:
            ts = self.tensors[0]
            if ts.name == _DEFAULT_INPUT_NAME and ts.dtype == "float":
                return ",".join(str(d) for d in ts.shape)
        return ";".join(
            f"{ts.name}:{','.join(str(d) for d in ts.shape)}:{ts.dtype}"
            for ts in self.tensors
        )

    @classmethod
    def parse(cls, text: str) -> InputSpec:
        """Parse a CLI ``--input-shape`` string.

        Single input (unchanged):      ``"3,224,224"``
        Multiple, ``;``-separated:     ``"input_ids:128;attention_mask:128"``
        With explicit dtype:           ``"input_ids:128:long;mask:128:long"``

        Each segment is ``name:dims[:dtype]`` where ``dims`` is comma-separated.
        A bare segment with no ``:`` is treated as the single default input.
        """
        text = text.strip()
        if not text:
            raise ValueError("empty input shape")

        segments = [s for s in text.split(";") if s.strip()]
        if len(segments) == 1 and ":" not in segments[0]:
            return cls.single(_parse_dims(segments[0]))

        tensors: list[TensorSpec] = []
        for seg in segments:
            parts = seg.split(":")
            if len(parts) < 2:
                raise ValueError(
                    f"multi-input segment {seg!r} must be 'name:dims[:dtype]', "
                    "e.g. 'input_ids:128:long'"
                )
            name, dims = parts[0].strip(), parts[1].strip()
            dtype = parts[2].strip() if len(parts) > 2 else "float"
            if dtype not in ("float", "long"):
                raise ValueError(f"unsupported dtype {dtype!r} (use float or long)")
            tensors.append(TensorSpec(name, _parse_dims(dims), dtype))
        return cls(tuple(tensors))

    # ── queries ───────────────────────────────────────────────────────────────

    @property
    def is_single(self) -> bool:
        return len(self.tensors) == 1

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tensors]

    @property
    def primary_shape(self) -> list[int]:
        """Shape of the first input — for legacy code paths wanting one shape."""
        return list(self.tensors[0].shape)

    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        axes: dict[str, dict[int, str]] = {n: {0: "batch"} for n in self.names}
        axes["output"] = {0: "batch"}
        return axes

    # ── input construction ────────────────────────────────────────────────────

    def build(
        self,
        batch_size: int,
        *,
        samples: list[Any] | None = None,
        torch_dtype: Any = None,
        device: Any = "cpu",
        vocab: int | None = None,
    ) -> Any:
        """Build the model input(s): a tensor when single, else a tuple.

        For single-input specs a representative ``samples[0]`` is tiled to the
        batch size when its shape matches (the same behaviour as the old
        ``_timing_input``); multi-input specs use random data per tensor.
        """
        import torch

        if torch_dtype is None:
            torch_dtype = torch.float32

        built: list[Any] = []
        for ts in self.tensors:
            tensor = None
            if self.is_single and samples:
                tensor = _tile_sample(samples[0], ts, batch_size, torch_dtype, device, vocab)
            if tensor is None:
                tensor = ts.random(batch_size, torch_dtype, device, vocab)
            built.append(tensor)

        return built[0] if self.is_single else tuple(built)


def _parse_dims(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in text.split(",") if x.strip() != "")
    except ValueError as exc:
        raise ValueError(f"invalid dims {text!r}: expected comma-separated ints") from exc


def _tile_sample(
    sample: Any, ts: TensorSpec, batch_size: int, torch_dtype: Any, device: Any, vocab: int | None
) -> Any | None:
    """Tile a representative sample to ``batch_size``; None if it can't be used."""
    import torch

    if not isinstance(sample, torch.Tensor):
        return None
    try:
        unit = sample[0] if sample.dim() == len(ts.shape) + 1 else sample
        if list(unit.shape) != list(ts.shape):
            return None
        batch = unit.unsqueeze(0).repeat(batch_size, *([1] * len(ts.shape)))
        target_dtype = torch.long if ts.dtype == "long" else torch_dtype
        return batch.to(device=device, dtype=target_dtype)
    except Exception:  # noqa: BLE001 — any coercion failure → caller uses random
        return None
