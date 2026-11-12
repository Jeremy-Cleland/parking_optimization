"""Optional Apple GPU batching for geographic distances, with a NumPy baseline."""

import platform
import time
from importlib.metadata import version
from typing import Any

import numpy as np


def mlx_available() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core as mx

        return bool(mx.metal.is_available())
    except ImportError:
        return False


def walking_distances(
    destinations: np.ndarray, zones: np.ndarray, backend: str = "auto"
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return driver-by-zone meters; center coordinates before float32 conversion.

    Centering avoids subtracting nearly equal float32 latitude/longitude values.
    GPU results are evaluated and transferred before this function returns.
    """
    if backend not in {"auto", "numpy", "mlx"}:
        raise ValueError("Array backend must be auto, numpy or mlx")
    destinations = np.asarray(destinations, dtype=np.float64).reshape(-1, 2)
    zones = np.asarray(zones, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(destinations).all() or not np.isfinite(zones).all():
        raise ValueError("Coordinates must be finite")
    size = len(destinations) * len(zones)
    selected = backend
    if backend == "auto":
        selected = "mlx" if size >= 100_000 and mlx_available() else "numpy"
    if selected == "mlx" and not mlx_available():
        raise ValueError(
            "MLX Metal is unavailable; install the apple extra or use numpy"
        )
    metadata = {
        "backend": selected,
        "device": "CPU",
        "dtype": "float32",
        "drivers": len(destinations),
        "zones": len(zones),
    }
    if not size:
        return np.empty((len(destinations), len(zones)), dtype=np.float32), metadata
    origin = np.deg2rad(zones[0])
    destination_radians = np.deg2rad(destinations)
    zone_radians = np.deg2rad(zones)
    d = (destination_radians - origin).astype(np.float32)
    z = (zone_radians - origin).astype(np.float32)
    latitude_origin = np.float32(origin[0])
    xp: Any = np
    if selected == "mlx":
        import mlx.core as mx

        xp = mx
        metadata.update(
            device=mx.device_info().get("device_name", "Apple GPU"),
            version=version("mlx"),
        )
        # Scope the stream rather than changing a caller's global default device.
        with mx.stream(mx.gpu):
            a, b = mx.array(d), mx.array(z)
            delta_lat = a[:, None, 0] - b[None, :, 0]
            delta_lon = a[:, None, 1] - b[None, :, 1]
            h = (
                mx.sin(delta_lat / 2) ** 2
                + mx.cos(a[:, None, 0] + latitude_origin)
                * mx.cos(b[None, :, 0] + latitude_origin)
                * mx.sin(delta_lon / 2) ** 2
            )
            result = 12_742_000 * mx.arcsin(mx.sqrt(mx.clip(h, 0, 1)))
            mx.eval(result)
            mx.synchronize(mx.gpu)
            return np.array(result), metadata
    delta_lat = d[:, None, 0] - z[None, :, 0]
    delta_lon = d[:, None, 1] - z[None, :, 1]
    h = (
        xp.sin(delta_lat / 2) ** 2
        + xp.cos(d[:, None, 0] + latitude_origin)
        * xp.cos(z[None, :, 0] + latitude_origin)
        * xp.sin(delta_lon / 2) ** 2
    )
    result = 12_742_000 * xp.arcsin(xp.sqrt(xp.clip(h, 0, 1)))
    return result, metadata


def benchmark_distances(
    destinations: np.ndarray, zones: np.ndarray, repeats: int = 5
) -> dict:
    """Include GPU dispatch, evaluation, synchronization and NumPy transfer time."""
    timings = {}
    outputs = {}
    backends = ["numpy", "mlx"] if mlx_available() else ["numpy"]
    for backend in backends:
        walking_distances(destinations, zones, backend)  # Exclude one warmup.
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            outputs[backend], metadata = walking_distances(destinations, zones, backend)
            samples.append(time.perf_counter() - started)
        timings[backend] = {
            "median_seconds": float(np.median(samples)),
            "samples_seconds": samples,
            **metadata,
        }
    result = {
        "scope": "batched_walking_distances_only",
        "timings": timings,
        "includes": "coordinate conversion, allocation, GPU evaluation/synchronization, output transfer",
    }
    if "mlx" in outputs:
        result["max_absolute_difference_meters"] = float(
            np.max(np.abs(outputs["mlx"] - outputs["numpy"]))
        )
        result["numpy_over_mlx_speedup"] = (
            timings["numpy"]["median_seconds"] / timings["mlx"]["median_seconds"]
        )
    return result
