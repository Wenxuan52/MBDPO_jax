"""MBDPO package exports.

Top-level agent classes are loaded lazily so metadata-only imports such as
``MBDPO.common.mmbench`` do not require training dependencies or instantiate
anything environment-related.
"""

__all__ = ["MBDPO", "Diffusion"]


def __getattr__(name):
    if name == "MBDPO":
        from .mbdpo import MBDPO

        return MBDPO
    if name == "Diffusion":
        from diffusion import Diffusion

        return Diffusion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
