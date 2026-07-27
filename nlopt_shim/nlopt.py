"""Shim for the `nlopt` package, which has no aarch64 build on PyPI (2026-07).

CadQuery imports nlopt at module load (cadquery/occ_impl/sketch_solver.py) but only
*calls* it when solving 2D sketch constraints — a feature this project's generated
code never uses. This shim satisfies the import and raises a clear error if any
real solve is attempted. Installed into the venv by `install.sh` / documented in README.
"""

LD_SLSQP = 40  # matches the real nlopt constant


class opt:  # noqa: N801 - mirrors the real nlopt API name
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "nlopt shim: real nlopt is not installed on this machine (no aarch64 wheel). "
            "Sketch constraint solving is unavailable; model the geometry directly instead."
        )


def __getattr__(name):
    raise ImportError(f"nlopt shim: attribute {name!r} unavailable (real nlopt not installed)")
