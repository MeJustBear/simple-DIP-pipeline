"""Common protocol shared by all pipeline stages.

A :class:`Stage` is a small, self-contained processing step that maps one
array to another. Stages know nothing about files or the CLI; IO is handled by
the pipeline runner via :mod:`dippipe.io.artifacts`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Stage(ABC):
    """A single processing step in the ISP pipeline."""

    #: Short identifier used for artifact filenames and CLI subcommands.
    name: str = "stage"

    @abstractmethod
    def run(self, data: np.ndarray) -> np.ndarray:
        """Transform ``data`` and return the result array."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
