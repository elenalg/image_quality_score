"""Convenient exports for the image_quality package."""

from .image_quality import ImageQuality, load_config, get_default_config

try:
    from .iqa_metrics import IQAMetrics, is_iqa_available
except ImportError:  # pragma: no cover - optional dependency
    IQAMetrics = None
    is_iqa_available = lambda: False  # type: ignore
