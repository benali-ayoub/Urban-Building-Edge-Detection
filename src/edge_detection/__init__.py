"""
dge detection and processing modules.
"""

from .edge_processors import detect_edges, enhance_building_edges
from .noise_reduction import remove_small_components, clean_edges

__all__ = [
    "detect_edges",
    "enhance_building_edges",
    "remove_small_components",
    "clean_edges",
]
