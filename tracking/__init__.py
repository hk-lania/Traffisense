"""TraffiSense - tracking (stand-in until Hitarth's ByteTrack module lands)."""

from .byte_tracker import CLASS_MAP, ByteTracker, normalise_class

__all__ = ["ByteTracker", "normalise_class", "CLASS_MAP"]
