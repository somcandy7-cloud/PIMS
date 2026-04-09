from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AlarmThreshold:
    high: float = float("inf")
    low: float = float("-inf")
    rate_of_change: float = float("inf")


@dataclass
class SignalSchema:
    name: str
    unit: str
    signal_type: Literal["analog", "digital"]
    description: str = ""
    threshold: AlarmThreshold = field(default_factory=AlarmThreshold)
