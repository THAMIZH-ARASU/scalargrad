"""
Configuration and logging setup for ScalarGrad.
"""

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LogLevel(Enum):
    """Enumeration for logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


@dataclass
class Config:
    """Global configuration for ScalarGrad."""
    log_level: LogLevel = LogLevel.INFO
    seed: Optional[int] = None
    gradient_clip: Optional[float] = None
    numerical_stability_epsilon: float = 1e-8
    
    def __post_init__(self):
        """Initialize logging and random seed."""
        logging.basicConfig(
            level=self.log_level.value,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        if self.seed is not None:
            random.seed(self.seed)


# Global configuration instance
config = Config()
logger = logging.getLogger(__name__)
