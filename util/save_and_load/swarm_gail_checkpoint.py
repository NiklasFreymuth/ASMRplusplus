from dataclasses import dataclass
from typing import Optional

from src.algorithms.rl.normalizers.abstract_environment_normalizer import (
    AbstractEnvironmentNormalizer,
)
from src.modules.abstract_architecture import AbstractArchitecture


@dataclass
class SwarmGAILCheckpoint:
    architecture: AbstractArchitecture
    discriminator: AbstractArchitecture
    normalizer: Optional[AbstractEnvironmentNormalizer]
