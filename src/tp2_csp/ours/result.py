from dataclasses import dataclass

from .individual import Individual
from .parameters import Parameters


@dataclass
class SimulationResult:
    parameters: Parameters
    generations: int
    best_solutions_by_generation: list[Individual]
