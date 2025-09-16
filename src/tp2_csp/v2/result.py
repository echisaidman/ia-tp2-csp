from dataclasses import dataclass

from .common_types import CutsLayoutStrategy, MutationStrategy
from .individual import Individual


@dataclass
class SimulationResult:
    bar_length: int
    required_cuts: dict[int, int]
    population_size: int
    generations: int
    tournament_size: int
    mutation_rate: float
    crossover_rate: float
    elitism_size: int
    mutation_strategy: MutationStrategy
    cuts_layout_strategy: CutsLayoutStrategy
    best_solutions_by_generation: list[Individual]
