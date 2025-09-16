from pydantic import BaseModel, Field, PositiveInt

from .common_types import CutsLayoutStrategy, MutationStrategy

DEFAULT_BAR_LENGTH = 5600
DEFAULT_REQUIRED_CUTS = {
    1380: 22,
    1520: 25,
    1560: 12,
    1710: 14,
    1820: 18,
    1880: 18,
    1930: 20,
    2000: 10,
    2050: 12,
    2100: 14,
    2140: 16,
    2150: 18,
    2200: 20,
}


class Parameters(BaseModel):
    bar_length: PositiveInt = Field(default=DEFAULT_BAR_LENGTH, repr=False)
    required_cuts: dict[int, int] = Field(default=DEFAULT_REQUIRED_CUTS, repr=False)
    population_size: int = 500
    generations: int = 500
    tournament_size: int = 5
    mutation_rate: float = 0.50
    crossover_rate: float = 0.80
    elitism_size: int = Field(default=0, repr=False)
    mutation_strategy: MutationStrategy = "SwapCuts"
    cuts_layout_strategy: CutsLayoutStrategy = Field(default="InOrder", repr=False)
