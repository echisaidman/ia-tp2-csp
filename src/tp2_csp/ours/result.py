from pydantic import BaseModel

from .individual import Individual
from .parameters import Parameters


class SimulationResult(BaseModel):
    parameters: Parameters
    generations: int
    best_solutions_by_generation: list[Individual]
