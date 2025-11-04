from typing import Literal

from pydantic import BaseModel, Field, PositiveInt

type SelectionStrategy = Literal["Tournament", "Ranking"]
type MutationStrategy = Literal["SwapCuts", "ReverseSubsequence"]
type CutsLayoutStrategy = Literal["InOrder", "BestFit"]


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
    selection_strategy: SelectionStrategy = "Tournament"
    tournament_size: int = 5
    selection_size: int = 100  # How many individuals we will have in the selection population
    mutation_strategy: MutationStrategy = "SwapCuts"
    mutation_rate: float = 0.50
    percentage_of_individuals_to_mutate: float | None = Field(default=None, repr=False)
    cuts_layout_strategy: CutsLayoutStrategy = Field(default="InOrder", repr=False)
    solution_target: int | None = Field(default=None, repr=False)

    def __repr__(self) -> str:
        """Ignore repr=False fields if they have the default value"""
        fields_to_show = self.__get_fields_to_show()
        fields_values = [f"{field_name}={repr(getattr(self, field_name))}" for field_name in fields_to_show]
        return f"{self.__class__.__name__}({', '.join(fields_values)})"

    def __str__(self) -> str:
        fields_to_show = self.__get_fields_to_show()
        fields_values = [f"{field_name}={repr(getattr(self, field_name))}" for field_name in fields_to_show]
        return f"{' '.join(fields_values)}"

    def __get_fields_to_show(self) -> list[str]:
        """Ignore repr=False fields if they have the default value"""
        fields_to_show: list[str] = []
        for field_name, field_info in Parameters.model_fields.items():
            current_value = getattr(self, field_name)
            default_value = field_info.default
            if field_info.repr or current_value != default_value:
                fields_to_show.append(field_name)

        if self.selection_strategy != "Tournament" and "tournament_size" in fields_to_show:
            fields_to_show.remove("tournament_size")

        return fields_to_show
