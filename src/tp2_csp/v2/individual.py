import random

type Chromosome = list[int]


class Individual:
    bar_length: int
    chromosome: Chromosome
    fitness_score: float
    cuts_layout: list[list[int]]
    bar_remainders: list[int]

    def __init__(
        self,
        bar_length: int,
        chromosome: Chromosome | None = None,
        all_cuts: list[int] | None = None,
    ) -> None:
        self.bar_length = bar_length

        if chromosome is not None:
            self.chromosome = chromosome
        else:
            if all_cuts is None:
                raise ValueError("all_cuts can't be None if chromosome is None")
            chromosome = all_cuts[:]
            random.shuffle(chromosome)
            self.chromosome = chromosome

        self.cuts_layout = self.__calculate_cuts_layout()
        self.bar_remainders = self.__calculate_bar_remainders(self.cuts_layout)
        self.fitness_score = self.__calculate_fitness()

    def calculate_required_bars(self) -> int:
        return len(self.cuts_layout)

    def calculate_total_waste(self) -> int:
        return sum(self.bar_remainders)

    def calculate_percentage_wasted(self) -> float:
        total_wasted = self.calculate_total_waste()
        required_bars = self.calculate_required_bars()
        used_amount = required_bars * self.bar_length
        return (total_wasted * 100.0) / used_amount

    def __calculate_fitness(self) -> float:
        """
        Calculates the fitness of a chromosome using a First Fit heuristic.

        Fitness is inversely proportional to the total waste.

        Returns:
            A tuple containing: (fitness_score, layout_of_bars, num_bars_used, total_waste).
        """

        total_waste = self.calculate_total_waste()

        # Fitness: Higher is better. We want to minimize waste.
        # Adding 1.0 to the denominator to avoid division by zero.
        # fitness_score = 1.0 / (total_waste + 1.0)
        fitness_score = -total_waste
        return fitness_score

    def __calculate_cuts_layout(self) -> list[list[int]]:
        bars_cuts: list[list[int]] = []
        bar_remainders: list[int] = []

        for cut in self.chromosome:
            placed = False
            # Try to place the cut in an existing bar (First Fit)
            for i, remainder in enumerate(bar_remainders):
                if cut <= remainder:
                    bars_cuts[i].append(cut)
                    bar_remainders[i] -= cut
                    placed = True
                    break

            # If it doesn't fit anywhere, start a new bar
            if not placed:
                bars_cuts.append([cut])
                bar_remainders.append(self.bar_length - cut)

        cuts_layout = bars_cuts
        return cuts_layout

    def __calculate_bar_remainders(self, cuts_layout: list[list[int]]) -> list[int]:
        bar_remainders: list[int] = []
        for cuts_per_bar in cuts_layout:
            used_length = sum(cuts_per_bar)
            wasted_length = self.bar_length - used_length
            bar_remainders.append(wasted_length)
        bar_remainders = bar_remainders
        return bar_remainders
