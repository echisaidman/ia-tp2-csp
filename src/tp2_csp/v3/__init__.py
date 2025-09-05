from .algorithm import CSPGeneticAlgorithmV3, Individual


def print_best_solution(best_solution: Individual) -> None:
    """
    Prints the details of the best solution found.
    """
    print("\n=================================")
    print("      Best Solution Found (v3)   ")
    print("=================================")
    if not best_solution.is_valid:
        print("NOTE: This is an INVALID solution.")
        print("The quantities of cuts do not match the requirements.")
        print("---")

    print(f"Bars Required: {best_solution.get_required_bars()}")
    print(f"Total Waste: {best_solution.total_waste} units")

    # Calculate and display deviation from required cuts
    print("\nCut Quantities (Produced vs. Required):")
    total_deviation = 0
    for i, produced_qty in enumerate(best_solution.chromosome):
        cut_type = best_solution.cut_types[i]
        required_qty = best_solution.required_quantities[i]
        deviation = produced_qty - required_qty
        total_deviation += abs(deviation)
        if deviation != 0:
            print(
                f"  - Cut {cut_type:4d}: {produced_qty:3d} produced, {required_qty:3d} required (Diff: {deviation:3d})"
            )

    if total_deviation == 0:
        print("  All cut quantities match requirements perfectly.")

    print("\n---------------------------------\n")

    for i, bar in enumerate(best_solution.cuts_layout):
        bar_sum = sum(bar)
        waste = best_solution.bar_length - bar_sum
        print(f"Bar {i + 1:2d} (Length: {best_solution.bar_length}):")
        print(f"  Cuts:  {bar}")
        print(f"  Sum:   {bar_sum}")
        print(f"  Waste: {waste}\n")


def run_v3() -> None:
    BAR_LENGTH = 5600

    # Define the required cuts: {length: quantity}
    REQUIRED_CUTS = {
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

    # --- GA Parameters ---
    POPULATION_SIZE = 300
    GENERATIONS = 500  # May need more generations for this approach
    TOURNAMENT_SIZE = 5
    MUTATION_RATE = 0.60
    CROSSOVER_RATE = 0.9
    ELITISM_SIZE = 5

    # This is a critical parameter to tune. It determines how heavily
    # invalid solutions are penalized. A higher value forces the
    # algorithm to find valid solutions faster.
    PENALTY_FACTOR = 1000.0

    # --- Run the Algorithm ---
    solver = CSPGeneticAlgorithmV3(
        REQUIRED_CUTS,
        BAR_LENGTH,
        POPULATION_SIZE,
        GENERATIONS,
        CROSSOVER_RATE,
        MUTATION_RATE,
        ELITISM_SIZE,
        TOURNAMENT_SIZE,
        PENALTY_FACTOR,
    )
    solution = solver.run()
    print_best_solution(solution)
