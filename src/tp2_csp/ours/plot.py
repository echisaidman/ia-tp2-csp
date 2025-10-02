import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .result import SimulationResult


def create_plot_for_run(result: SimulationResult) -> tuple[Figure, Axes]:
    x_values = list(range(1, result.generations + 1))
    y_values = [solution.calculate_required_bars() for solution in result.best_solutions_by_generation]

    fig, ax = plt.subplots(
        figsize=(40, 20),
    )
    ax.tick_params(axis="both", labelsize=32)

    ax.plot(
        x_values,
        y_values,
        # marker="o",
        linestyle="-",
        color="blue",
    )

    ax.set_xlabel("Generación", fontsize=32)
    ax.set_ylabel("Barras usadas", fontsize=32)

    str_parameters = str(result.parameters).split(" ")
    first_paragraph = " ".join(str_parameters[: len(str_parameters) // 2])
    second_paragraph = " ".join(str_parameters[len(str_parameters) // 2 :])
    title = f"{first_paragraph}\n{second_paragraph}"
    ax.set_title(title, fontsize=32)

    ax.grid(True)

    fig.tight_layout()

    return fig, ax
