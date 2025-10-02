import random
from collections import Counter

import numpy as np
import pygad
from numpy import typing as npt


def _get_fill_elements(source_parent: list[int], inherited_segment: list[int]) -> list[int]:
    """
    Helper function to get the elements from a source parent that are
    not present in the inherited segment from the other parent.
    This correctly handles repeated elements.

    Returns:
        list: An ordered list of elements from source_parent to be used
              to fill the child's remaining genes.
    """
    inherited_counts = Counter(inherited_segment)

    fill_elements: list[int] = []
    for element in source_parent:
        # If the element is in our count and the count is > 0, it means
        # this occurrence is "accounted for" in the inherited segment.
        if inherited_counts.get(element, 0) > 0:
            inherited_counts[element] -= 1
        else:
            # This element is not in the inherited segment, or all its
            # occurrences have been accounted for, so add it to the fill list.
            fill_elements.append(element)

    return fill_elements


def _create_child(
    parent_to_copy_slice: list[int], parent_to_fill_missing_elements: list[int], slice_start: int, slice_end: int
) -> list[int]:
    size = len(parent_to_copy_slice)
    child: list[int | None] = [None] * size

    # Copy the crossover segment from parent to child
    slice_from_parent = parent_to_copy_slice[slice_start : slice_end + 1]
    child[slice_start : slice_end + 1] = slice_from_parent

    # Fill the remaining slots in the child
    fill_elements = _get_fill_elements(parent_to_fill_missing_elements, slice_from_parent)
    current_pos = (slice_end + 1) % size
    fill_idx = 0
    while current_pos > slice_end or current_pos < slice_start:
        child[current_pos] = fill_elements[fill_idx]
        fill_idx += 1
        current_pos = (current_pos + 1) % size

    filled_child = [cut for cut in child if cut is not None]
    assert len(filled_child) == size
    return filled_child


def generalized_order_crossover(
    parents: npt.NDArray[np.int64],
    offspring_size: tuple[int, int],
    ga_instance: pygad.GA,
) -> npt.NDArray[np.int64]:
    """
    Performs Generalized Order Crossover (GOX) on two parents to create two children.
    This crossover is suitable for ordered chromosomes where elements can be repeated,
    such as in the Cutting Stock Problem.

    Returns:
        tuple: A tuple containing the two generated children (child1, child2).
    """
    offspring: list[list[int]] = []
    idx = 0
    while len(offspring) < offspring_size[0]:
        parent1: list[int] = parents[idx % parents.shape[0], :].tolist()
        parent2: list[int] = parents[(idx + 1) % parents.shape[0], :].tolist()

        size = len(parent1)

        # Select two random crossover points
        start, end = sorted(random.sample(range(size), 2))

        child1 = _create_child(parent1, parent2, start, end)
        child2 = _create_child(parent2, parent1, start, end)

        offspring.append(child1)
        if len(offspring) < offspring_size[0]:
            offspring.append(child2)
        idx += 1

    return np.array(offspring)
