from typing import Literal

type Chromosome = list[int]

type MutationStrategy = Literal["SwapCuts", "ReverseSubsequence"]
type CutsLayoutStrategy = Literal["InOrder", "BestFit"]
