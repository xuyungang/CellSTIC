# Simulated Cell–Cell Communication Data

These spatial RNA/ATAC simulations were generated with
[scMultiSim](https://www.nature.com/articles/s41592-025-02651-0) for
ligand–receptor communication benchmarking. There are 15 datasets: three
spatial scenarios, each with five simulation seeds (11, 23, 37,
51, and 73).

| Scenario | Spatial pattern | Relative noise |
|---|---|---|
| Domains | Discrete domains and local contact signaling | Low |
| Gradients | Layered interfaces, expression gradients, and diffusible signaling | Medium |
| Heterogeneous | Mixed niches, rare cell states, and heterogeneous signaling | High |

Each dataset has 2,000 paired spatial observations, 2,000 RNA features, 6,000
ATAC features, 10 parent cell types with two states each, and 25 simulated
ligand–receptor pairs. The original stored LR design is
`sparse_within_pair_on_off_v1`: each pair is active in two or eight directed
sender/receiver state contexts, with other contexts serving as inactive
comparisons. Ground truth comprises one sparse, directed sender-by-receiver
matrix per LR pair. There are no separate decoy LR pairs.
