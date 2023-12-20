# Qubit_allocation_problem

You will find here group 5's solution to the Qubit Allocation Problem presented as the 2023-2024 project of UMONS's Combinatorial Optimization and Graphs class.

## Brief introduction

As explained during the presentation, we came to use a mutli-processed (AKA multi-threaded) Simulated Annealing algorithm, consisting of 2 phases :
- A diversification phase (either by Simulated Annealing or Genetic Algorithm)
- An intensification phase of the bests results obtained during the first phase.

We will first initialize $n$ threads depending on the number of instances needed to be run.
Each thread will then initialize 10 random solutions on an optimized local neighborhood of the problem (depending on the instance $m$ and $n$ parameters), and run a parallel Simulated Annealing using a high temparture and alpha, for better diversification.

Once the diversification phase is over, the best solutions of each thread will be gathered and sorted by their cost. The bests will then be used as initial solutions for a parallel Simulated Annealing using a low temperature and alpha, for better intensification.
