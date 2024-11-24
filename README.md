This is a hobby project that intends to implement couple of useful and commonly used numerical methods of theoretical quantum many-body physics by leveraging the parallel GPU resources with CUDA.

For now, only the exact diagonalization algorithm for (arbitrary) Heisenberg model is implemented. The implementation supports the usage of Abelian symmetries and as such could be found perhaps useful.

The project is structured such that implementing additional methods would be straightforward such that some of the existing classes (such as symmetries, lattice models and arbitrary operators) could be made use of. The current plan is to implement next either DMRG or Variational Monte Carlo.
