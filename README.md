This is a hobby project that intends to implement couple of useful and commonly used numerical methods of theoretical quantum many-body physics by leveraging the parallel GPU resources with CUDA.

For now, only the exact diagonalization algorithm for (arbitrary) Heisenberg model is implemented. The implementation supports the usage of Abelian symmetries and as such could be found perhaps useful.

The project is structured such that implementing additional methods would be straightforward such that some of the existing classes (such as symmetries, lattice models and arbitrary operators) could be made use of. The current plan is to implement next either DMRG or Variational Monte Carlo. The project is WIP.

DEPENDENCIES:
- Cuda toolkit
- Armadillo (for user-friendliness of computationally non-demanding linear algebra)
- Built, compiled and tested by using Visual Studio 2022 on W11.

Immediate TO DO:
- Implement DMRG
- Fix Heisenberg model class to deal with a more generic set of operators and make it a real XYZ (now it's XXZ)
  
TO DO in the longer time period:
 - Implement VMC
 - Implement support for bosonic and fermionic lattice models

Tehcnial debt:
- Get rid off Armadillo dependency
- Make logging more systematic
