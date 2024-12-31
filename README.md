This is a hobby project that intends to implement couple of useful and commonly used numerical methods of theoretical quantum many-body physics by leveraging the parallel GPU resources with CUDA.

For now, only the exact diagonalization algorithm for the extended Heisenberg model is implemented. The implementation supports the usage of Abelian symmetries to reduce the memory limitations.

The project is structured such that implementing additional methods would be straightforward such that some of the existing classes (such as symmetries, lattice models and arbitrary operators) could be made use of. The current plan is to implement next either DMRG or Variational Monte Carlo. The project is WIP.

The project is built, compiled and tested by using Visual Studio 2022 on W11.


DEPENDENCIES:
- Cuda toolkit
- Armadillo (for user-friendliness of computationally non-demanding linear algebra)

Immediate TO DO:
- Fix Heisenberg model class to deal with a more generic set of operators and make it a true XXZ (now it's in practice just a XXZ with real-valued spin-spin coupling). This feature is work in progress.

  
TO DO in the longer time period:
 - Implement the Monte Carlo suite and worm algorithm
 - Implement support for bosonic and fermionic lattice models
 - Implement DMFT
 - Implement DMRG
 - Implement VMC (?)
 
 - Provide a Python interface

Tehcnial debt:
- Implement unit tests and make sure each new merge request passes the tests and does not break the code
- Overall cleaning of the code, add comments etc.
- Lack of documentation
- Get rid off Armadillo dependency
- Make logging more professional and systematic
- Move long arg lists of functions to structs
