# LES-model (incompressible 3D LES)

Small 3D Large-Eddy Simulation (LES) prototype written in C++.
Features:
- 3D structured grid solver (u,v,w,p) with Smagorinsky SGS model
- Explicit momentum update
- Pressure Poisson solver (Jacobi, GS SOR, PCG with Jacobi, IC and Chebyshev preconditioners)
- Restart/continue from previous step checkpoint in .data and .meta files
- OpenMP parallelization

Prerequisites
- C++17-capable compiler (g++ recommended)
- OpenMP support
- make

Build
- From project root:
  - make
  - Executable: `bin/les_solver`

Run
- Default run:
  - `make run` or `./bin/les_solver`
- On start the program prompts:
  - `c` to continue from latest `checkpoint/checkpoint.data` and `checkpoint/checkpoint.meta`(if present)
  - `r` to restart from scratch

I/O
- Output VTK files written to `output/output_<step>.vtk` (ASCII STRUCTURED_POINTS)
  - Contains velocity VECTORS, pressure SCALARS, nu_t and solid mask
- The program can read the VECTORS and pressure blocks to resume a simulation.
- To continue, keep `output/` files and start with the `c` option.

Project layout
- src/ : source files
- build/ : object files (created by Makefile)
- bin/ : executable
- checkpoint/ : checkpoint data from continuation of simulation supporting graceful termination `Ctrl + C`
- output/ : VTK outputs (ignored by .gitignore)

Makefile
- Uses `g++ -std=c++17 -fopenmp` (see `Makefile` for flags)
- To change compiler flags edit `Makefile` variable `CXXFLAGS`.

DONE:
- Graceful shutdown
- Correct checkpoint saving all Field with precision
- IC(0) preconditioner for PCG for pressure laplacian solver
- Chebyshev preconditioner for PCG for pressure laplacian solver
- Statistic for time-average flow after nsteps

TODO
- Implement SOTA multigrid solver for Pressure
- Semi-implicit methods for momentum: Convection Explicitly, Diffusion Implicitly
- Conservative equation form
- Continue to unorganized mesh and FVM