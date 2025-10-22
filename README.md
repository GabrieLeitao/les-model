# LES-model (incompressible 3D LES)

Small 3D Large-Eddy Simulation (LES) prototype written in C++.
Features:
- 3D structured grid solver (u,v,w,p) with Smagorinsky SGS model
- Explicit momentum update, pressure Poisson solver (Jacobi / SOR)
- Restart/continue from VTK output files
- OpenMP parallel loops

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
  - `c` to continue from latest `output/output_<N>.vtk` (if present)
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
- output/ : VTK outputs (ignored by .gitignore)

Makefile
- Uses `g++ -std=c++17 -fopenmp` (see `Makefile` for flags)
- To change compiler flags edit `Makefile` variable `CXXFLAGS`.

TODO
- Implement PCG with better Preconditioner
- Implement SOTA multigrid solver for Pressure
- Semi-implicit methods for momentum: Convection Explicitly, Diffusion Implicitly