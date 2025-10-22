#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <string>
#include <sstream>

#include "field.h"
#include "pressure.h"
#include "io_vtk.h"

// --- Function declarations (prototypes) ---
void apply_velocity_BCs(Field& f, const Mesh &m);
double compute_adaptive_dt(const Field& field, const Mesh& mesh, double nu);

void define_obstacle_mask(Field &f, const Mesh &m);

void compute_sgs_stress(struct Field& f, struct Stress& sgs, const struct Mesh& m);
void apply_nut_BCs(Field& f, const Mesh& m);
void solve_momentum(struct Field &f, const struct Mesh &m, double dt, double nu);

void apply_pressure_BCs(Field& f, const Mesh& m);

void correct_velocity(struct Field &f, const struct Mesh &m, double dt, double rho);

char get_start_choice(void);
int try_continue(Field &field, const Mesh &mesh, double &time_simulated, double dt);

/**
 * main - Main code flow of the solver.
 */
int main()
{
    Mesh  mesh{ 100, 100, 100, 0.01, 0.01, 0.01 };
    Field field(mesh.nx, mesh.ny, mesh.nz);
    int n_nodes = mesh.nx * mesh.ny * mesh.nz;
    Stress sgs(n_nodes);
    // Pressure stencil matrix
    StencilMatrix A(n_nodes);
    build_stencil(A, field, mesh);

    define_obstacle_mask(field, mesh);

    double rho    = 1.2;
    double nu     = 1.0e-5;

    double dt     = 1e-3;
    int    nSteps = 2000;
    double time_simulated = 0.0;

    // ask user and act accordingly
    int startStep = 0;
    char choice = get_start_choice();
    if (choice == 'c')
    {
        startStep = try_continue(field, mesh, time_simulated, dt);
        if (startStep == 0)
        {
            std::cout << "Continuing failed or no file found — restarting from scratch.\n";
            apply_velocity_BCs(field, mesh);
        }
    }
    else
    {
        apply_velocity_BCs(field, mesh);
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    // start loop from startStep so we continue rather than restart
    for (int step = startStep; step < nSteps; ++step)
    {
        if (step <= 20)
        {
            dt = 1e-3; 
        }
        else
        {
            dt = compute_adaptive_dt(field, mesh, nu);
        }
        // std::cout << "Step " << step << ", dt = " << dt << "\n";
        time_simulated += dt;

        auto t1 = std::chrono::high_resolution_clock::now();
        compute_sgs_stress(field, sgs, mesh);
        apply_nut_BCs(field, mesh);
        auto t2 = std::chrono::high_resolution_clock::now();
        solve_momentum(field, mesh, dt, nu);  
        apply_velocity_BCs(field, mesh);
        auto t3 = std::chrono::high_resolution_clock::now();
        // solve_pressure_jacobi(field, mesh, dt, rho);
        solve_pressure_GS(field, mesh, dt, rho, 100);
        // solve_pressure_pcg(field, mesh, A, dt, rho, 1000);
        auto t4 = std::chrono::high_resolution_clock::now();
        correct_velocity(field, mesh, dt, rho);
        auto t5 = std::chrono::high_resolution_clock::now();
 
        apply_velocity_BCs(field, mesh);

        std::chrono::duration<double> dt_sgs      = t2 - t1;
        std::chrono::duration<double> dt_momentum = t3 - t2;
        std::chrono::duration<double> dt_pressure = t4 - t3;
        std::chrono::duration<double> dt_correct  = t5 - t4;

        if(step % 100 == 0)
        {
            // #pragma omp parallel for
            // for(int i=0;i<mesh.nx*mesh.ny*mesh.nz;++i)
            // {
            //     if(std::isnan(field.u[i]) || std::isnan(field.v[i]) || std::isnan(field.w[i]) ||
            //     std::isnan(field.p[i]))
            //     {
            //         std::cerr << "NaN detected at index " << i << "\n";
            //         exit(1);
            //     }
            // }
            std::cout << "Step " << step << " times (s): "
                      << "SGS=" << dt_sgs.count()
                      << ", Momentum=" << dt_momentum.count()
                      << ", Pressure=" << dt_pressure.count()
                      << ", Correction=" << dt_correct.count()
                      << "\n";

            write_field_vtk(field, mesh, step);
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;
    
    double sec = total_elapsed.count();
    int h = static_cast<int>(sec / 3600);
    int m = static_cast<int>(std::fmod(sec, 3600) / 60);
    int s = static_cast<int>(std::fmod(sec, 60));

    std::cout << "Total simulation time: ";
    if (h > 0)
        std::cout << h << " h ";
    if (m > 0 || h > 0)  // print minutes if any hours or nonzero minutes
        std::cout << m << " min ";
    std::cout << s << " s\n";

    std::cout << "Total simulated physical time: " << time_simulated << " s\n";

    return 0;
}

/**
 * compute_adaptive_dt - Compute adaptive time step based on CFL and viscous criteria.
 * @param field Field const& : velocity field (u,v,w) to find max velocity magnitude. 
 * @param mesh  Mesh const&  : mesh geometry for dx,dy,dz
 * @param nu    double      : kinematic viscosity for viscous time step calculation
 * Returns:
 * - double : computed time step satisfying stability criteria
 * Operations:
 * - Find maximum velocity magnitude u_max in the field
 * - Compute convective time step dt_conv = C * min(dx,dy,dz) / u_max
 * - Compute viscous time step dt_visc = C * min(dx,dy,dz)^2 / nu
 * - Return the minimum of dt_conv, dt_visc, and a predefined maximum dt
 */
double compute_adaptive_dt(const Field& field, const Mesh& mesh, double nu)
{
    const double CFL_CONV = 0.4;  // Convective (0.4)
    const double CFL_VISC = 0.2;  // Viscous (0.2-0.25)
    const double DT_MAX   = 1e-4;  // Maximum allowable time step

    // --- 1. Find maximum velocity in the field ---
    double u_max_sq = 0.0;
    
    #pragma omp parallel for reduction(max:u_max_sq)
    for (int i = 0; i < field.nx * field.ny * field.nz; ++i)
    {
        double mag_sq = field.u[i]*field.u[i] + 
                        field.v[i]*field.v[i] + 
                        field.w[i]*field.w[i];
        
        u_max_sq = std::max(u_max_sq, mag_sq);
    }
    
    double u_max = std::sqrt(u_max_sq);

    // --- 2. Calculate Convective dt (CFL) ---
    double min_dx = std::min({mesh.dx, mesh.dy, mesh.dz});
    double dt_conv = std::numeric_limits<double>::max();
    
    // Avoid division by zero
    if (u_max > 1e-12) 
    {
        dt_conv = CFL_CONV * min_dx / u_max;
    }

    // --- 3. Calculate Viscous dt (Diffusion) ---
    double nu_t_max = 0.0;
    #pragma omp parallel for reduction(max:nu_t_max)
    for (int i = 0; i < field.nx * field.ny * field.nz; ++i)
    {
        nu_t_max = std::max(nu_t_max, field.nu_t[i]);
    }
    double min_dx_sq = min_dx * min_dx;
    double dt_visc = CFL_VISC * min_dx_sq / (nu + nu_t_max); // dt = C * (dx^2 / ν)

    return std::min({dt_conv, dt_visc, DT_MAX});
}

/**
 * @brief Defines a solid obstacle in the domain.
 * This function marks cells as 'solid' (true) in the f.is_solid mask.
 */
void define_obstacle_mask(Field &f, const Mesh &m)
{
    int nx = m.nx, ny = m.ny, nz = m.nz;
    const int stride_i = ny * nz;
    const int stride_j = nz;

    // --- Box near inlet ---
    int i_min_box = nx / 5;
    int i_max_box = nx / 4.5;
    int j_min_box = ny * 2 / 5;
    int j_max_box = ny * 3 / 5;
    int k_min_box = nz * 2 / 5;
    int k_max_box = nz * 4 / 5;

    #pragma omp parallel for
    for (int i = 0; i < nx; ++i) {
        const int i_base = i * stride_i;
        for (int j = 0; j < ny; ++j) {
            const int j_base = i_base + j * stride_j;
            for (int k = 0; k < nz; ++k) {
                const int id = j_base + k;

                bool in_box =
                    (i >= i_min_box && i < i_max_box &&
                     j >= j_min_box && j < j_max_box &&
                     k >= k_min_box && k < k_max_box);

                if (in_box)
                    f.is_solid[id] = true;
            }
        }
    }
}




/**
 * appy_no_slip_BCs - Enforce no-slip boundary conditions on walls.
 * @param f  Field&   : in/out; velocity fields f.u, f.v, f.w
 * @param m  Mesh const& : mesh geometry for indexing
 * Operations:
 *  - Set velocity components to zero at y=0, y=ny-1 (bottom and top walls)
 *  - Set velocity components to zero at z=0, z=nz-1 (front and back walls)
 */
void apply_velocity_BCs(Field &f, const Mesh &m)
{
    int nx = m.nx, ny = m.ny, nz = m.nz;
    // auto idx = ...; // Não precisamos mais da lambda

    // --- OTIMIZAÇÃO: Pré-calcular strides (passos) ---
    const int stride_i = ny * nz; // O "passo" para i+1
    const int stride_j = nz;      // O "passo" para j+1

    // --- 1. Paredes Exteriores ---
    
    // Inlet (X=0) - Dirichlet
    #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        // Pré-calcula (i*stride_i + j*stride_j) onde i=0
        const int j_base = j * stride_j;
        for (int k = 0; k < nz; ++k) {
            const int id = j_base + k; // Apenas uma adição
            if (f.is_solid[id]) continue;
            f.u[id] = 1.0;
            f.v[id] = 0.0;
            f.w[id] = 0.0;
        }
    }

    // Outlet (X=nx-1) - Neumann
    #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        // Pré-calcula os 'j_base' para a face (nx-1) e (nx-2)
        const int j_base_N   = (nx-1) * stride_i + j * stride_j;
        const int j_base_Nm1 = (nx-2) * stride_i + j * stride_j;
        for (int k = 0; k < nz; ++k) {
            const int id_N   = j_base_N + k;
            const int id_Nm1 = j_base_Nm1 + k;
            f.u[id_N] = f.u[id_Nm1];
            f.v[id_N] = f.v[id_Nm1];
            f.w[id_N] = f.w[id_Nm1];
        }
    }

    // Paredes Y (j=0, j=ny-1) (No-Slip)
    // (O loop 'k' é interno e rápido, esta é a otimização ideal)
    #pragma omp parallel for
    for (int i = 0; i < nx; ++i) {
        // Pré-calcula i*stride_i
        const int i_base = i * stride_i; 
        // Pré-calcula i*stride_i + j*stride_j
        const int ij_base_bot = i_base;           // j=0
        const int ij_base_top = i_base + (ny-1) * stride_j;
        for (int k = 0; k < nz; ++k) {
            const int id_bot = ij_base_bot + k;
            const int id_top = ij_base_top + k;
            f.u[id_bot] = 0.0; f.v[id_bot] = 0.0; f.w[id_bot] = 0.0;
            f.u[id_top] = 0.0; f.v[id_top] = 0.0; f.w[id_top] = 0.0;
        }
    }

    // Paredes Z (k=0, k=nz-1) (No-Slip)
    // (O loop 'j' é interno aqui, com stride_j)
    #pragma omp parallel for
    for (int i = 0; i < nx; ++i) {
        const int i_base = i * stride_i;
        for (int j = 0; j < ny; ++j) {
            // Pré-calcula i*stride_i + j*stride_j
            const int ij_base = i_base + j * stride_j;
            const int id_bot = ij_base;       // k=0
            const int id_top = ij_base + (nz-1); // k=nz-1
            f.u[id_bot] = 0.0; f.v[id_bot] = 0.0; f.w[id_bot] = 0.0;
            f.u[id_top] = 0.0; f.v[id_top] = 0.0; f.w[id_top] = 0.0;
        }
    }

    // --- 2. Obstáculo Interior (SOBRESCREVE) ---
    // (Este é o loop 3D, beneficia o máximo)
    #pragma omp parallel for
    for (int i = 1; i < nx - 1; ++i) { // Apenas interior
        const int i_base = i * stride_i;
        for (int j = 1; j < ny - 1; ++j) {
            const int j_base = i_base + j * stride_j;
            for (int k = 1; k < nz - 1; ++k) {
                const int id = j_base + k;
                if (f.is_solid[id]) {
                    f.u[id] = 0.0;
                    f.v[id] = 0.0;
                    f.w[id] = 0.0;
                }
            }
        }
    }
}

/**
 * compute_sgs_stress - compute subgrid-scale stress tensor using Smagorinsky model.
 * @param f  Field&   : input velocity fields (f.u, f.v, f.w); used to compute gradients.
 * @param sgs Stress& : output; populated with tau_ij = -2 * nu_t * S_ij for interior cells.
 * @param m  Mesh const& : grid sizes and spacings (nx,ny,nz,dx,dy,dz) used for finite differences and filter width.
 * Operations:
 *  - compute central differences of u,v,w to obtain velocity gradients
 *  - form strain-rate tensor S_ij
 *  - compute magnitude |S| and Smagorinsky eddy viscosity nu_t = (Cs*delta)^2 * |S|
 *  - set sgs.tau_* = -2 * nu_t * S_*
 */
void compute_sgs_stress(Field& f, Stress& sgs, const Mesh& m)
{
    double Cs = 0.18;
    double delta = std::pow(m.dx * m.dy * m.dz, 1.0 / 3.0);
    
    #pragma omp parallel for
    for (int i = 1; i < m.nx - 1; i++)
    for (int j = 1; j < m.ny - 1; j++)
    for (int k = 1; k < m.nz - 1; k++)
    {
        int id = f.idx(i, j, k);

        // --- Compute gradients ---
        double du_dx = (f.u[f.idx(i+1,j,k)] - f.u[f.idx(i-1,j,k)]) / (2*m.dx);
        double du_dy = (f.u[f.idx(i,j+1,k)] - f.u[f.idx(i,j-1,k)]) / (2*m.dy);
        double du_dz = (f.u[f.idx(i,j,k+1)] - f.u[f.idx(i,j,k-1)]) / (2*m.dz);

        double dv_dx = (f.v[f.idx(i+1,j,k)] - f.v[f.idx(i-1,j,k)]) / (2*m.dx);
        double dv_dy = (f.v[f.idx(i,j+1,k)] - f.v[f.idx(i,j-1,k)]) / (2*m.dy);
        double dv_dz = (f.v[f.idx(i,j,k+1)] - f.v[f.idx(i,j,k-1)]) / (2*m.dz);

        double dw_dx = (f.w[f.idx(i+1,j,k)] - f.w[f.idx(i-1,j,k)]) / (2*m.dx);
        double dw_dy = (f.w[f.idx(i,j+1,k)] - f.w[f.idx(i,j-1,k)]) / (2*m.dy);
        double dw_dz = (f.w[f.idx(i,j,k+1)] - f.w[f.idx(i,j,k-1)]) / (2*m.dz);

        // --- Strain-rate tensor components ---
        double Sxx = du_dx;
        double Syy = dv_dy;
        double Szz = dw_dz;
        double Sxy = 0.5 * (du_dy + dv_dx);
        double Sxz = 0.5 * (du_dz + dw_dx);
        double Syz = 0.5 * (dv_dz + dw_dy);

        // --- Strain magnitude ---
        double Smag = std::sqrt(2.0 * (Sxx*Sxx + Syy*Syy + Szz*Szz)
                              + 4.0 * (Sxy*Sxy + Sxz*Sxz + Syz*Syz));

        // --- Eddy viscosity ---
        double nu_t = std::pow(Cs * delta, 2) * Smag;
        f.nu_t[id] = nu_t; // store eddy viscosity

        // --- Subgrid stress tensor ---
        sgs.tau_xx[id] = -2.0 * nu_t * Sxx;
        sgs.tau_yy[id] = -2.0 * nu_t * Syy;
        sgs.tau_zz[id] = -2.0 * nu_t * Szz;
        sgs.tau_xy[id] = -2.0 * nu_t * Sxy;
        sgs.tau_xz[id] = -2.0 * nu_t * Sxz;
        sgs.tau_yz[id] = -2.0 * nu_t * Syz;
    }
}

/**
 * apply_nut_BCs - Apply boundary conditions to eddy viscosity field (f.nu_t).
 * @param f  Field&   : in/out; eddy viscosity field f.nu_t
 * @param m  Mesh const& : mesh geometry for indexing
 * Operations:
 *  - Set nu_t = 0 at walls (y=0, y=ny-1, z=0, z=nz-1)
 *  - Set nu_t = 0 at inlet (x=0)
 *  - Apply zero-Neumann at outlet (x=nx-1)
 */
void apply_nut_BCs(Field &f, const Mesh &m)
{
    int nx = m.nx, ny = m.ny, nz = m.nz;
    const int stride_i = ny * nz;
    const int stride_j = nz;

    // --- 1. Outer Boundaries ---

    // Inlet (X=0) - Dirichlet
    #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        const int j_base = j * stride_j;
        for (int k = 0; k < nz; ++k) {
            f.nu_t[j_base + k] = 0.0;
        }
    }

    // Outlet (X=nx-1) - Neumann
    #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        const int j_base_N   = (nx - 1) * stride_i + j * stride_j;
        const int j_base_Nm1 = (nx - 2) * stride_i + j * stride_j;
        for (int k = 0; k < nz; ++k) {
            f.nu_t[j_base_N + k] = f.nu_t[j_base_Nm1 + k];
        }
    }

    // Y-Walls (j=0, j=ny-1) - Dirichlet
    #pragma omp parallel for
    for (int i = 0; i < nx; ++i) {
        const int i_base = i * stride_i;
        const int ij_base_bot = i_base;
        const int ij_base_top = i_base + (ny - 1) * stride_j;
        for (int k = 0; k < nz; ++k) {
            f.nu_t[ij_base_bot + k] = 0.0;
            f.nu_t[ij_base_top + k] = 0.0;
        }
    }

    // Z-Walls (k=0, k=nz-1) - Dirichlet
    #pragma omp parallel for
    for (int i = 0; i < nx; ++i) {
        const int i_base = i * stride_i;
        for (int j = 0; j < ny; ++j) {
            const int ij_base = i_base + j * stride_j;
            f.nu_t[ij_base] = 0.0;           // k=0
            f.nu_t[ij_base + (nz - 1)] = 0.0; // k=nz-1
        }
    }

    // --- 2. Inner Obstacle ---
    #pragma omp parallel for
    for (int i = 1; i < nx - 1; ++i) {
        const int i_base = i * stride_i;
        for (int j = 1; j < ny - 1; ++j) {
            const int j_base = i_base + j * stride_j;
            for (int k = 1; k < nz - 1; ++k) {
                const int id = j_base + k;
                if (f.is_solid[id]) {
                    f.nu_t[id] = 0.0; // Turbulence dies at the wall
                }
            }
        }
    }
}

/**
 * solve_momentum - advance provisional velocity field (explicit Euler).
 * @param f  Field&   : in/out; reads current velocities and f.nu_t (eddy viscosity), updates f.u,f.v,f.w with new provisional values.
 * @param m  Mesh const& : grid geometry for derivatives and Laplacian.
 * @param dt double   : time step for explicit update.
 * @param nu double   : molecular viscosity (added to eddy viscosity to form nu_eff).
 * Operations:
 *  - compute central velocity gradients and advective terms
 *  - form effective viscosity nu_eff = nu + f.nu_t[id]
 *  - compute Laplacian (diffusion) for each component
 *  - update provisional velocities: u_new = u + dt * (-adv + nu_eff * lap)
 *  - enforce no-slip (zero) boundary values
 */
// A função agora usa os arrays "new" que já existem em 'f'
void solve_momentum(Field &f, const Mesh &m, double dt, double nu = 1e-5)
{
    int nx = m.nx, ny = m.ny, nz = m.nz;
    auto idx = [&](int i, int j, int k) { return f.idx(i, j, k); };

    // --- Optimization: Precompute constants ---
    const double inv_dx2 = 1.0 / (m.dx * m.dx);
    const double inv_dy2 = 1.0 / (m.dy * m.dy);
    const double inv_dz2 = 1.0 / (m.dz * m.dz);
    const double inv_2dx = 1.0 / (2.0 * m.dx);
    const double inv_2dy = 1.0 / (2.0 * m.dy);
    const double inv_2dz = 1.0 / (2.0 * m.dz);

    
    // --- Cache Optimization: LOOP 2 V ---
    #pragma omp parallel for
    for (int i = 1; i < nx - 1; ++i)
    for (int j = 1; j < ny - 1; ++j)
    for (int k = 1; k < nz - 1; ++k)
    {
        int id = idx(i,j,k);
        double nu_eff = nu + f.nu_t[id]; // nu_t local

        // Gradients
        double du_dx = (f.u[idx(i+1,j,k)] - f.u[idx(i-1,j,k)]) * inv_2dx;
        double du_dy = (f.u[idx(i,j+1,k)] - f.u[idx(i,j-1,k)]) * inv_2dy;
        double du_dz = (f.u[idx(i,j,k+1)] - f.u[idx(i,j,k-1)]) * inv_2dz;

        // Advection
        double adv_u = f.u[id]*du_dx + f.v[id]*du_dy + f.w[id]*du_dz;

        // Diffusion
        double lap_u = (f.u[idx(i+1,j,k)] - 2*f.u[id] + f.u[idx(i-1,j,k)]) * inv_dx2
                     + (f.u[idx(i,j+1,k)] - 2*f.u[id] + f.u[idx(i,j-1,k)]) * inv_dy2
                     + (f.u[idx(i,j,k+1)] - 2*f.u[id] + f.u[idx(i,j,k-1)]) * inv_dz2;

        // Update
        f.u_new[id] = f.u[id] + dt * (-adv_u + nu_eff * lap_u);
    }

    // --- Cache Optimization: LOOP 2 V ---
    #pragma omp parallel for
    for (int i = 1; i < nx - 1; ++i)
    for (int j = 1; j < ny - 1; ++j)
    for (int k = 1; k < nz - 1; ++k)
    {
        int id = idx(i,j,k);
        double nu_eff = nu + f.nu_t[id];

        // Gradients
        double dv_dx = (f.v[idx(i+1,j,k)] - f.v[idx(i-1,j,k)]) * inv_2dx;
        double dv_dy = (f.v[idx(i,j+1,k)] - f.v[idx(i,j-1,k)]) * inv_2dy;
        double dv_dz = (f.v[idx(i,j,k+1)] - f.v[idx(i,j,k-1)]) * inv_2dz;
        
        // Advection
        double adv_v = f.u[id]*dv_dx + f.v[id]*dv_dy + f.w[id]*dv_dz;
        
        // Diffusion
        double lap_v = (f.v[idx(i+1,j,k)] - 2*f.v[id] + f.v[idx(i-1,j,k)]) * inv_dx2
                     + (f.v[idx(i,j+1,k)] - 2*f.v[id] + f.v[idx(i,j-1,k)]) * inv_dy2
                     + (f.v[idx(i,j,k+1)] - 2*f.v[id] + f.v[idx(i,j,k-1)]) * inv_dz2;

        // Update
        f.v_new[id] = f.v[id] + dt * (-adv_v + nu_eff * lap_v);
    }

    // --- Cache Optimization: LOOP 3 W ---
    #pragma omp parallel for
    for (int i = 1; i < nx - 1; ++i)
    for (int j = 1; j < ny - 1; ++j)
    for (int k = 1; k < nz - 1; ++k)
    {
        int id = idx(i,j,k);
        double nu_eff = nu + f.nu_t[id];

        // Gradients
        double dw_dx = (f.w[idx(i+1,j,k)] - f.w[idx(i-1,j,k)]) * inv_2dx;
        double dw_dy = (f.w[idx(i,j+1,k)] - f.w[idx(i,j-1,k)]) * inv_2dy;
        double dw_dz = (f.w[idx(i,j,k+1)] - f.w[idx(i,j,k-1)]) * inv_2dz;

        // Advection
        double adv_w = f.u[id]*dw_dx + f.v[id]*dw_dy + f.w[id]*dw_dz;

        // Diffusion
        double lap_w = (f.w[idx(i+1,j,k)] - 2*f.w[id] + f.w[idx(i-1,j,k)]) * inv_dx2
                     + (f.w[idx(i,j+1,k)] - 2*f.w[id] + f.w[idx(i,j-1,k)]) * inv_dy2
                     + (f.w[idx(i,j,k+1)] - 2*f.w[id] + f.w[idx(i,j,k-1)]) * inv_dz2;
        
        // Update
        f.w_new[id] = f.w[id] + dt * (-adv_w + nu_eff * lap_w);
    }

    // Optimização: pointer swap much better than copying arrays
    f.u.swap(f.u_new);
    f.v.swap(f.v_new);
    f.w.swap(f.w_new);
}

/**
 * correct_velocity - apply pressure-gradient correction to provisional velocities.
 * @param f  Field&   : in/out; reads pressure f.p and updates velocities f.u,f.v,f.w (interior cells).
 * @param m  Mesh const& : grid geometry for computing central pressure gradients.
 * @param dt double   : time step used in the correction factor.
 * @param rho double  : fluid density used in the correction factor.
 * Operations:
 *  - compute central differences of pressure to obtain dp/dx, dp/dy, dp/dz
 *  - subtract correction: u -= (dt/rho) * dp_dx (and similarly for v,w) for interior cells
 */
void correct_velocity(Field &f, const Mesh &m, double dt, double rho)
{
    int nx = m.nx, ny = m.ny, nz = m.nz;
    auto idx = [&](int i,int j,int k){ return f.idx(i,j,k); };

    // --- Precompute constants ---
    const double dt_over_rho = dt / rho;
    const double inv_2dx = 1.0 / (2.0 * m.dx);
    const double inv_2dy = 1.0 / (2.0 * m.dy);
    const double inv_2dz = 1.0 / (2.0 * m.dz);

    // --- LOOP 1: Correct U ---
    #pragma omp parallel for
    for (int i = 1; i < nx - 1; ++i)
    for (int j = 1; j < ny - 1; ++j)
    for (int k = 1; k < nz - 1; ++k)
    {
        int id = idx(i,j,k);
        double dp_dx = (f.p[idx(i+1,j,k)] - f.p[idx(i-1,j,k)]) * inv_2dx;
        f.u[id] -= dt_over_rho * dp_dx;
    }

    // --- LOOP 2: Correct V ---
    #pragma omp parallel for
    for (int i = 1; i < nx - 1; ++i)
    for (int j = 1; j < ny - 1; ++j)
    for (int k = 1; k < nz - 1; ++k)
    {
        int id = idx(i,j,k);
        double dp_dy = (f.p[idx(i,j+1,k)] - f.p[idx(i,j-1,k)]) * inv_2dy;
        f.v[id] -= dt_over_rho * dp_dy;
    }

    // --- LOOP 3: Correct W ---
    #pragma omp parallel for
    for (int i = 1; i < nx - 1; ++i)
    for (int j = 1; j < ny - 1; ++j)
    for (int k = 1; k < nz - 1; ++k)
    {
        int id = idx(i,j,k);
        double dp_dz = (f.p[idx(i,j,k+1)] - f.p[idx(i,j,k-1)]) * inv_2dz;
        f.w[id] -= dt_over_rho * dp_dz;
    }
}


/**
 * get_start_choice - prompt user for start option (continue or restart).
 * Returns 'c' for continue, 'r' for restart (default 'c' on invalid input).    
 */
char get_start_choice(void)
{
    std::cout << "Start option: (c)ontinue from latest output, (r)estart from scratch [c/r] ? ";
    char choice = 'c';
    std::cin >> choice;
    if (choice != 'c' && choice != 'r') choice = 'c';
    return choice;
}

/**
 * try_continue - attempt to load latest output file to continue simulation.
 * @param field Field& : in/out; field to populate with loaded data
 * @param mesh  Mesh const& : mesh geometry for validation
 * @param time_simulated double& : out; updated with time corresponding to loaded step
 * @param dt    double : time step size used to estimate time from step number
 * Returns starting step number (1 + loaded step) if successful, 0 if no valid file found.
 */
int try_continue(Field &field, const Mesh &mesh, double &time_simulated, double dt)
{
    int stepnum = 0; 
    std::string latest = find_latest_output("output", stepnum);
    if (latest.empty()) return 0;
    bool ok = read_field_vtk(latest, field, mesh);
    if (!ok) return 0;

    int startStep = stepnum + 1;
    time_simulated = startStep * dt; // approximate; dt may adapt later
    std::cout << "Loaded state from " << latest << ", continuing at step " << startStep << "\n";
    return startStep;
}