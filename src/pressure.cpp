#include "pressure.h"

/**
 * apply_pressure_BCs - apply boundary conditions to pressure field.
 * @param f  Field&   : in/out; pressure field f.p
 * @param nx int      : number of grid points in x direction
 * @param ny int      : number of grid points in y direction
 * @param nz int      : number of grid points in z direction
 * Operations:
 *  - Enforce zero-Neumann (zero normal gradient) on Y and Z walls
 *  - Enforce zero-Neumann at inlet (x=0)
 *  - Enforce Dirichlet (p=0) at outlet (x=nx-1) for stability
 */
void apply_pressure_BCs(Field &f, const Mesh &m)
{
    int nx = m.nx, ny = m.ny, nz = m.nz;
    const int stride_i = ny * nz;
    const int stride_j = nz;

    // --- 1. Outer Boundaries ---

    // Inlet (X=0) - Neumann
    #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        const int j_base = j * stride_j;
        for (int k = 0; k < nz; ++k) {
            const int id_0 = j_base + k;       // i=0
            const int id_1 = id_0 + stride_i;  // i=1
            f.p[id_0] = f.p[id_1];
        }
    }

    // Outlet (X=nx-1) - Dirichlet
    #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        const int j_base = (nx - 1) * stride_i + j * stride_j;
        for (int k = 0; k < nz; ++k) {
            f.p[j_base + k] = 0.0;
        }
    }

    // Y-Walls (j=0, j=ny-1) - Neumann
    #pragma omp parallel for
    for (int i = 0; i < nx; ++i) {
        const int i_base = i * stride_i;
        const int ij_base_bot = i_base;                      // j=0
        const int ij_base_top = i_base + (ny - 1) * stride_j;
        const int ij_base_bot_neighbor = i_base + stride_j;  // j=1
        const int ij_base_top_neighbor = i_base + (ny - 2) * stride_j;

        for (int k = 0; k < nz; ++k) {
            f.p[ij_base_bot + k] = f.p[ij_base_bot_neighbor + k];
            f.p[ij_base_top + k] = f.p[ij_base_top_neighbor + k];
        }
    }

    // Z-Walls (k=0, k=nz-1) - Neumann
    #pragma omp parallel for
    for (int i = 0; i < nx; ++i) {
        const int i_base = i * stride_i;
        for (int j = 0; j < ny; ++j) {
            const int ij_base = i_base + j * stride_j;
            const int id_0 = ij_base;      // k=0
            const int id_1 = ij_base + 1;  // k=1
            const int id_N   = ij_base + (nz - 1);
            const int id_Nm1 = ij_base + (nz - 2);
            f.p[id_0] = f.p[id_1];
            f.p[id_N] = f.p[id_Nm1];
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
                    f.p[id] = f.p[id + stride_i]; // Neumann (e.g., copy from i+1)
                }
            }
        }
    }
}

/**
 * solve_pressure_jacobi - solve pressure Poisson equation (Jacobi) to reduce velocity divergence.
 * @param f  Field&   : in/out; reads provisional velocities (f.u,f.v,f.w) and pressure f.p, updates f.p.
 * @param m  Mesh const& : grid spacings/sizes for Laplacian and divergence operators.
 * @param dt double   : time step used to scale RHS (rho/dt factor).
 * @param rho double  : fluid density used in RHS scaling.
 * Operations:
 *  - compute divergence of provisional velocity field
 *  - perform a fixed number of Jacobi iterations to update pressure
 *  - swap updated pressure into f.p and enforce Neumann (zero-normal-gradient) boundary conditions
 */
void solve_pressure_jacobi(Field &f, const Mesh &m, double dt, double rho)
{
    int nx = m.nx, ny = m.ny, nz = m.nz;
    auto idx = [&](int i,int j,int k){ return f.idx(i,j,k); };

    Vector p_new(f.p.size());
    Vector rhs(f.p.size());

    double coef = rho / dt;
    double dx2 = m.dx*m.dx, dy2 = m.dy*m.dy, dz2 = m.dz*m.dz;
    double denom = 2*(1.0/dx2 + 1.0/dy2 + 1.0/dz2);

    // Precompute RHS (divergence of provisional velocity)
    #pragma omp parallel for
    for (int i = 1; i < nx-1; ++i)
    for (int j = 1; j < ny-1; ++j)
    for (int k = 1; k < nz-1; ++k)
    {
        int id = idx(i,j,k);
        // divergence of provisional velocity u*
        double div = (f.u[idx(i+1,j,k)] - f.u[idx(i-1,j,k)]) / (2*m.dx)
                   + (f.v[idx(i,j+1,k)] - f.v[idx(i,j-1,k)]) / (2*m.dy)
                   + (f.w[idx(i,j,k+1)] - f.w[idx(i,j,k-1)]) / (2*m.dz);
        
        // RHS of Poisson equation
        rhs[id] = coef * div;
    }

    for (int iter = 0; iter < 100; ++iter)
    {
        double maxResidual = 0.0;

        #pragma omp parallel for reduction(max:maxResidual)
        for (int i = 1; i < nx-1; ++i)
        for (int j = 1; j < ny-1; ++j)
        for (int k = 1; k < nz-1; ++k)
        {
            int id = idx(i,j,k);
            double p_old = f.p[id];

            // Jacobi update
            p_new[id] = (
                (f.p[idx(i+1,j,k)] + f.p[idx(i-1,j,k)]) / dx2 +
                (f.p[idx(i,j+1,k)] + f.p[idx(i,j-1,k)]) / dy2 +
                (f.p[idx(i,j,k+1)] + f.p[idx(i,j,k-1)]) / dz2 -
                rhs[id]
            ) / denom;

            // compute residual
            double res = std::abs(p_new[id] - p_old);
            if(res > maxResidual) maxResidual = res;
        }
        // std::cout << "Iteration " << iter << ", max residual = " << maxResidual << "\n";

        f.p.swap(p_new);

        apply_pressure_BCs(f, m);
    
        // check convergence
        if(maxResidual < 1e-6)
        {
            // std::cout << "Pressure converged in " << iter+1 << " iterations, residual = "
            //             << maxResidual << "\n";
            break;
        }
    }
}

/**
 * solve_pressure - solve pressure Poisson equation using Gauss-Seidel iteration to reduce velocity divergence.
 * @param f  Field&   : in/out; reads provisional velocities (f.u,f.v,f.w) and pressure f.p, updates f.p.
 * @param m  Mesh const& : grid spacings/sizes for Laplacian and divergence operators.
 * @param dt double   : time step used to scale RHS (rho/dt factor).
 * @param rho double  : fluid density used in RHS scaling.
 * @param maxIter int : maximum number of Gauss-Seidel iterations.
 * Operations:
 *  - compute divergence of provisional velocity field
 *  - perform Gauss-Seidel iterations to update pressure
 *  - enforce Neumann (zero-normal-gradient) boundary conditions
 */
void solve_pressure_GS(Field &f, const Mesh &m, double dt, double rho, int maxIter=200)
{
    int nx = m.nx, ny = m.ny, nz = m.nz;
    auto idx = [&](int i,int j,int k){ return f.idx(i,j,k); };

    Vector rhs(f.p.size());

    double dx2 = m.dx*m.dx, dy2 = m.dy*m.dy, dz2 = m.dz*m.dz;
    double coef = 1.0 / (2.0*(1.0/dx2 + 1.0/dy2 + 1.0/dz2));
    double omega = 1.7; // relaxation factor

    #pragma omp parallel for
    for (int i = 1; i < nx-1; ++i)
    for (int j = 1; j < ny-1; ++j)
    for (int k = 1; k < nz-1; ++k)
    {
        int id = idx(i,j,k);
        // divergence of provisional velocity u*
        double div = (f.u[idx(i+1,j,k)] - f.u[idx(i-1,j,k)]) / (2*m.dx)
                   + (f.v[idx(i,j+1,k)] - f.v[idx(i,j-1,k)]) / (2*m.dy)
                   + (f.w[idx(i,j,k+1)] - f.w[idx(i,j,k-1)]) / (2*m.dz);
        
        // RHS of Poisson equation
        rhs[id] = rho/dt * div;
    }

    for(int iter = 0; iter < maxIter; ++iter)
    {
        double maxResidual = 0.0;

        for(int i = 1; i < nx-1; ++i)
        for(int j = 1; j < ny-1; ++j)
        for(int k = 1; k < nz-1; ++k)
        {
            int id = idx(i,j,k);
            // old pressure for residual computation
            double p_old = f.p[id];

            f.p[id] = (1-omega)*f.p[id] + omega * coef * (
                (f.p[idx(i+1,j,k)] + f.p[idx(i-1,j,k)])/dx2 +
                (f.p[idx(i,j+1,k)] + f.p[idx(i,j-1,k)])/dy2 +
                (f.p[idx(i,j,k+1)] + f.p[idx(i,j,k-1)])/dz2 -
                rhs[id]
            ); // SOR update

            // compute residual
            double res = std::abs(f.p[id] - p_old);
            if(res > maxResidual) maxResidual = res;
        }
        // std::cout << "Iteration " << iter << ", max residual = " << maxResidual << "\n";

        apply_pressure_BCs(f, m);

        // check convergence
        if(maxResidual < 1e-6)
        {
            std::cout << "Pressure converged in " << iter+1 << " iterations, residual = "
                      << maxResidual << "\n";
            break;
        }
    }
}

/**
 * @brief (RUN EVERY STEP) Calculates the divergence of the provisional velocity.
 * Computes f = (rho/dt) * div(u_new)
 *
 * This version is SAFE and prevents segfaults by using one-sided
 * formulas at the boundaries.
 */
void calculate_divergence(Vector& rhs, const Field& f, const Mesh& m, 
                          double dt, double rho)
{
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    const double scale = rho / dt;

    // 1 / (2*dx) for central differences
    const double idx = 1.0 / (2.0 * m.dx);
    const double idy = 1.0 / (2.0 * m.dy);
    const double idz = 1.0 / (2.0 * m.dz);
    
    // 1 / dx for one-sided differences
    const double idx_1s = 1.0 / m.dx;
    const double idy_1s = 1.0 / m.dy;
    const double idz_1s = 1.0 / m.dz;

    #pragma omp parallel for
    for (int i = 0; i < nx; ++i) { // Loop 0..nx-1
    for (int j = 0; j < ny; ++j) { // Loop 0..ny-1
    for (int k = 0; k < nz; ++k) { // Loop 0..nz-1
        
        const int id = f.idx(i, j, k);

        // --- Skip Solid and Dirichlet (Outlet) cells ---
        // Their RHS is set by build_b_vector, not by divergence.
        if (f.is_solid[id] || i == nx - 1) {
            rhs[id] = 0.0;
            continue;
        }

        // --- Calculate divergence for all fluid cells ---
        double du_dx, dv_dy, dw_dz;

        // --- X-Direction ---
        // This 'if' block prevents the crash at i=0
        if (i == 0) { // Inlet (u=1) - Use 1st-order forward
            // Accesses i and i+1. SAFE.
            du_dx = (f.u_new[f.idx(i+1, j, k)] - f.u_new[id]) * idx_1s;
        } else { // Interior (i > 0)
            // Accesses i+1 and i-1. SAFE.
            du_dx = (f.u_new[f.idx(i+1, j, k)] - f.u_new[f.idx(i-1, j, k)]) * idx;
        }

        // --- Y-Direction ---
        // This 'if/else' block prevents the crash at j=0 and j=ny-1
        if (j == 0) { // Bottom Wall (v=0) - Use 1st-order forward
            // Accesses j and j+1. SAFE.
            // (Assumes v_new[id] == 0 from apply_velocity_BCs)
            dv_dy = (f.v_new[f.idx(i, j+1, k)] - f.v_new[id]) * idy_1s; 
        } else if (j == ny - 1) { // Top Wall (v=0) - Use 1st-order backward
            // Accesses j and j-1. SAFE.
            // (Assumes v_new[id] == 0 from apply_velocity_BCs)
            dv_dy = (f.v_new[id] - f.v_new[f.idx(i, j-1, k)]) * idy_1s;
        } else { // Interior
            // Accesses j+1 and j-1. SAFE.
            dv_dy = (f.v_new[f.idx(i, j+1, k)] - f.v_new[f.idx(i, j-1, k)]) * idy;
        }

        // --- Z-Direction ---
        // This 'if/else' block prevents the crash at k=0 and k=nz-1
        if (k == 0) { // Front Wall (w=0) - Use 1st-order forward
            // Accesses k and k+1. SAFE.
            dw_dz = (f.w_new[f.idx(i, j, k+1)] - f.w_new[id]) * idz_1s; // w_new[id] is 0
        } else if (k == nz - 1) { // Back Wall (w=0) - Use 1st-order backward
            // Accesses k and k-1. SAFE.
            dw_dz = (f.w_new[id] - f.w_new[f.idx(i, j, k-1)]) * idz_1s; // w_new[id] is 0
        } else { // Interior
            // Accesses k+1 and k-1. SAFE.
            dw_dz = (f.w_new[f.idx(i, j, k+1)] - f.w_new[f.idx(i, j, k-1)]) * idz;
        }
        
        rhs[id] = scale * (du_dx + dv_dy + dw_dz);
    }}}
}

/**
 * solve_pressure_pcg - solve pressure Poisson equation using Preconditioned Conjugate Gradient.
 */
void solve_pressure_pcg(Field &f, const Mesh &m, StencilMatrix &A, double dt, 
                        double rho, int maxIter, double tol)
{
    const int n_nodes = m.nx * m.ny * m.nz;
    
    calculate_divergence(f.p_rhs, f, m, dt, rho);

    Vector b(n_nodes);
    build_b_vector(b, f.p_rhs, f, m);

    // 2. Solve the pressure equation: A*p = b
    solve_pcg(A, b, f.p, f, maxIter, tol);
}

/**
 * @brief Solves the linear system A*x = b using PCG with Jacobi preconditioner.
 *
 * @param A       The pre-built StencilMatrix
 * @param b       The right-hand-side vector (from build_b_vector)
 * @param x       Input: Initial guess (field.p). Output: The solution.
 * @param field   Const reference to the field (for fast_apply_laplacian)
 * @param max_iter Max iterations
 * @param tol     Convergence tolerance
 * @return int    Number of iterations performed
 */
int solve_pcg(const StencilMatrix& A, const Vector& b, Vector& x,
              const Field& field, int max_iter, double tol)
{
    const int n_nodes = field.nx * field.ny * field.nz;
    
    // 1. Allocate temporary vectors
    Vector r(n_nodes); // Residual (r = b - A*x)
    Vector z(n_nodes); // Preconditioned residual (z = M_inv * r)
    Vector p(n_nodes); // Search direction
    Vector q(n_nodes); // A * p

    // 2. Initial calculation
    // r = b - A*x
    fast_apply_laplacian(q, x, A, field); // q = A*x
    parallel_saxpy(r, 1.0, b, -1.0, q); // r = 1.0*b - 1.0*q

    double initial_error = parallel_norm(r);
    if (initial_error < tol) return 0; // Already converged

    // z = M_inv * r (Jacobi preconditioner, where M is diag(A))
    apply_jacobi_preconditioner(p, r, A);

    // p = z
    // parallel_copy(p, z);

    // rho = r . z
    double rho = parallel_dot(r, p);

    // 3. Main PCG Iteration Loop
    for (int k = 1; k <= max_iter; ++k)
    {
        // q = A * p
        fast_apply_laplacian(q, p, A, field);

        // alpha = rho / (p . q)
        double alpha = rho / parallel_dot(p, q);

        // x = x + alpha * p
        parallel_axpy(x, alpha, p);

        // r = r - alpha * q
        parallel_axpy(r, -alpha, q);

        // Check for convergence
        double error = parallel_norm(r);
        // std::cout << "PCG Iter: " << k << ", Error: " << error << std::endl;
        if (error < tol) {
            std::cout << "PCG converged in " << k << " iterations.\n";
            return k;
        }

        // z = M_inv * r
        apply_jacobi_preconditioner(z, r, A);

        // rho_new = r . z
        double rho_new = parallel_dot(r, z);

        // beta = rho_new / rho
        double beta = rho_new / rho;

        // p = z + beta * p
        parallel_saxpy(p, 1.0, z, beta, p);

        // rho = rho_new
        rho = rho_new;
    }

    std::cerr << "Warning: PCG did not converge after " 
              << max_iter << " iterations.\n";
    return max_iter;
}

/**
 * @brief Applies the Jacobi preconditioner: z = M_inv * r
 * M is the diagonal of A, so M_inv is just 1.0 / A.A_c
 */
void apply_jacobi_preconditioner(Vector& z, const Vector& r, 
                                 const StencilMatrix& A)
{
    #pragma omp parallel for
    for (size_t i = 0; i < z.size(); ++i) {
        // A.A_c[i] is 1.0 for solid/Dirichlet cells, so no divide-by-zero
        z[i] = r[i] / A.A_c[i];
    }
}

/**
 * @brief (RUN ONCE) Builds the right-hand-side vector 'b' for A*x = b.
 * Starts with the divergence (f) and modifies it for Dirichlet/Solid BCs.
 * * @param b         Output vector (e.g., field.p_rhs)
 * @param divergence Input vector (e.g., field.div_u)
 * @param field     Const reference to the field (for is_solid)
 * @param m         Const reference to the mesh
 */
void build_b_vector(Vector& b, const Vector& divergence, 
                    const Field& field, const Mesh& m)
{
    const int nx = m.nx, ny = m.ny, nz = m.nz;

    #pragma omp parallel for
    for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
    for (int k = 0; k < nz; ++k) {
        
        const int id = field.idx(i, j, k);

        if (field.is_solid[id]) {
            // Solid cells are set to p=0
            b[id] = 0.0;
        }
        else if (i == nx - 1) {
            // Outlet (Dirichlet p=0)
            b[id] = 0.0;
        }
        else {
            // This is a fluid cell
            b[id] = -divergence[id];
            
            // If neighbor to outlet, modify 'b'
            // (p_outlet = 0.0, so this subtraction is 0,
            // but this is the general form)
            // if (i == nx - 2) {
            //     const double A_e_val = 1.0 / (m.dx * m.dx);
            //     b[id] -= A_e_val * 0.0; // b = f - A*x_known
            // }
        }
    }}}
}

/**
 * @brief (RUN ONCE) Pre-computes the 7 stencil coefficients for A.
 * This function "bakes" all BCs into the matrix.
 * * @param A       Output StencilMatrix to be filled
 * @param field   Const reference to the field (for is_solid)
 * @param m       Const reference to the mesh
 */
void build_stencil(StencilMatrix& A, const Field& field, const Mesh& m)
{
    const int nx = m.nx, ny = m.ny, nz = m.nz;

    // --- 1. Base (interior) stencil coefficients ---
    const double idx2 = 1.0 / (m.dx * m.dx);
    const double idy2 = 1.0 / (m.dy * m.dy);
    const double idz2 = 1.0 / (m.dz * m.dz);

    const double A_e_base = -idx2;
    const double A_w_base = -idx2;
    const double A_n_base = -idy2;
    const double A_s_base = -idy2;
    const double A_t_base = -idz2;
    const double A_b_base = -idz2;
    const double A_c_base = +(A_e_base + A_w_base + A_n_base + A_s_base + A_t_base + A_b_base);

    // --- 2. Loop over all cells and set BCs ---
    #pragma omp parallel for
    for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
    for (int k = 0; k < nz; ++k) {
        
        const int id = field.idx(i, j, k);

        // --- Case 1: Solid Cell (Treated as p=0) ---
        if (field.is_solid[id]) {
            A.A_c[id] = 1.0;
            // All neighbor coeffs (A_n, A_s, etc.) remain 0.0
        }
        // --- Case 2: Outlet Cell (Dirichlet p=0) ---
        else if (i == nx - 1) {
            A.A_c[id] = 1.0;
            // All neighbor coeffs (A_n, A_s, etc.) remain 0.0
        }
        // --- Case 3: Fluid Cell ---
        else {
            // Start with interior stencil
            A.A_c[id] = A_c_base;
            A.A_e[id] = A_e_base;
            A.A_w[id] = A_w_base;
            A.A_n[id] = A_n_base;
            A.A_s[id] = A_s_base;
            A.A_t[id] = A_t_base;
            A.A_b[id] = A_b_base;

            // --- Apply Neumann/Solid BCs by modifying the stencil ---
            // We use 2nd-order Neumann: p_ghost = p_neighbor
            // e.g., at i=0, p_west (i-1) = p_east (i+1)
            // Stencil: ... + A_w*p_west + A_e*p_east ...
            // Becomes: ... + A_w*p_east + A_e*p_east ...
            // Becomes: ... + (A_w + A_e)*p_east ...
            // So: A_e_new = A_e_base + A_w_base, and A_w_new = 0.
            
            // X-dir
            if (i == 0) { // Inlet (Neumann)
                A.A_e[id] += A_w_base; A.A_w[id] = 0.0;
            } else if (field.is_solid[field.idx(i-1, j, k)]) { // West Solid
                A.A_e[id] += A_w_base; A.A_w[id] = 0.0;
            }
            if (i == nx - 2) { // Neighbor to Outlet (Dirichlet)
                A.A_e[id] = 0.0;
            } else if (field.is_solid[field.idx(i+1, j, k)]) { // East Solid
                A.A_w[id] += A_e_base; A.A_e[id] = 0.0;
            }
            
            // Y-dir
            if (j == 0) { // Bottom Wall (Neumann)
                A.A_n[id] += A_s_base; A.A_s[id] = 0.0;
            } else if (field.is_solid[field.idx(i, j-1, k)]) { // South Solid
                A.A_n[id] += A_s_base; A.A_s[id] = 0.0;
            }
            if (j == ny - 1) { // Top Wall (Neumann)
                A.A_s[id] += A_n_base; A.A_n[id] = 0.0;
            } else if (field.is_solid[field.idx(i, j+1, k)]) { // North Solid
                A.A_s[id] += A_n_base; A.A_n[id] = 0.0;
            }

            // Z-dir
            if (k == 0) { // Front Wall (Neumann)
                A.A_t[id] += A_b_base; A.A_b[id] = 0.0;
            } else if (field.is_solid[field.idx(i, j, k-1)]) { // Bottom Solid
                A.A_t[id] += A_b_base; A.A_b[id] = 0.0;
            }
            if (k == nz - 1) { // Back Wall (Neumann)
                A.A_b[id] += A_t_base; A.A_t[id] = 0.0;
            } else if (field.is_solid[field.idx(i, j, k+1)]) { // Top Solid
                A.A_b[id] += A_t_base; A.A_t[id] = 0.0;
            }
        }
    }}}
}

/**
 * @brief (RUN IN PCG LOOP) Computes q = A*p.
 * This is the fast, parallel matrix-vector product.
 * * @param q       Output vector
 * @param p       Input vector (current PCG guess)
 * @param A       Const reference to the pre-computed stencil
 * @param field   Const reference to the field (for idx() and dimensions)
 */
void fast_apply_laplacian(Vector& q, const Vector& p, 
                          const StencilMatrix& A, const Field& field)
{
    const int nx = field.nx, ny = field.ny, nz = field.nz;
    const int stride_i = ny * nz;
    const int stride_j = nz;

    #pragma omp parallel for
    for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
    for (int k = 0; k < nz; ++k) {
        
        const int id = field.idx(i, j, k);

        // Start with center value
        // The stencil A is pre-built so that this line
        // is 100% correct for solid and Dirichlet cells.
        double val = A.A_c[id] * p[id];

        // Add neighbor values
        // These 'if' checks prevent reading out-of-bounds memory.
        // The stencil A is pre-built so that the *coefficient*
        // is 0.0 for boundary-adjacent cells (e.g., A.A_w[id] == 0 for i=0)
        // This means we *could* skip the 'if', but that risks
        // a segfault. This 'if' is safe and predictable for the CPU.
        
        if (i > 0)    val += A.A_w[id] * p[id - stride_i];
        if (i < nx-1) val += A.A_e[id] * p[id + stride_i];
        
        if (j > 0)    val += A.A_s[id] * p[id - stride_j];
        if (j < ny-1) val += A.A_n[id] * p[id + stride_j];
        
        if (k > 0)    val += A.A_b[id] * p[id - 1];
        if (k < nz-1) val += A.A_t[id] * p[id + 1];

        q[id] = val;
    }}}
}

// (Place these in a utility header or at the top of your .cpp file)

/**
 * @brief Computes the parallel dot product: result = v1 . v2
 */
double parallel_dot(const Vector& v1, const Vector& v2)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < v1.size(); ++i) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

/**
 * @brief Computes the L2-norm of a vector: result = ||v||
 */
double parallel_norm(const Vector& v)
{
    return std::sqrt(parallel_dot(v, v));
}

/**
 * @brief Parallel vector update: v_out = v_out + a * v_in
 */
void parallel_axpy(Vector& v_out, double a, const Vector& v_in)
{
    #pragma omp parallel for
    for (size_t i = 0; i < v_out.size(); ++i) {
        v_out[i] += a * v_in[i];
    }
}

/**
 * @brief Parallel scaled update: v_out = a * v_in1 + b * v_in2
 * (Used for p = z + beta*p)
 */
void parallel_saxpy(Vector& v_out, double a, const Vector& v_in1, 
                    double b, const Vector& v_in2)
{
    #pragma omp parallel for
    for (size_t i = 0; i < v_out.size(); ++i) {
        v_out[i] = a * v_in1[i] + b * v_in2[i];
    }
}

/**
 * @brief Parallel vector copy: v_out = v_in
 */
void parallel_copy(Vector& v_out, const Vector& v_in)
{
    #pragma omp parallel for
    for (size_t i = 0; i < v_out.size(); ++i) {
        v_out[i] = v_in[i];
    }
}