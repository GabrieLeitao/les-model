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
                    // Find the *first available* fluid neighbor and copy its pressure
                    if      (i < nx - 1 && !f.is_solid[id + stride_i]) { f.p[id] = f.p[id + stride_i]; }
                    else if (i > 0      && !f.is_solid[id - stride_i]) { f.p[id] = f.p[id - stride_i]; }
                    else if (j < ny - 1 && !f.is_solid[id + stride_j]) { f.p[id] = f.p[id + stride_j]; }
                    else if (j > 0      && !f.is_solid[id - stride_j]) { f.p[id] = f.p[id - stride_j]; }
                    else if (k < nz - 1 && !f.is_solid[id + 1])        { f.p[id] = f.p[id + 1];        }
                    else if (k > 0      && !f.is_solid[id - 1])        { f.p[id] = f.p[id - 1];        }
                    // If it's fully enclosed by solid, its pressure doesn't matter
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
void solve_pressure_GS(Field &f, const Mesh &m, double dt, double rho, int maxIter)
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
int solve_pressure_pcg(const StencilMatrix& A, const Vector& b, Vector& x,
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
    // apply_jacobi_preconditioner(p, r, A);
    apply_ic_preconditioner(p, r, A, field);

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
            // std::cout << "PCG converged in " << k << " iterations.\n";
            return k;
        }

        // z = M_inv * r
        // apply_jacobi_preconditioner(z, r, A);
        apply_ic_preconditioner(z, r, A, field);

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
 * @brief (RUN IN PCG LOOP) Applies the IC(0) preconditioner.
 *
 * Solves M*z = r, where M = L*L^T. This is done in two serial steps:
 * 1. Forward-substitution:  Solves L*y = r  (result stored in z)
 * 2. Back-substitution:   Solves L^T*z = y (result stored in z)
 *
 * @param z       Output: The preconditioned vector.
 * @param r       Input: The residual vector.
 * @param L       Input: The pre-computed L matrix.
 * @param field   Input: For grid dimensions and indexing.
 */
void apply_ic_preconditioner(Vector& z, const Vector& r, const StencilMatrix& L, const Field& field)
{
    const int nx = field.nx, ny = field.ny, nz = field.nz;
    const int stride_i = ny * nz;
    const int stride_j = nz;

    // --- 1. Forward-substitution: L*y = r (storing y in z) ---
    // Solves y[id] = (r[id] - L_w*y_w - L_s*y_s - L_b*y_b) / L_c
    // This loop MUST be serial and in increasing order.
    for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
    for (int k = 0; k < nz; ++k) {
        
        int id = field.idx(i, j, k);
        double sum = r[id];

        if (i > 0) sum -= L.A_w[id] * z[id - stride_i]; // z[w_idx] is y_w
        if (j > 0) sum -= L.A_s[id] * z[id - stride_j]; // z[s_idx] is y_s
        if (k > 0) sum -= L.A_b[id] * z[id - 1];       // z[b_idx] is y_b

        z[id] = sum / L.A_c[id];
    }}}
    
    // --- 2. Back-substitution: L^T*z = y (y is already in z) ---
    // Solves z[id] = (y[id] - L_e*z_e - L_n*z_n - L_t*z_t) / L_c
    // (where L_e = L.A_w of the eastern neighbor, etc.)
    // This loop MUST be serial and in decreasing order.
    for (int i = nx - 1; i >= 0; --i) {
    for (int j = ny - 1; j >= 0; --j) {
    for (int k = nz - 1; k >= 0; --k) {

        int id = field.idx(i, j, k);
        double sum = z[id]; // z[id] is y_i from the forward pass

        if (i < nx-1) sum -= L.A_w[id + stride_i] * z[id + stride_i]; // L_w[east] * z_east
        if (j < ny-1) sum -= L.A_s[id + stride_j] * z[id + stride_j]; // L_s[north] * z_north
        if (k < nz-1) sum -= L.A_b[id + 1]       * z[id + 1];       // L_b[top] * z_top

        z[id] = sum / L.A_c[id];
    }}}
}

/**
 * @brief (RUN ONCE) Computes the IC(0) factorization of matrix A.
 *
 * Calculates a sparse lower-triangular matrix L such that A ≈ L*L^T.
 * The stencil of L has 4 diagonals: L_c, L_w, L_s, L_b.
 * This function MUST be run serially.
 *
 * @param L       Output: The StencilMatrix to store L.
 * @param A       Input: The positive-definite StencilMatrix A.
 * @param field   Input: For grid dimensions and indexing.
 */
void build_ic_preconditioner(StencilMatrix& L, const StencilMatrix& A,
                             const Field& field, const Mesh& m)
{
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    const int stride_i = ny * nz;
    const int stride_j = nz;

    const double small_val = 1e-30; // To prevent divide-by-zero

    // Loop from (0,0,0) to (nx-1, ny-1, nz-1).
    // This MUST be a serial loop. Do not add OpenMP.
    for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
    for (int k = 0; k < nz; ++k) {

        int id = field.idx(i, j, k);
        double sum_sq = 0.0;

        // --- 1. Calculate L_w, L_s, L_b (the off-diagonals) ---
        
        // West neighbor (L_w)
        if (i > 0) {
            int w_idx = id - stride_i;
            double Lc_neighbor = L.A_c[w_idx];
            if (std::abs(Lc_neighbor) > small_val) {
                L.A_w[id] = A.A_w[id] / Lc_neighbor;
                sum_sq += L.A_w[id] * L.A_w[id];
            }
        }
        
        // South neighbor (L_s)
        if (j > 0) {
            int s_idx = id - stride_j;
            double Lc_neighbor = L.A_c[s_idx];
            if (std::abs(Lc_neighbor) > small_val) {
                L.A_s[id] = A.A_s[id] / Lc_neighbor;
                sum_sq += L.A_s[id] * L.A_s[id];
            }
        }
        
        // Bottom neighbor (L_b)
        if (k > 0) {
            int b_idx = id - 1;
            double Lc_neighbor = L.A_c[b_idx];
            if (std::abs(Lc_neighbor) > small_val) {
                L.A_b[id] = A.A_b[id] / Lc_neighbor;
                sum_sq += L.A_b[id] * L.A_b[id];
            }
        }

        // --- 2. Calculate L_c (the diagonal) ---
        // L_c[id] = sqrt( A_c[id] - (L_w^2 + L_s^2 + L_b^2) )
        double diag_val = A.A_c[id] - sum_sq;

        // Robustness check: Ensure diagonal is positive
        if (diag_val <= small_val) {
            L.A_c[id] = 1.0; // Failsafe to prevent NAN/INF
        } else {
            L.A_c[id] = std::sqrt(diag_val);
        }
    }}}
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
    const double A_c_base = (2.0*idx2 + 2.0*idy2 + 2.0*idz2);

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

/**
 * @brief Solves the linear system A*x = b using PCG with Chebyshev polynomial preconditioner.
 *
 * @param A           The pre-built StencilMatrix
 * @param b           The right-hand-side vector (from build_b_vector)
 * @param x           Input: Initial guess (field.p). Output: The solution.
 * @param field       Const reference to the field (for fast_apply_laplacian)
 * @param lambda_min  Estimated smallest eigenvalue of A
 * @param lambda_max  Estimated largest eigenvalue of A
 * @param poly_degree Degree of the Chebyshev polynomial
 * @param max_iter    Max iterations
 * @param tol         Convergence tolerance
 * @return int        Number of iterations performed
 */
int solve_pressure_pcg_chebyshev(const StencilMatrix& A, const Vector& b, Vector& x,
              const Field& field, double lambda_min, double lambda_max,
               int poly_degree, int max_iter, double tol)
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

    // z = M_inv * r (Chebyshev preconditioner, where M is diag(A))
    apply_chebyshev_preconditioner(p, r, A, field, lambda_min, lambda_max, poly_degree); // NEW


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
            // std::cout << "PCG converged in " << k << " iterations.\n";
            return k;
        }

        // z = M_inv * r
        apply_chebyshev_preconditioner(z, r, A, field, lambda_min, lambda_max, poly_degree); // NEW


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
 * @brief (RUN IN PCG LOOP) Applies a Chebyshev polynomial preconditioner.
 *
 * Solves M*z = r, where M is an approximation of A.
 * This is fully parallel.
 *
 * @param z           Output: The preconditioned vector.
 * @param r           Input: The residual vector.
 * @param A           Input: The StencilMatrix A.
 * @param field       Input: For grid dimensions and indexing.
 * @param lambda_min  Input: Estimated smallest eigenvalue.
 * @param lambda_max  Input: Estimated largest eigenvalue.
 * @param poly_degree Input: Degree of the polynomial (e.g., 3, 5, 7).
 */
void apply_chebyshev_preconditioner(Vector& z, const Vector& r,
                                    const StencilMatrix& A, const Field& field,
                                    double lambda_min, double lambda_max,
                                    int poly_degree)
{
    const int n_nodes = field.nx * field.ny * field.nz;
    
    // Allocate temporary vectors
    Vector p(n_nodes);
    Vector res(n_nodes);

    // --- 1. Calculate Chebyshev constants ---
    const double d = (lambda_max + lambda_min) / 2.0;
    const double c = (lambda_max - lambda_min) / 2.0;
    
    double alpha = 0.0;
    double beta = 0.0;

    // --- 2. Run Chebyshev Iteration ---
    
    // z_0 = 0 (initial guess for this preconditioner step)
    #pragma omp parallel for
    for (int i = 0; i < n_nodes; ++i) z[i] = 0.0;

    // res_0 = r - A*z_0 = r
    parallel_copy(res, r);
    
    // alpha_0 = 2 / d
    alpha = 2.0 / d;
    
    // p_0 = res_0
    parallel_copy(p, res);
    
    // z_1 = z_0 + alpha*p_0
    parallel_axpy(z, alpha, p);
    
    // Loop for k=1 to m-1 (where m = poly_degree)
    for (int k = 1; k < poly_degree; ++k)
    {
        // res_k = r - A*z_k
        fast_apply_laplacian(res, z, A, field); // res = A*z_k
        parallel_saxpy(res, 1.0, r, -1.0, res); // res = r - res
        
        // beta_k = (c * alpha_k-1 / 2.0)^2
        beta = (c * alpha / 2.0) * (c * alpha / 2.0);
        
        // alpha_k = 1.0 / (d - beta_k)
        alpha = 1.0 / (d - beta);
        
        // p_k = res_k + beta_k * p_k-1
        parallel_saxpy(p, 1.0, res, beta, p);
        
        // z_k+1 = z_k + alpha_k * p_k
        parallel_axpy(z, alpha, p);
    }
}

/**
 * @brief (RUN ONCE) Estimates the largest eigenvalue (lambda_max)
 * of the matrix A using the parallel Power Iteration method.
 *
 * @param A       Input: The positive-definite StencilMatrix A.
 * @param field   Input: For grid dimensions and indexing.
 * @param n_iter  Input: Number of iterations (10-20 is usually enough).
 * @return double The estimated largest eigenvalue.
 */
double estimate_lambda_max(const StencilMatrix& A, const Field& field, int n_iter)
{
    const int n_nodes = field.nx * field.ny * field.nz;
    
    // Create temporary vectors
    Vector v(n_nodes);
    Vector w(n_nodes);

    // 1. Create a random starting vector
    std::mt19937 gen(1337); // Mersenne Twister engine
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    #pragma omp parallel for
    for (int i = 0; i < n_nodes; ++i) {
        v[i] = dis(gen);
    }
    parallel_copy(w, v); // Just to have w initialized
    
    double lambda_max = 0.0;
    
    // 2. Run Power Iteration
    for (int k = 0; k < n_iter; ++k)
    {
        // w = A * v (The most expensive, parallel part)
        fast_apply_laplacian(w, v, A, field);
        
        // lambda = v . w
        lambda_max = parallel_dot(v, w);
        
        // v = w / ||w||
        double norm = parallel_norm(w);
        parallel_copy(v, w); // v = w
        #pragma omp parallel for
        for (int i = 0; i < n_nodes; ++i) {
            v[i] /= norm;
        }
    }
    
    return lambda_max;
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