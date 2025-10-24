#ifndef LES_MODEL_FIELD_H
#define LES_MODEL_FIELD_H

#include <vector>

struct Mesh { int nx, ny, nz; double dx, dy, dz; };

// 3D field as 1D vector
struct Field
{
    int nx, ny, nz;

    std::vector<double> u, v, w;       // velocity components
    std::vector<double> u_new, v_new, w_new; // provisional velocities

    std::vector<double> p;             // pressure
    std::vector<double> p_rhs;             // storde RHS: divergence u * rho/dt


    std::vector<double> nu_t;          // <-- eddy viscosity (LES)
    
    std::vector<bool> is_solid;          // Obstacle mask

    Field() : nx(0), ny(0), nz(0) {}
    
    Field(int nx_, int ny_, int nz_)
        : nx(nx_), ny(ny_), nz(nz_),
            u(nx_ * ny_ * nz_, 0.0),
            v(nx_ * ny_ * nz_, 0.0),
            w(nx_ * ny_ * nz_, 0.0),
            u_new(nx_ * ny_ * nz_, 0.0),
            v_new(nx_ * ny_ * nz_, 0.0),
            w_new(nx_ * ny_ * nz_, 0.0),
            p(nx_ * ny_ * nz_, 0.0),
            p_rhs(nx_ * ny_ * nz_, 0.0),
            nu_t(nx_ * ny_ * nz_, 0.0),
            is_solid(nx_ * ny_ * nz_, false)
    {}

    int idx(int i, int j, int k) const { return i * ny * nz + j * nz + k; }
};

struct Stress {
    std::vector<double> tau_xx, tau_xy, tau_xz;
    std::vector<double> tau_yy, tau_yz, tau_zz;
    Stress(int n) : tau_xx(n, 0.0), tau_xy(n, 0.0), tau_xz(n, 0.0),
                    tau_yy(n, 0.0), tau_yz(n, 0.0), tau_zz(n, 0.0) {}
};

#endif // LES_MODEL_FIELD_H
