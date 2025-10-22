#ifndef PRESSURE_H
#define PRESSURE_H

#include "field.h"

#include <iostream>
#include <vector>
#include <cmath>

using Vector = std::vector<double>;

/**
 * @brief Holds the 7-point stencil coefficients for the entire grid.
 * This is the "matrix" A.
 */
struct StencilMatrix
{
    // A_c[idx] = Center coeff for cell (i,j,k)
    // A_n[idx] = North (j+1) coeff for cell (i,j,k)
    // ... etc.
    Vector A_c, A_n, A_s, A_e, A_w, A_t, A_b;

    StencilMatrix(int n_nodes)
        : A_c(n_nodes, 0.0), A_n(n_nodes, 0.0), A_s(n_nodes, 0.0),
          A_e(n_nodes, 0.0), A_w(n_nodes, 0.0), A_t(n_nodes, 0.0),
          A_b(n_nodes, 0.0)
    {}
};

void apply_pressure_BCs(struct Field &f, const struct Mesh &m);
void solve_pressure_jacobi(struct Field &f, const struct Mesh &m, double dt, double rho);
void solve_pressure_GS(struct Field &f, const struct Mesh &m, double dt, double rho, int maxIter);
void solve_pressure_pcg(Field &f, const Mesh &m, StencilMatrix &A, double dt, 
                        double rho, int maxIter=200, double tol=1e-6);

void calculate_divergence(Vector& divergence, const Field& f, const Mesh& m, double dt, double rho);
void build_stencil(StencilMatrix& A, const Field& field, const Mesh& m);
void build_b_vector(Vector& b, const Vector& divergence, 
                    const Field& field, const Mesh& m);
void fast_apply_laplacian(Vector& q, const Vector& p, 
                          const StencilMatrix& A, const Field& field);
int solve_pcg(const StencilMatrix& A, const Vector& b, Vector& x, 
              const Field& field, int maxIter, double tot);

void apply_jacobi_preconditioner(Vector& z, const Vector& r, const StencilMatrix& A);
void parallel_axpy(Vector& v_out, double a, const Vector& v_in);
double parallel_norm(const Vector& v);
double parallel_dot(const Vector& v1, const Vector& v2);
void parallel_saxpy(Vector& v_out, double a, const Vector& v_in1, 
                    double b, const Vector& v_in2);
void parallel_copy(Vector& v_out, const Vector& v_in);

#endif // PRESSURE_H