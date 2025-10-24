#include "statistics.h"
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>    // For OpenMP parallelization
#include <iomanip>  // For std::setprecision

/**
 * @brief Helper function to write the final statistics to a VTK file.
 */
void write_statistics_vtk(
    const Field &mean_field,
    const std::vector<double> &uu, const std::vector<double> &vv, const std::vector<double> &ww,
    const std::vector<double> &uv, const std::vector<double> &uw, const std::vector<double> &vw,
    const Mesh &m,
    const std::string &filename)
{
    std::cout << "Writing final statistics to " << filename << "...\n";
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open statistics file: " << filename << "\n";
        return;
    }

    file << "# vtk DataFile Version 3.0\n";
    file << "Time-averaged statistics\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << m.nx << " " << m.ny << " " << m.nz << "\n";
    file << "ORIGIN 0 0 0\n";
    file << "SPACING " << m.dx << " " << m.dy << " " << m.dz << "\n";
    file << "POINT_DATA " << m.nx*m.ny*m.nz << "\n";
    file << std::setprecision(10); // Use good precision

    // 1. Mean Velocity
    file << "VECTORS mean_velocity double\n";
    for(int i=0; i<m.nx*m.ny*m.nz; ++i) {
        file << mean_field.u[i] << " " << mean_field.v[i] << " " << mean_field.w[i] << "\n";
    }

    // 2. Mean Pressure
    file << "SCALARS mean_pressure double 1\nLOOKUP_TABLE default\n";
    for(int i=0; i<m.nx*m.ny*m.nz; ++i) {
        file << mean_field.p[i] << "\n";
    }

    // 3. Reynolds Stresses (as 6 separate scalar fields)
    // ParaView can combine these later if needed.
    file << "SCALARS uu_stress double 1\nLOOKUP_TABLE default\n";
    for(const auto& val : uu) { file << val << "\n"; }

    file << "SCALARS vv_stress double 1\nLOOKUP_TABLE default\n";
    for(const auto& val : vv) { file << val << "\n"; }

    file << "SCALARS ww_stress double 1\nLOOKUP_TABLE default\n";
    for(const auto& val : ww) { file << val << "\n"; }

    file << "SCALARS uv_stress double 1\nLOOKUP_TABLE default\n";
    for(const auto& val : uv) { file << val << "\n"; }

    file << "SCALARS uw_stress double 1\nLOOKUP_TABLE default\n";
    for(const auto& val : uw) { file << val << "\n"; }

    file << "SCALARS vw_stress double 1\nLOOKUP_TABLE default\n";
    for(const auto& val : vw) { file << val << "\n"; }

    file.close();
    std::cout << "Statistics file written.\n";
}


void init_statistics(Statistics &stats, const Mesh &m, int stat_start_step)
{
    std::cout << "Initializing statistics module. Collection will start at step " 
              << stat_start_step << ".\n";

    stats.start_step = stat_start_step;
    stats.num_samples = 0;

    size_t num_points = static_cast<size_t>(m.nx) * m.ny * m.nz;

    // Allocate and zero-out all vectors
    stats.sum_field.u.assign(num_points, 0.0);
    stats.sum_field.v.assign(num_points, 0.0);
    stats.sum_field.w.assign(num_points, 0.0);
    stats.sum_field.p.assign(num_points, 0.0);
    stats.sum_field.nu_t.assign(num_points, 0.0); // We can average nu_t too
    stats.sum_field.is_solid.assign(num_points, false); // Not used, but part of Field

    stats.uu_sum.assign(num_points, 0.0);
    stats.vv_sum.assign(num_points, 0.0);
    stats.ww_sum.assign(num_points, 0.0);
    stats.uv_sum.assign(num_points, 0.0);
    stats.uw_sum.assign(num_points, 0.0);
    stats.vw_sum.assign(num_points, 0.0);
}

void update_statistics(Statistics &stats, const Field &f, int current_step)
{
    // Only collect statistics after the specified start step
    if (current_step <= stats.start_step) {
        return;
    }

    stats.num_samples++;
    const size_t num_points = stats.sum_field.u.size();

    #pragma omp parallel for
    for (size_t i = 0; i < num_points; ++i)
    {
        // Get instantaneous values
        const double u = f.u[i];
        const double v = f.v[i];
        const double w = f.w[i];
        const double p = f.p[i];

        // Add to sums
        stats.sum_field.u[i] += u;
        stats.sum_field.v[i] += v;
        stats.sum_field.w[i] += w;
        stats.sum_field.p[i] += p;

        // Add to sums of squares/cross-products
        stats.uu_sum[i] += u * u;
        stats.vv_sum[i] += v * v;
        stats.ww_sum[i] += w * w;
        stats.uv_sum[i] += u * v;
        stats.uw_sum[i] += u * w;
        stats.vw_sum[i] += v * w;
    }
}

void finalize_and_save(Statistics &stats, const Mesh &m, const std::string &filename)
{
    if (stats.num_samples == 0) {
        std::cout << "No statistics samples collected. Skipping final save.\n";
        return;
    }

    std::cout << "Finalizing statistics from " << stats.num_samples << " samples...\n";

    const size_t num_points = stats.sum_field.u.size();
    const double n_inv = 1.0 / static_cast<double>(stats.num_samples);

    // Create a new Field to store the final mean values
    Field mean_field;
    mean_field.u.resize(num_points);
    mean_field.v.resize(num_points);
    mean_field.w.resize(num_points);
    mean_field.p.resize(num_points);
    // (We don't need to resize nu_t or is_solid unless we also save them)

    // Create vectors to store the final Reynolds stresses
    std::vector<double> uu(num_points);
    std::vector<double> vv(num_points);
    std::vector<double> ww(num_points);
    std::vector<double> uv(num_points);
    std::vector<double> uw(num_points);
    std::vector<double> vw(num_points);

    #pragma omp parallel for
    for (size_t i = 0; i < num_points; ++i)
    {
        // 1. Calculate Mean values
        double u_m = stats.sum_field.u[i] * n_inv;
        double v_m = stats.sum_field.v[i] * n_inv;
        double w_m = stats.sum_field.w[i] * n_inv;
        double p_m = stats.sum_field.p[i] * n_inv;

        mean_field.u[i] = u_m;
        mean_field.v[i] = v_m;
        mean_field.w[i] = w_m;
        mean_field.p[i] = p_m;

        // 2. Calculate Reynolds Stresses: <u'u'> = <u^2> - <u>^2
        uu[i] = (stats.uu_sum[i] * n_inv) - (u_m * u_m);
        vv[i] = (stats.vv_sum[i] * n_inv) - (v_m * v_m);
        ww[i] = (stats.ww_sum[i] * n_inv) - (w_m * w_m);
        uv[i] = (stats.uv_sum[i] * n_inv) - (u_m * v_m);
        uw[i] = (stats.uw_sum[i] * n_inv) - (u_m * w_m);
        vw[i] = (stats.vw_sum[i] * n_inv) - (v_m * w_m);
    }

    // 3. Write all calculated data to one VTK file
    write_statistics_vtk(mean_field, uu, vv, ww, uv, uw, vw, m, filename);
}
