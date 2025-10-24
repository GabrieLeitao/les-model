#ifndef STATISTICS_H
#define STATISTICS_H

#include "field.h"
#include <vector>
#include <string>

/**
 * @struct Statistics
 * @brief Holds all accumulating data for time-averaged statistics.
 *
 * We accumulate sums (e.g., u_sum) and sums of squares (e.g., uu_sum)
 * to calculate the mean (<u>) and Reynolds stress (<u'u'> = <u^2> - <u>^2).
 */
struct Statistics
{
    int num_samples;
    int start_step; // The step number to begin collecting statistics

    // A Field object to hold the sums of <u, v, w, p>
    Field sum_field; 

    // Sums of squares and cross-products for Reynolds stresses
    std::vector<double> uu_sum;
    std::vector<double> vv_sum;
    std::vector<double> ww_sum;
    std::vector<double> uv_sum;
    std::vector<double> uw_sum;
    std::vector<double> vw_sum;
};

/**
 * @brief Allocates memory for statistics and initializes them to zero.
 * @param stats The Statistics object to initialize.
 * @param m The mesh (for getting data size).
 * @param stat_start_step The time step at which to begin collecting data.
 */
void init_statistics(Statistics &stats, const Mesh &m, int stat_start_step);

/**
 * @brief Updates the sums in the statistics object with the current field data.
 * @param stats The Statistics object to update.
 * @param f The current, instantaneous Field data.
 * @param current_step The current simulation step number.
 */
void update_statistics(Statistics &stats, const Field &f, int current_step);

/**
 * @brief Calculates the final averages and saves them to a VTK file.
 * @param stats The Statistics object.
 * @param m The mesh.
 * @param filename The output filename (e.g., "output/statistics_final.vtk").
 */
void finalize_and_save(Statistics &stats, const Mesh &m, const std::string &filename);

#endif // STATISTICS_H
