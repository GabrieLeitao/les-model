#ifndef LES_MODEL_IO_VTK_H
#define LES_MODEL_IO_VTK_H

#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <iostream>
#include <dirent.h>   // POSIX directory access
#include <sys/stat.h>

#include <iomanip> // For std::setprecision

#include "field.h"

void save_state(const Field& field, const Mesh& mesh, double time_simulated, double dt, int next_step_to_run);
int load_state(Field& field, Mesh& mesh, double& time_simulated, double& dt);

std::string find_latest_output(const std::string &dir, int &out_step);
bool read_field_vtk(const std::string &filename, Field &f, const Mesh &m);
void write_field_vtk(const Field &f, const Mesh &m, int step);

#endif // LES_MODEL_IO_VTK_H
