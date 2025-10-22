#ifndef LES_MODEL_IO_VTK_H
#define LES_MODEL_IO_VTK_H

#include <string>
#include "field.h"

std::string find_latest_output(const std::string &dir, int &out_step);
bool read_field_vtk(const std::string &filename, Field &f, const Mesh &m);
void write_field_vtk(const Field &f, const Mesh &m, int step);

#endif // LES_MODEL_IO_VTK_H
