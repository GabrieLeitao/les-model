#include "io_vtk.h"
#include <fstream>
#include <sstream>
#include <limits>
#include <iostream>
#include <dirent.h>   // POSIX directory access
#include <sys/stat.h>

std::string find_latest_output(const std::string &dir, int &out_step)
{
    DIR *d = opendir(dir.c_str());
    if (!d) return "";

    std::uint64_t max_step = 0;
    bool found = false;
    std::string best_path;

    struct dirent *ent;
    while ((ent = readdir(d)) != nullptr)
    {
        std::string name(ent->d_name);
        // expecting name like "output_<N>.vtk"
        if (name.rfind("output_", 0) != 0) continue;
        auto dot = name.rfind('.');
        if (dot == std::string::npos) continue;
        std::string ext = name.substr(dot);
        if (ext != ".vtk") continue;

        auto pos = name.rfind('_');
        if (pos == std::string::npos || dot <= pos+1) continue;
        std::string numstr = name.substr(pos+1, dot - pos - 1);
        try {
            std::uint64_t step = std::stoull(numstr);
            if (!found || step >= max_step) {
                max_step = step;
                best_path = dir + "/" + name;
                found = true;
            }
        } catch (...) {
            continue;
        }
    }
    closedir(d);
    out_step = max_step;
    return best_path;
}

bool read_field_vtk(const std::string &filename, Field &f, const Mesh &m)
{
    std::ifstream in(filename);
    if (!in.is_open()) return false;

    const size_t npoints = static_cast<size_t>(m.nx) * m.ny * m.nz;
    std::string line;
    bool have_vel = false, have_p = false;
    // Read file line by line and detect blocks
    while (std::getline(in, line))
    {
        // trim leading spaces
        size_t first_non_space = line.find_first_not_of(" \t\r\n");
        std::string header = (first_non_space==std::string::npos) ? "" : line.substr(first_non_space);

        if (!have_vel && header.rfind("VECTORS", 0) == 0)
        {
            // read npoints vectors (skip empty lines)
            size_t read = 0;
            while (read < npoints && std::getline(in, line))
            {
                if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
                std::istringstream iss(line);
                double uu, vv, ww;
                if (!(iss >> uu >> vv >> ww))
                {
                    std::vector<double> vals;
                    std::istringstream iss2(line);
                    double v;
                    while (iss2 >> v) vals.push_back(v);
                    if (vals.size() >= 3) { uu = vals[0]; vv = vals[1]; ww = vals[2]; }
                    else return false;
                }
                f.u[read] = uu;
                f.v[read] = vv;
                f.w[read] = ww;
                ++read;
            }
            if (read != npoints) return false;
            have_vel = true;
        }
        else if (!have_p && header.rfind("SCALARS pressure", 0) == 0)
        {
            // next line is LOOKUP_TABLE default (skip)
            std::getline(in, line);
            size_t read = 0;
            while (read < npoints && std::getline(in, line))
            {
                if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
                std::istringstream iss(line);
                double pp;
                if (!(iss >> pp)) return false;
                f.p[read] = pp;
                ++read;
            }
            if (read != npoints) return false;
            have_p = true;
        }
        // optional: read nu_t and solid if needed
    }

    return have_vel && have_p;
}

void write_field_vtk(const Field &f, const Mesh &m, int step)
{
    std::ofstream file("output/output_" + std::to_string(step) + ".vtk");
    file << "# vtk DataFile Version 3.0\n";
    file << "LES output step " << step << "\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << m.nx << " " << m.ny << " " << m.nz << "\n";
    file << "ORIGIN 0 0 0\n";
    file << "SPACING " << m.dx << " " << m.dy << " " << m.dz << "\n";
    file << "POINT_DATA " << m.nx*m.ny*m.nz << "\n";

    // velocity
    file << "VECTORS velocity double\n";
    for(int i=0;i<m.nx*m.ny*m.nz;++i)
        file << f.u[i] << " " << f.v[i] << " " << f.w[i] << "\n";

    // pressure
    file << "SCALARS pressure double 1\nLOOKUP_TABLE default\n";
    for(int i=0;i<m.nx*m.ny*m.nz;++i)
        file << f.p[i] << "\n";

    // nu_t
    file << "SCALARS nu_t double 1\nLOOKUP_TABLE default\n";
    for(int i=0;i<m.nx*m.ny*m.nz;++i)
        file << f.nu_t[i] << "\n";

    // Obstacle
    file << "SCALARS solid int 1\nLOOKUP_TABLE default\n";
    for(int i=0;i<m.nx*m.ny*m.nz;++i)
        file << (f.is_solid[i] ? 1 : 0) << "\n";
}
