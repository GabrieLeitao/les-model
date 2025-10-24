#include "io.h"

/**
 * save_state - save current simulation state to checkpoint files.
 * @param field Field const& : current field data to save
 * @param mesh  Mesh const&  : mesh geometry for validation
 * @param time_simulated double : current simulated time
 * @param dt    double      : current time step size
 * @param next_step_to_run int : step number for next run to start from
 * Operations:
 * - Write metadata file (text) with next step, time, dt
 * - Write data file (binary) with field arrays
 */
void save_state(const Field& field, const Mesh& mesh, double time_simulated, double dt, int next_step_to_run)
{
    std::cout << "Saving checkpoint state...\n";

    // 1. Create a metadata file (text)
    std::ofstream meta_file("checkpoint/checkpoint.meta");
    if (!meta_file) {
        std::cerr << "Error: Could not create checkpoint.meta\n";
        return;
    }
    meta_file << next_step_to_run << "\n";
    meta_file << std::setprecision(std::numeric_limits<double>::max_digits10);
    meta_file << time_simulated << "\n";
    meta_file << dt << "\n";
    meta_file.close();

    // 2. Create a data file (binary)
    // We use std::ios::binary
    std::ofstream data_file("checkpoint/checkpoint.data", std::ios::binary);
    if (!data_file) {
        std::cerr << "Error: Could not create checkpoint.data\n";
        return;
    }

    // Get the total number of points
    size_t num_points = static_cast<size_t>(mesh.nx) * mesh.ny * mesh.nz;

    // Write all the field arrays as raw bytes, preserving precision
    data_file.write(reinterpret_cast<const char*>(field.u.data()), num_points * sizeof(double));
    data_file.write(reinterpret_cast<const char*>(field.v.data()), num_points * sizeof(double));
    data_file.write(reinterpret_cast<const char*>(field.w.data()), num_points * sizeof(double));
    data_file.write(reinterpret_cast<const char*>(field.p.data()), num_points * sizeof(double));
    data_file.write(reinterpret_cast<const char*>(field.nu_t.data()), num_points * sizeof(double));

    std::vector<char> solid_data(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        solid_data[i] = field.is_solid[i] ? 1 : 0;
    }
    data_file.write(
        reinterpret_cast<const char*>(solid_data.data()),
        num_points * sizeof(char) // sizeof(char) is always 1
    );

    data_file.close();
    std::cout << "Checkpoint saved. Next run will start from step " << next_step_to_run << ".\n";
}

/**
 * load_state - attempt to load latest output file to continue simulation.
 * @param field Field& : in/out; field to populate with loaded data
 * @param mesh  Mesh const& : mesh geometry for validation
 * @param time_simulated double& : out; updated with time corresponding to loaded step
 * @param dt    double : time step size used to estimate time from step number
 * Returns:
 * - int : step number to continue from, or 0 if no checkpoint found.
 * Operations:
 * - Read metadata file (text) for step number, time, dt
 * - Read data file (binary) for field arrays
 */
int load_state(Field& field, Mesh& mesh, double& time_simulated, double& dt)
{
    // 1. Read metadata file (text)
    std::ifstream meta_file("checkpoint/checkpoint.meta");
    if (!meta_file) {
        return 0; // No file, so we start from scratch
    }
    
    int startStep;
    meta_file >> startStep;
    meta_file >> time_simulated;
    meta_file >> dt;
    meta_file.close();

    // 2. Read data file (binary)
    std::ifstream data_file("checkpoint/checkpoint.data", std::ios::binary);
    if (!data_file) {
        std::cerr << "Error: Found checkpoint.meta but not checkpoint.data!\n";
        return 0; // Failed
    }

    size_t num_points = static_cast<size_t>(mesh.nx) * mesh.ny * mesh.nz;

    
    data_file.read(reinterpret_cast<char*>(field.u.data()), num_points * sizeof(double));
    data_file.read(reinterpret_cast<char*>(field.v.data()), num_points * sizeof(double));
    data_file.read(reinterpret_cast<char*>(field.w.data()), num_points * sizeof(double));
    data_file.read(reinterpret_cast<char*>(field.p.data()), num_points * sizeof(double));
    data_file.read(reinterpret_cast<char*>(field.nu_t.data()), num_points * sizeof(double));
  
    // Read the raw bytes back into the field arrays
    std::vector<char> solid_data(num_points);
    data_file.read(reinterpret_cast<char*>(solid_data.data()), num_points * sizeof(char));
    // 2. Convert back from std::vector<char> to std::vector<bool>
    for (size_t i = 0; i < num_points; ++i) {
        field.is_solid[i] = (solid_data[i] == 1);
    }
   
    data_file.close();

    if (data_file.gcount() == 0) {
        std::cerr << "Error reading checkpoint data. File might be corrupt.\n";
        return 0;
    }

    std::cout << "Successfully loaded checkpoint. Resuming from step " << startStep << ".\n";
    return startStep;
}

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
