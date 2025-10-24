// In test_checkpoint.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// Include your headers for Field, Mesh, save_state, try_continue
#include "field.h"
#include "io.h" // Or wherever you put the functions

bool compare_fields(const Field& f1, const Field& f2, const Mesh& m);


int main(void) 
{
    std::cout << "Running checkpoint I/O test...\n";

    Mesh  mesh{ 10, 10, 10, 0.01, 0.01, 0.01 };
    Field field_source(mesh.nx, mesh.ny, mesh.nz);
    
    size_t num_points = static_cast<size_t>(mesh.nx) * mesh.ny * mesh.nz;
    double time_source = 1.23456789;
    double dt_source = 0.00123;
    int step_source = 101; // The step to start *next*

    // 2. FILL SOURCE with known data
    for (size_t i = 0; i < num_points; ++i) {
        field_source.u[i] = i * 0.1 - 5.0;
        field_source.v[i] = i * -0.2 + 10.0;
        field_source.w[i] = i * 0.3;
        field_source.p[i] = i * -0.4;
        field_source.nu_t[i] = i * 0.5;
        field_source.is_solid[i] = (i % 3 == 0);
    }
    std::cout << "Source data created.\n";

    // 3. SAVE
    save_state(field_source, mesh, time_source, dt_source, step_source);
    std::cout << "Data saved to checkpoint files.\n";

    // 4. LOAD
    Field field_dest(mesh.nx, mesh.ny, mesh.nz);
    
    double time_dest = 0.0;
    double dt_dest = 0.0;

    int step_dest = load_state(field_dest, mesh, time_dest, dt_dest);
    std::cout << "Data loaded from checkpoint files.\n";

    // 5. COMPARE
    bool metadata_match = true;
    const double epsilon = std::numeric_limits<double>::epsilon();  
    if (step_source != step_dest) {
        metadata_match = false;
        std::cerr << "Test FAILED: Step mismatch. (Expected " << step_source << ", Got " << step_dest << ")\n";
    }
    if (std::fabs(time_source - time_dest) > epsilon) {
        metadata_match = false;
        // Set precision for the error message so you can see the difference
        std::cerr << std::setprecision(16); 
        std::cerr << "Test FAILED: Time mismatch. (Expected " << time_source << ", Got " << time_dest << ")\n";
    }
    if (std::fabs(dt_source - dt_dest) > epsilon) {
        metadata_match = false;
        std::cerr << std::setprecision(16);
        std::cerr << "Test FAILED: dt mismatch. (Expected " << dt_source << ", Got " << dt_dest << ")\n";
    }

    bool fields_match = compare_fields(field_source, field_dest, mesh);

    if (metadata_match && fields_match) {
        std::cout << "\n*** SUCCESS! Checkpoint test passed. ***\n";
    } else {
        std::cout << "\n*** FAILURE! Checkpoint test failed. ***\n";
    }

    return (metadata_match && fields_match) ? 0 : 1;
}

#include <iostream>
#include <vector>
#include <cmath> // For std::fabs

// Assuming 'Field' and 'Mesh' definitions are included
// ...

/**
 * Compares two field objects for bitwise equality.
 * For binary I/O, the data should be *exactly* the same.
 */
bool compare_fields(const Field& f1, const Field& f2, const Mesh& m)
{
    size_t num_points = static_cast<size_t>(m.nx) * m.ny * m.nz;
    
    // Check sizes first
    if (f1.u.size() != num_points || f2.u.size() != num_points) {
        std::cerr << "Test Error: Vector size mismatch.\n";
        return false;
    }

    for (size_t i = 0; i < num_points; ++i) {
        // We use == because binary I/O should be bit-perfect.
        // If this fails, you can use std::fabs(f1.u[i] - f2.u[i]) < 1e-15
        if (f1.u[i] != f2.u[i]) {
            std::cerr << "Test FAILED: 'u' mismatch at index " << i << "\n";
            return false;
        }
        if (f1.v[i] != f2.v[i]) {
            std::cerr << "Test FAILED: 'v' mismatch at index " << i << "\n";
            return false;
        }
        if (f1.w[i] != f2.w[i]) {
            std::cerr << "Test FAILED: 'w' mismatch at index " << i << "\n";
            return false;
        }
        if (f1.p[i] != f2.p[i]) {
            std::cerr << "Test FAILED: 'p' mismatch at index " << i << "\n";
            return false;
        }
        if (f1.nu_t[i] != f2.nu_t[i]) {
            std::cerr << "Test FAILED: 'nu_t' mismatch at index " << i << "\n";
            return false;
        }
        if (f1.is_solid[i] != f2.is_solid[i]) {
            std::cerr << "Test FAILED: 'is_solid' mismatch at index " << i << "\n";
            return false;
        }
    }
    return true;
}