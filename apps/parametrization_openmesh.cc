/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <TinyAD/Support/OpenMesh.hh>
#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/Utils/NewtonDirection.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>
#include <TinyAD/Utils/LineSearch.hh>

#include <TinyAD-Examples/Filesystem.hh>
#include <TinyAD-Examples/GlowViewerOpenMesh.hh>
#include <TinyAD-Examples/TutteEmbeddingOpenMesh.hh>

 /**
  * Injectively map a disk-topology triangle mesh to the plane
  * and optimize the symmetric Dirichlet energy via projected Newton.
  * This corresponds to Figure 2 in the paper (+performance evaluation in Figure 7).
  */
template<typename T>
void save_sparse_mat(const Eigen::SparseMatrix<T>& mat, const std::string& filename) {
    std::ofstream file(filename);

    for (int k = 0; k < mat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<T>::InnerIterator it(mat, k); it; ++it) {
            file << it.row() << " " << it.col() << " " << it.value() << "\n";
        }
    }

    file.close();
}

int main()
{
    // Init viewer
    glow::glfw::GlfwContext ctx;
    auto g = gv::grid(); // Create grid. Viewer opens on destructor.

    // Read disk-topology mesh as OpenMesh and compute Tutte embedding
    OpenMesh::TriMesh mesh;

    OpenMesh::IO::read_mesh(mesh, DATA_PATH.string() + "/" + "bunnyhead.obj");
    auto ph_param = tutte_embedding(mesh);

    glow_view_mesh(mesh, true, "Input Mesh"); // Add input mesh to viewer gird
    glow_view_param(mesh, ph_param, "Initial Parametrization"); // Add initial param to viewer grid

    // Pre-compute rest shapes of triangles in 2D local coordinate systems
    OpenMesh::FPropHandleT<Eigen::Matrix2d> ph_rest_shapes;
    mesh.add_property(ph_rest_shapes);
    for (auto t : mesh.faces())
    {
        // Get 3D vertex positions
        Eigen::Vector3d ar_3d = mesh.point(t.halfedge().to());
        Eigen::Vector3d br_3d = mesh.point(t.halfedge().next().to());
        Eigen::Vector3d cr_3d = mesh.point(t.halfedge().from());

        // Set up local 2D coordinate system
        Eigen::Vector3d n = (br_3d - ar_3d).cross(cr_3d - ar_3d);
        Eigen::Vector3d b1 = (br_3d - ar_3d).normalized();
        Eigen::Vector3d b2 = n.cross(b1).normalized();

        // Express a, b, c in local 2D coordiante system
-        Eigen::Vector2d ar_2d(0.0, 0.0);
        Eigen::Vector2d br_2d((br_3d - ar_3d).dot(b1), 0.0);
        Eigen::Vector2d cr_2d((cr_3d - ar_3d).dot(b1), (cr_3d - ar_3d).dot(b2));

        // Save 2-by-2 matrix with edge vectors as colums
        //std::cout << "\n==\n";
        //std::cout << TinyAD::col_mat(br_2d - ar_2d, cr_2d - ar_2d);
        mesh.property(ph_rest_shapes, t) = TinyAD::col_mat(br_2d - ar_2d, cr_2d - ar_2d);
    }

    // Set up function with 2D vertex positions as variables.
    auto func = TinyAD::scalar_function<2>(mesh.vertices());

    // Add objective term per triangle. Each connecting 3 vertices.
    func.add_elements<3>(mesh.faces(), [&](auto& element)->TINYAD_SCALAR_TYPE(element)
    {
        // Element is evaluated with either double or TinyAD::Double<6>
        using T = TINYAD_SCALAR_TYPE(element);

        // Get variable 2D vertex positions of triangle t
        OpenMesh::SmartFaceHandle t = element.handle;
        Eigen::Vector2<T> a = element.variables(t.halfedge().to());
        Eigen::Vector2<T> b = element.variables(t.halfedge().next().to());
        Eigen::Vector2<T> c = element.variables(t.halfedge().from());

        // Triangle flipped?
        Eigen::Matrix2<T> M = TinyAD::col_mat(b - a, c - a);
        if (M.determinant() <= 0.0)
            return (T)INFINITY;

        // Get constant 2D rest shape and area of triangle t
        Eigen::Matrix2d Mr = mesh.property(ph_rest_shapes, t);
        double A = 0.5 * Mr.determinant();

        // Compute symmetric Dirichlet energy
        Eigen::Matrix2<T> J = M * Mr.inverse();
        T res = A * (J.squaredNorm() + J.inverse().squaredNorm());

        //std::cout << "F   = " << t.idx() << "\n";
       // std::cout << "fun = " << res.val << "\n";
        //std::cout << "grad = " << res.grad << "\n";

        return res;
    });

    // Assemble inital x vector from parametrization property.
    // x_from_data(...) takes a lambda function that maps
    // each variable handle (OpenMesh::SmartVertexHandle) to its initial 2D value (Eigen::Vector2d).
    Eigen::VectorXd x = func.x_from_data([&](OpenMesh::SmartVertexHandle v) {
        return mesh.property(ph_param, v);
        });

    // Projected Newton
    TinyAD::LinearSolver solver;
    int max_iters = 1000;
    double convergence_eps = 1e-2;
    for (int i = 0; i < max_iters; ++i)
    {
        auto [f, g, H_proj] = func.eval_with_hessian_proj(x);

        //{
        //    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> rowPerm(2 * mesh.n_vertices());
        //    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> colPerm(2 * mesh.n_vertices());
        //
        //    rowPerm.indices() << 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16, 8, 17;
        //    colPerm.indices() << 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16, 8, 17;
        //
        //    Eigen::MatrixXd permutedMatrix = rowPerm * H_proj;
        //    permutedMatrix = permutedMatrix * colPerm.transpose();
        //
        //    double tolerance = 1e-5; // Elements smaller than this will be considered zero.
        //    Eigen::SparseMatrix<double> sparseMatrix = permutedMatrix.sparseView(tolerance);
        //
        //    
        //    std::cout << "\n **** H **** \n";
        //    int width = 20;
        //    for (int i = 0; i < sparseMatrix.rows(); ++i) {
        //        for (int j = 0; j < sparseMatrix.cols(); ++j) {
        //            std::cout << sparseMatrix.coeff(i, j) << std::setw(width);
        //        }
        //        std::cout << "\n";
        //    }
        //
        //    Eigen::VectorXd permutedVector = rowPerm * g;
        //    g = permutedVector;
        //
        //    std::cout << "\n **** g **** \n" << g << "\n";
        //
        //    Eigen::VectorXd d = TinyAD::newton_direction(permutedVector, sparseMatrix, solver);
        //    std::cout << "\n **** d **** \n" << d << "\n";
        //}

        //std::cout << "\n **** g **** \n" << g << "\n";

        //std::cout << "\n **** H **** \n";
        //int width = 20;
        //for (int i = 0; i < H_proj.rows(); ++i) {
        //    for (int j = 0; j < H_proj.cols(); ++j) {
        //        std::cout << H_proj.coeff(i, j)<< std::setw(width);
        //    }
        //    std::cout << "\n";
        //}
        //save_sparse_mat(H_proj, "H_proj");

        TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);

        Eigen::VectorXd d = TinyAD::newton_direction(g, H_proj, solver);

        //std::cout << "\n **** d **** \n" << d << "\n";

        if (TinyAD::newton_decrement(d, g) < convergence_eps)
            break;
        x = TinyAD::line_search(x, d, f, g, func);

        break;
    }
    TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));

    // Write final x vector to parametrization property.
    // x_to_data(...) takes a lambda function that writes the final value
    // of each variable (Eigen::Vector2d) back our parametrization property.
    func.x_to_data(x, [&](OpenMesh::SmartVertexHandle v, const Eigen::Vector2d& _p) {
        mesh.property(ph_param, v) = _p;
        });

    // Add resulting param to viewer grid
    glow_view_param(mesh, ph_param, "Optimized Parametrization");

    return 0;
}
