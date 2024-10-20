
#include <TinyAD/Support/OpenMesh.hh>
#include <TinyAD/Scalar.hh>
#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>
#include <TinyAD/Utils/NewtonDirection.hh>
#include <TinyAD/Utils/LineSearch.hh>
#include <TinyAD/Support/OpenMesh.hh>

#include <TinyAD-Examples/Filesystem.hh>
#include <TinyAD-Examples/GlowViewerOpenMesh.hh>

int main()
{
    glow::glfw::GlfwContext ctx;

    auto g = gv::grid();

    OpenMesh::TriMesh mesh;

    OpenMesh::IO::read_mesh(mesh, DATA_PATH.string() + "/" + "bunnyhead.obj");
    glow_view_mesh(mesh, true, "Input Mesh");


    //3 because we optimize for the vertex position 
    auto func = TinyAD::scalar_function<3>(mesh.vertices());

    //new vertex position (init with old position)
    OpenMesh::VPropHandleT<Eigen::Vector3d> v_pos;
    mesh.add_property(v_pos);

    for (auto& v : mesh.vertices()) {
        mesh.property(v_pos, v) = mesh.point(v);
    }


    //2 because every edge accesses the two end vertices of the edge 
    func.add_elements<2>(mesh.edges(), [&](auto& element)->TINYAD_SCALAR_TYPE(element) {

        using T = TINYAD_SCALAR_TYPE(element);

        OpenMesh::SmartEdgeHandle t = element.handle;
        Eigen::Vector3<T> v0 = element.variables(t.v0());
        //std::cout << "v0=\n" << v0 << "\n";
        Eigen::Vector3<T> v1 = element.variables(t.v1());
        //std::cout << "v1=\n" << v1 << "\n";

        Eigen::Vector3<T> res = (v0 - v1);
        //std::cout << "res=\n" << res << "\n";

        T ret = res.squaredNorm();
        //std::cout << "ret =\n" << ret << "\n";

        return ret;
    });



    Eigen::VectorXd x = func.x_from_data([&](OpenMesh::SmartVertexHandle v) {
        return mesh.property(v_pos, v);
        });

    int max_iter = 100;
    double learning_rate = 0.005;
    for (int i = 0; i < max_iter; ++i) {

        auto [f, g] = func.eval_with_gradient(x);

        std::cout << "f = " << f << "\n";

        //take a step         
        x = x - learning_rate * g;

    }

    func.x_to_data(x, [&](OpenMesh::SmartVertexHandle v, const Eigen::Vector3d& _p) {
        mesh.point(v) = _p;
        });

    glow_view_mesh(mesh, true, "Output Mesh");

    return 0;
}
