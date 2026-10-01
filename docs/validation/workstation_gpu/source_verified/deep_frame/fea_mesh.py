import json
import sys
from pathlib import Path

import gmsh
import numpy as np


def generate_mesh(request_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    settings = request["settings"]
    destination = Path(request["output_dir"])
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", settings.get("mesh_threads", 1))
        gmsh.option.setNumber("Mesh.MeshSizeMax", settings["mesh_size_mm"])
        gmsh.option.setNumber("Mesh.MeshSizeMin", settings.get("mesh_min_size_mm", 0.0))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", settings.get("mesh_curvature_points", 12))
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.RandomSeed", 1)
        second_order_linear = settings.get("mesh_second_order_linear", False)
        high_order_optimize = settings.get("mesh_high_order_optimize", 0)
        if not isinstance(second_order_linear, bool) or high_order_optimize not in (0, 1, 2, 3, 4):
            raise ValueError("Explicit boolean mesh_second_order_linear and optimization mode 0..4 required")
        gmsh.option.setNumber("Mesh.SecondOrderLinear", int(second_order_linear))
        gmsh.model.add("frame")
        gmsh.model.occ.importShapes(request["step_path"])
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        if len(volumes) != 1:
            raise ValueError("The STEP input must contain exactly one volume")
        gmsh.model.addPhysicalGroup(3, [volumes[0][1]], name="FRAME")
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
        if high_order_optimize in (2, 3):
            gmsh.model.mesh.optimize("HighOrderElastic")
        if high_order_optimize in (1, 2):
            gmsh.model.mesh.optimize("HighOrder")
        if high_order_optimize == 4:
            gmsh.model.mesh.optimize("HighOrderFastCurving")
        types, tags, _ = gmsh.model.mesh.getElements(3)
        if list(types) != [11] or not len(tags[0]):
            raise ValueError("The volume mesh must consist of C3D10 tetrahedra")
        quality = gmsh.model.mesh.getElementQualities(tags[0], "minDetJac")
        if not np.all(np.isfinite(quality)) or np.min(quality) <= 0:
            raise ValueError("The mesh contains a non-positive Jacobian")
        gmsh.write(str(destination / "mesh.inp"))
        gmsh.write(str(destination / "mesh.msh"))
        metadata = {
            "gmsh_version": gmsh.__version__,
            "element_type": "C3D10",
            "element_count": len(tags[0]),
            "minimum_jacobian_mm3": float(np.min(quality)),
            "mesh_size_mm": settings["mesh_size_mm"],
            "second_order_linear": second_order_linear,
            "high_order_optimize": high_order_optimize,
            "boundary_geometry": "piecewise planar quadratic tetrahedra with straight midside nodes" if second_order_linear else "quadratic boundary nodes projected to CAD",
        }
        (destination / "mesh_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    generate_mesh(sys.argv[1])
