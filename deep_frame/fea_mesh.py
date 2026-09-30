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
        gmsh.model.add("frame")
        gmsh.model.occ.importShapes(request["step_path"])
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        if len(volumes) != 1:
            raise ValueError("The STEP input must contain exactly one volume")
        gmsh.model.addPhysicalGroup(3, [volumes[0][1]], name="FRAME")
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
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
        }
        (destination / "mesh_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    generate_mesh(sys.argv[1])
