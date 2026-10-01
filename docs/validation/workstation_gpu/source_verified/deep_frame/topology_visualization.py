import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from build123d import Compound, import_step


def _plot_modules():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    return plt, Poly3DCollection


def _preserve_shape(domain):
    from deep_frame.topology_geometry import region_shape

    shapes = [region_shape(region) for region in domain["regions"] if region["role"] == "preserve"]
    holes = [region_shape(region) for region in domain["regions"] if region["role"] == "forbidden"]
    shape = Compound(children=shapes)
    return shape.cut(*holes) if holes else shape


def _mesh_axes(figure, slot, shape, color, title, grid, top=False):
    _, collection = _plot_modules()
    axis = figure.add_subplot(2, 3, slot, projection="3d")
    vertices, faces = shape.tessellate(0.1, 0.2)
    vertices = np.asarray([tuple(point) for point in vertices])
    triangles = vertices[np.asarray(faces)]
    mesh = collection(triangles, facecolors=color, linewidths=0, shade=True)
    axis.add_collection3d(mesh)
    lower = np.asarray(grid["origin_mm"])
    size = np.asarray(grid["spacing_mm"]) * np.asarray(grid["shape"])
    upper = lower + size
    axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]), xlabel="x / mm", ylabel="y / mm", zlabel="z / mm", title=title)
    axis.set_box_aspect(size)
    axis.set_proj_type("ortho")
    axis.view_init(elev=90 if top else 28, azim=-90 if top else -45)
    axis.grid(False)
    if top:
        axis.set_zticks([])
        axis.set_zlabel("")
    return axis


def render_topology_evidence(domain, density, reference_solid, candidate_solid, history, output_dir, label=""):
    plt, _ = _plot_modules()
    field = np.asarray(density, dtype=float)
    if field.shape != tuple(domain["grid"]["shape"]) or not np.all(np.isfinite(field)):
        raise ValueError("Density field must be finite and match the domain grid")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    preserve = _preserve_shape(domain)
    figure = plt.figure(figsize=(16, 10), layout="constrained")
    models = [
        (preserve, "#d98236", "Fixed interfaces only"),
        (reference_solid, "#7994ab", "Parametric v0 reference"),
        (candidate_solid, "#248c87", "Automatically reconstructed free topology"),
    ]
    for index, (solid, color, title) in enumerate(models):
        _mesh_axes(figure, index + 1, solid, color, title, domain["grid"])
        _mesh_axes(figure, index + 4, solid, color, "Top view", domain["grid"], top=True)
    fraction = domain["metadata"]["preserve_fraction_of_allowed"]
    prefix = label + " | " if label else ""
    figure.suptitle(f"{prefix}Deep_Frame Phase 1 | fixed cells {100 * fraction:.2f}% of allowed space", fontsize=15)
    image_path = directory / "geometry_comparison.png"
    figure.savefig(image_path, dpi=160)
    plt.close(figure)
    result["geometry_comparison"] = str(image_path.resolve())

    grid = domain["grid"]
    origin, spacing, shape = np.asarray(grid["origin_mm"]), np.asarray(grid["spacing_mm"]), np.asarray(grid["shape"])
    layers = np.unique(np.linspace(0, shape[2] - 1, min(4, shape[2]), dtype=int))
    figure, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4.5), layout="constrained", squeeze=False)
    extent = [origin[0], origin[0] + spacing[0] * shape[0], origin[1], origin[1] + spacing[1] * shape[1]]
    for axis, layer in zip(axes[0], layers):
        values = np.ma.masked_array(field[:, :, layer].T, mask=domain["forbidden"][:, :, layer].T)
        image = axis.imshow(values, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        preserve_points = np.argwhere(domain["preserve"][:, :, layer])
        if len(preserve_points):
            axis.scatter(origin[0] + spacing[0] * (preserve_points[:, 0] + 0.5), origin[1] + spacing[1] * (preserve_points[:, 1] + 0.5), marker="s", facecolors="none", edgecolors="#f28c28", linewidths=0.8, s=32, label="fixed cells")
        axis.set(title=f"z = {origin[2] + spacing[2] * (layer + 0.5):g} mm", xlabel="x / mm", ylabel="y / mm", aspect="equal")
    figure.colorbar(image, ax=axes[0].tolist(), shrink=0.7, label="Physical material density (0 to 1)")
    figure.suptitle(prefix + "Density field | white: forbidden | orange outline: prescribed interface", fontsize=13)
    image_path = directory / "density_slices.png"
    figure.savefig(image_path, dpi=160)
    plt.close(figure)
    result["density_slices"] = str(image_path.resolve())

    if history:
        iterations = [row["iteration"] for row in history]
        objective = [row["objective"] for row in history]
        volumes = [row["physical_density_sum"] * float(np.prod(spacing)) for row in history]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        axes[0].plot(iterations, objective, color="#248c87", marker=".")
        axes[0].set(xlabel="Evaluation", ylabel="Normalized weighted compliance", title="SIMP objective history")
        axes[1].plot(iterations, volumes, color="#406984", marker=".")
        axes[1].set(xlabel="Evaluation", ylabel="Density volume / mm3", title="Material budget")
        for axis in axes:
            axis.grid(alpha=0.2)
        if label:
            figure.suptitle(label)
        image_path = directory / "optimization_history.png"
        figure.savefig(image_path, dpi=160)
        plt.close(figure)
        result["optimization_history"] = str(image_path.resolve())
    result["rendering"] = "Matplotlib orthographic CAD triangle projections; identical limits and scale for all geometry panels; no synthetic or hand-edited geometry"
    result["label"] = label
    result["matplotlib_version"] = version("matplotlib")
    result["density_sha256"] = hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest()
    result["cad_volumes_mm3"] = {"reference": float(reference_solid.volume), "candidate": float(candidate_solid.volume)}
    result["grid"] = domain["grid"]
    (directory / "visualization_manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def show_topology_comparison(reference_solid, candidate_solid, domain, port=3939):
    from build123d import Pos
    from ocp_vscode import port_check, show

    if not port_check(port):
        raise ConnectionError(f"Start OCP CAD Viewer in VS Code on port {port}.")
    width = domain["grid"]["shape"][0] * domain["grid"]["spacing_mm"][0]
    offset = width + 20.0
    shapes = [Pos(-offset, 0, 0) * reference_solid, candidate_solid, Pos(offset, 0, 0) * _preserve_shape(domain)]
    names = ["v0 reference", "free topology candidate", "prescribed interfaces only"]
    show(*shapes, names=names, colors=["#7994ab", "#248c87", "#d98236"], port=port, progress="", timeit=False)
    return {"shown": names, "separation_mm": offset, "port": port}


def render_saved_evidence(domain_path, field_path, reference_step, candidate_step, history_path, output_dir):
    domain = json.loads(Path(domain_path).read_text(encoding="utf-8"))
    with np.load(field_path, allow_pickle=False) as data:
        for name in ("allowed", "preserve", "forbidden"):
            domain[name] = data[name].copy()
        density = data["density"].copy()
    saved = json.loads(Path(history_path).read_text(encoding="utf-8"))
    history = saved["history"] if isinstance(saved, dict) else saved
    return render_topology_evidence(domain, density, import_step(reference_step), import_step(candidate_step), history, output_dir)
