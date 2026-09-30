import numpy as np
import pytest
import trimesh

from deep_frame.config import CONFIG
from deep_frame.model import build_smoke_body, export_body


@pytest.mark.parametrize("dimensions", [(30.0, 20.0, 3.0), (12.5, 8.0, 1.5)])
def test_body_and_stl(tmp_path, dimensions):
    config = CONFIG | dict(zip(("length_mm", "width_mm", "thickness_mm"), dimensions))
    body = build_smoke_body(config)

    assert body.is_valid
    assert len(body.solids()) == 1
    assert body.volume == pytest.approx(np.prod(dimensions))

    path = export_body(body, tmp_path / "exports" / "smoke.stl")
    mesh = trimesh.load_mesh(path)

    assert mesh.is_volume
    assert mesh.body_count == 1
    np.testing.assert_allclose(mesh.extents, dimensions)
    assert mesh.bounds[0, 2] == pytest.approx(0.0)
    assert mesh.volume == pytest.approx(body.volume)


@pytest.mark.parametrize("key", ["length_mm", "width_mm", "thickness_mm"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_dimensions(key, value):
    with pytest.raises(ValueError, match="Dimensions"):
        build_smoke_body(CONFIG | {key: value})
