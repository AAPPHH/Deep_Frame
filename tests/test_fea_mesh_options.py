import pytest

from deep_frame.fea import evaluate
from test_fea import analytical_beam, beam_inputs


@pytest.mark.parametrize("options", [{"mesh_second_order_linear": True}, {"mesh_high_order_optimize": 2}])
def test_opt_in_mesh_options_retain_real_beam_physics(tmp_path, options):
    solid, material, cases, settings, _ = beam_inputs(tmp_path)
    settings.update(options)
    result = evaluate(solid, material, [], cases, settings)
    expected = analytical_beam()
    assert result["status"] == "ok", result["diagnostics"]
    assert result["max_displacement_mm"] == pytest.approx(expected["deflection_mm"], rel=0.03)
    assert result["eigenfrequencies_hz"][0] == pytest.approx(expected["frequency_hz"], rel=0.03)
    assert result["mesh"]["minimum_jacobian_mm3"] > 0
    assert result["mesh"]["second_order_linear"] == options.get("mesh_second_order_linear", False)
    assert result["mesh"]["high_order_optimize"] == options.get("mesh_high_order_optimize", 0)
