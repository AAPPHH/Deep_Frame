from ocp_vscode import port_check, show

from deep_frame.config import CONFIG
from deep_frame.model import build_smoke_body, export_body


def main():
    body = build_smoke_body(CONFIG)
    export_body(body, CONFIG["stl_path"])
    if not port_check(CONFIG["viewer_port"]):
        raise ConnectionError(f"Start OCP CAD Viewer in VS Code on port {CONFIG['viewer_port']}.")
    show(body, names=["Smoke"], port=CONFIG["viewer_port"], progress="", timeit=False)


if __name__ == "__main__":
    main()
