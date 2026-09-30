import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path


CONFIG = {
    "url": "https://www.dhondt.de/calculix_2.22_4win.zip",
    "sha256": "a1f91281944c96d6cd914cc020421e8ae65973b3e15d055dc63a3e3e3066d281",
    "destination": Path(sys.prefix) / "calculix",
    "timeout_s": 60,
}


def install(config):
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this installer with the project .venv Python")
    destination = Path(config["destination"]).resolve()
    with urllib.request.urlopen(config["url"], timeout=config["timeout_s"]) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != config["sha256"]:
        raise ValueError("CalculiX archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError("Unsafe archive member path")
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)
    return str(destination)


if __name__ == "__main__":
    install(CONFIG)
