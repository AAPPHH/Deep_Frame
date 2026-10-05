import json
import subprocess
import sys
import threading
from time import perf_counter
from types import SimpleNamespace

import pytest

from tools import formulation_study as study

def test_neural_command_selects_the_shared_neural_optimizer(monkeypatch,tmp_path):
    import json
    path=tmp_path/'neural.json'
    path.write_text(json.dumps({"mma":{"optimizer":"mma"}}))
    called=[]
    monkeypatch.setattr(study,'frame_mma',lambda config:called.append(config))
    study.main(['frame_neural',str(path)])
    assert called[0]['mma']['optimizer']=='neural_al'


@pytest.mark.parametrize("kind,line,expected", [
    ("smi", "4242, 1536\n", {"process_gpu_bytes": 1536 * 2**20}),
    ("smi", "9999, 71680\n", {}),
    ("smi", "4242, [N/A]\n", {}),
    ("smi", "4242, nan\n", {}),
    ("smi", "4242, -1\n", {}),
    ("smi", "No running processes found\n", {}),
    ("counter", "1073741824 134217728\n", {"dedicated_bytes": 2**30, "shared_bytes": 2**27}),
    ("counter", "1,5e+09 2,5e+08\n", {"dedicated_bytes": 1.5e9, "shared_bytes": 2.5e8}),
    ("counter", "unavailable\n", {}),
])
def test_memory_probe_parses_pid_and_platform_counters(kind, line, expected):
    probe = study.MemoryProbe.__new__(study.MemoryProbe)
    probe.pid = 4242
    assert probe.parse(line, kind) == expected


class ProbePipe:
    def __init__(self, lines):
        self.lines, self.closed = lines, False
        self.consumed, self.ended = threading.Event(), threading.Event()
    def __iter__(self):
        yield from self.lines
        self.consumed.set()
        self.ended.wait()
    def close(self):
        self.closed = True


class ProbeReader:
    def __init__(self, command, force_kill):
        self.command, self.force_kill, self.killed = command, force_kill, False
        self.stdout = ProbePipe(["9999, 71680\n", "4242, [N/A]\n", "4242, 1536\n"] if command[0] == "nvidia-smi" else ["1073741824 134217728\n"])
    def poll(self):
        return 0 if self.stdout.ended.is_set() else None
    def terminate(self):
        if not self.force_kill:
            self.stdout.ended.set()
    def wait(self, timeout=None):
        if not self.stdout.ended.is_set():
            raise subprocess.TimeoutExpired(self.command, timeout)
        return 0
    def kill(self):
        self.killed = True
        self.stdout.ended.set()


@pytest.mark.parametrize("platform", ["posix", "nt"])
@pytest.mark.parametrize("force_kill", [False, True])
def test_memory_probe_is_portable_pid_scoped_and_joins_before_log_close(monkeypatch, tmp_path, platform, force_kill):
    local, polled, devices = threading.local(), threading.Event(), []
    local.device = 3
    class Device:
        def __init__(self, device):
            self.device = device
        def __enter__(self):
            local.device = self.device
            devices.append(self.device)
        def __exit__(self, *error):
            pass
    def memory_info():
        assert local.device == 3
        if threading.current_thread() is not threading.main_thread():
            polled.set()
        return 6 * 2**30, 8 * 2**30
    pool = SimpleNamespace(used_bytes=lambda: 256, total_bytes=lambda: 512)
    cupy = SimpleNamespace(cuda=SimpleNamespace(Device=Device, runtime=SimpleNamespace(getDevice=lambda: 3, memGetInfo=memory_info)), zeros=lambda size: None, get_default_memory_pool=lambda: pool)
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(study, "os", SimpleNamespace(name=platform, getpid=lambda: 4242))
    readers = []
    def spawn(command, **kwargs):
        reader = ProbeReader(command, force_kill)
        readers.append(reader)
        return reader
    monkeypatch.setattr(study.subprocess, "Popen", spawn)
    target = tmp_path / "memory.jsonl"
    probe = study.MemoryProbe(60, target)
    try:
        assert polled.wait(5)
        assert all(reader.stdout.consumed.wait(5) for reader in readers)
    finally:
        probe.close()
    assert all(not thread.is_alive() for thread in probe.threads)
    assert all(reader.stdout.closed and reader.poll() == 0 for reader in readers)
    assert all(reader.killed == force_kill for reader in readers)
    assert probe.log.closed and devices == [3]
    commands = [reader.command for reader in readers]
    assert any("--query-compute-apps=pid,used_memory" in command for command in commands)
    assert sum(command[0] == "powershell" for command in commands) == (platform == "nt")
    probe.phases.append(("probe", probe.started, perf_counter()))
    summary = probe.summary()
    assert summary["peaks"]["probe"]["process_gpu_gb"] == 1.5
    assert summary["peaks"]["probe"]["device_used_gb"] == 2
    assert summary["device_total_gb"] == 8
    if platform == "posix":
        assert "shared_gb" not in summary["peaks"]["probe"] and "dedicated_gb" not in summary["peaks"]["probe"]
        assert "unavailable" in summary["measurement_methods"]["shared_gb"]
    else:
        assert summary["peaks"]["probe"]["shared_gb"] == 0.125
        assert summary["peaks"]["probe"]["dedicated_gb"] == 1
    assert summary["reader_errors"] == {}
    samples = [json.loads(line) for line in target.read_text().splitlines()]
    assert any(row.get("process_gpu_gb") == 1.5 for row in samples)
    assert not any(row.get("process_gpu_gb") == 70 for row in samples)

@pytest.mark.parametrize('value,g',[(float('nan'),-1),(1,float('nan')),(float('inf'),-1)])
def test_reconstructed_physical_acceptance_rejects_nonfinite_measurements(tmp_path,monkeypatch,value,g):
    import numpy as np
    import trimesh
    from tools import formulation_study as study
    mesh=trimesh.creation.box(extents=[2,2,2])
    source=tmp_path/'frame.stl'
    mesh.export(source)
    domain={'grid':{},'material':{'density_g_cm3':1}}
    monkeypatch.setattr(study,'frame_setup',lambda *args:(domain,{}))
    monkeypatch.setattr(study,'shared_definition',lambda *args:{'sha256':'shared'})
    monkeypatch.setattr('deep_frame.topology_neural.cell_centers',lambda *args:np.zeros((1,3)))
    class Problem:
        def __init__(self,*args,**kwargs):
            pass
        def physical_report(self,physical):
            return {'rows':[{'name':'f1','status':'satisfied','value':value,'g':g,'limit':1,'unit':'Hz','sense':'>=','margin':1}]}
        def close(self):
            pass
    monkeypatch.setattr(study,'TopologyProblem',Problem)
    cfg={'mma':{'linear_solver':'none'},'shape':[2,2,2],'review':{'frame_source':str(source),'output_directory':str(tmp_path/'review')}}
    with pytest.raises(RuntimeError,match='violates shared mechanical'):
        study.frame_physical(cfg)
    report=json.loads((tmp_path/'review/physical_evaluation.json').read_text())
    assert report['passed'] is False
