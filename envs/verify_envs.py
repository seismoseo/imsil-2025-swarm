#!/usr/bin/env python
"""Compare the rebuilt PAGO envs with moho's exported versions (read-only check; exit 1 on any mismatch).

For each env the KEY packages (those the pipeline imports) must match moho exactly; every other package of the exported
pip list is reported as a count of matches/mismatches/missing (informative only). Also prints torch CUDA build and
whether the A100 is visible.
Run:  ~/miniforge3/bin/python envs/verify_envs.py            (any python; it calls each env's own interpreter)
"""
import json
import re
import subprocess
import sys

MF = "/home/msseo/miniforge3"
SPEC = "/home/msseo/works/_moho_reference/27.Haenam_sequence/docs/environment"
ENVS = {
    # env: (moho pip freeze, key packages)
    "dl-base": ("pip_freeze_base.txt", ["torch", "torchvision", "seisbench", "pyocto", "fsspec", "numba", "numpy",
                                         "pandas", "h5py", "obspy", "scipy", "pyrocko", "matplotlib", "pygmt"]),
    "pq-gpu": ("pip_freeze_pq-gpu.txt", ["torch", "seisbench", "obspy", "numpy", "pandas", "scipy", "numba", "h5py"]),
    "fmf": ("pip_freeze_fmf.txt", ["obspy", "numpy", "scipy", "pandas", "FastMatchedFilter"]),
}
# conda-built packages appear as "name @ file:///..." in moho's freeze; their versions come from the conda yml instead
CONDA_YML = {"dl-base": "conda_base.yml", "pq-gpu": "conda_pq-gpu.yml", "fmf": "conda_fmf.yml"}
PROBE = r'''
import json, importlib.metadata as md
out = {d.metadata["Name"].lower(): d.version for d in md.distributions()}
try:
    import torch; out["__torch_cuda__"] = torch.version.cuda; out["__gpu__"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
except Exception as e:
    out["__torch_cuda__"] = f"n/a ({type(e).__name__})"
print(json.dumps(out))
'''


def norm(v):
    # PEP 440-style normalisation of numeric components ("2025.12.04" == "2025.12.4"); local tags (+cu128) kept
    if v is None:
        return None
    base, _, local = str(v).partition("+")
    parts = [str(int(x)) if x.isdigit() else x for x in base.split(".")]
    return ".".join(parts) + (f"+{local}" if local else "")


def moho_versions(env):
    v = {}
    for line in open(f"{SPEC}/{CONDA_YML[env]}"):
        m = re.match(r"^\s+- ([A-Za-z0-9_.\-]+)=([^=\s]+)$", line)                 # conda: "  - name=ver"
        if m:
            v[m.group(1).lower()] = m.group(2)
        m = re.match(r"^\s+- ([A-Za-z0-9_.\-]+)==([^\s]+)$", line)                 # yml pip block: "      - name==ver"
        if m:
            v[m.group(1).lower()] = m.group(2)
    for line in open(f"{SPEC}/{ENVS[env][0]}"):
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s]+)$", line.strip())
        if m:
            v[m.group(1).lower()] = m.group(2)
        m = re.match(r"^-e git\+.*#egg=([A-Za-z0-9_.\-]+)", line.strip())          # editable FMF: version from yml pip list
        if m and m.group(1).lower() not in v:
            v[m.group(1).lower()] = "1.5.0"
    return v


def main():
    bad = 0
    for env, (_, keys) in ENVS.items():
        py = f"{MF}/envs/{env}/bin/python"
        try:
            have = json.loads(subprocess.run([py, "-c", PROBE], capture_output=True, text=True, timeout=300, check=True).stdout)
        except Exception as e:                                                     # noqa: BLE001
            print(f"[{env}] NOT USABLE: {e}"); bad += 1; continue
        want = moho_versions(env)
        print(f"[{env}] torch CUDA {have.get('__torch_cuda__')}, GPU {have.get('__gpu__', '-')}")
        for k in keys:
            w, h = want.get(k.lower()), have.get(k.lower())
            ok = (w is not None and h is not None and norm(h).split("+")[0] == norm(w).split("+")[0] and ("+" not in w or norm(h) == norm(w)))
            bad += 0 if ok else 1
            print(f"    {'ok ' if ok else 'BAD'} {k:20s} moho {w}   PAGO {h}")
        others = [k for k in want if k not in {x.lower() for x in keys}]
        same = sum(1 for k in others if norm(have.get(k)) == norm(want[k])); miss = sum(1 for k in others if k not in have)
        diff = [f"{k} {want[k]}->{have[k]}" for k in others if k in have and norm(have[k]) != norm(want[k])]
        print(f"    other exported packages: {same} same, {len(others) - same - miss} different version, {miss} not installed (informative)")
        if 0 < len(diff) <= 12:
            print("      different:", ", ".join(diff))
    print("RESULT:", "PASS" if bad == 0 else f"FAIL ({bad} key mismatches / unusable envs)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
