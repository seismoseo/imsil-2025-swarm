#!/usr/bin/env python
# Builder of notebooks/00_infrastructure_and_port_check.ipynb (Imsil DL-catalog port, stage S0).
# Writes the UNEXECUTED notebook; the user executes it in the dl-base kernel after
#   tools/phase0_copy_from_moho.sh run, envs/build_envs.sh all run, tools/smoke_envs.sh
# Every emitted code cell is compiled here before the notebook is written (cells use # comments only).
#   python notebooks/build_00_infrastructure.py
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "00_infrastructure_and_port_check.ipynb")
cells = []


def md(s):
    cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))


def co(s):
    s = s.strip("\n")
    compile(s, f"<cell {len(cells)}>", "exec")          # cell-level syntax check
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s))


md(r"""
# 00 Infrastructure and port check

This notebook checks that PAGO can run the Haenam detection pipeline for the 2025 Imsil swarm before any data are processed.
It answers four questions:

1. **Environments.** Do the three rebuilt Python environments (`dl-base`, `pq-gpu`, `fmf`) carry the same package versions as moho?
2. **Smoke tests.** Do the picker, the associator, HYPOINVERSE, hypoDD, the relocation framework and the template matcher run on PAGO and reproduce known results?
3. **Port check.** Is the ported code identical to the Haenam code, apart from the region values, paths and labels listed in the decision log?
4. **Provenance.** Where did each copied file come from (commit, checksum)?

Nothing here changes any file; the cells only read logs and run read-only checks.
""")
co(r"""
import os, re, glob, subprocess
import pandas as pd
pd.set_option('display.width', 200); pd.set_option('display.max_colwidth', 120)
PROJECT = '/home/msseo/works/72.2025_Imsil_swarm'
DL = f'{PROJECT}/DL_detection'
EXT = f'{PROJECT}/_external'
MF = '/home/msseo/miniforge3'
checks = []   # (check, status, detail) rows for the summary


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr
""")

md(r"""
## Environments

`envs/verify_envs.py` compares each rebuilt environment with moho's exported package list (2026-10-04).
The key packages, the ones the pipeline imports, must match exactly. All other packages are only counted.
""")
co(r"""
rc, txt = run(f'{MF}/bin/python {PROJECT}/envs/verify_envs.py')
print(txt)
checks.append(('Environments match moho (key packages)', 'pass' if rc == 0 else 'FAIL', txt.strip().splitlines()[-1] if txt.strip() else 'no output'))
""")

md(r"""
## Smoke tests

`tools/smoke_envs.sh` runs eight short tests in a scratch directory. The two most informative ones reproduce earlier results byte for byte:
the ML 1.9 HYPOINVERSE solution of `1.HypoInv/kim2011` (35.5506° N, 127.2951° E, 14.40 km). The hypoDD example must agree with moho's run within 1 m and 0.01 s: the binary is the same, but the system math library differs between the two machines, so the last digits differ.
The cell shows the latest log.
""")
co(r"""
logs = sorted(glob.glob(f'{DL}/logs/smoke_envs_*.txt'))
if logs:
    txt = open(logs[-1]).read()
    print(os.path.basename(logs[-1])); print('\n'.join(l for l in txt.splitlines() if l.startswith(('===', '    PASS', '    FAIL', 'RESULT'))))
    m = re.search(r'RESULT: (\d+) passed, (\d+) failed', txt)
    status = 'pass' if m and m.group(2) == '0' else 'FAIL'
    checks.append(('Smoke tests (tools/smoke_envs.sh)', status, m.group(0) if m else 'no RESULT line'))
else:
    print('no smoke log yet: run tools/smoke_envs.sh'); checks.append(('Smoke tests (tools/smoke_envs.sh)', 'not run', '-'))
""")

md(r"""
## Port check against the Haenam code

`tools/port_check.sh` compares `DL_detection` with the read-only Haenam copy:

- **Must be identical:** the picker and association code (`core.py`, `pick_augmentation.py`, ...), the velocity models, the SAC export, the magnitude engine and the FMF scan code.
- **Whitelisted:** region values, paths and labels. Their full diffs are written to `logs/port_diff_vs_haenam.txt`.
- **Other files:** listed; a modified copy that is not whitelisted fails the check.

The `pipeline/config.py` diff below shows every region value that changed.
""")
co(r"""
rc, txt = run(f'{PROJECT}/tools/port_check.sh')
print(txt)
checks.append(('Port check (identical core, whitelisted diffs only)', 'pass' if rc == 0 else 'FAIL', [l for l in txt.splitlines() if l.startswith('RESULT')][0] if 'RESULT' in txt else 'no RESULT'))
rep = open(f'{DL}/logs/port_diff_vs_haenam.txt').read()
sec = rep.split('---------------------------------------------------------------- pipeline/config.py')[1].split('----------------------------------------------------------------')[0]
print('pipeline/config.py vs Haenam (changed lines only):')
print('\n'.join(l for l in sec.splitlines() if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))))
""")

md(r"""
## Copied code, weights and binaries

`tools/phase0_copy_from_moho.sh` records every copy in `_external/PROVENANCE.tsv` and saves moho's `git diff` of each code tree in `_external/provenance/`.
The PhaseNet+ weights checksum must equal the one frozen in Haenam's `logs/full_run_params.json`.
""")
co(r"""
prov_path = f'{EXT}/PROVENANCE.tsv'
if os.path.exists(prov_path):
    prov = pd.read_csv(prov_path, sep='\t'); display(prov)
    w = prov[prov['item'] == 'model_99.pth']
    ok = len(w) and w.checksum.iloc[-1].endswith('f1643330e34bffaa4f7e3a5f7dd9205fb4019d76e5b9b6cb598232360d22f384')
    checks.append(('PhaseNet+ weights = Haenam frozen weights', 'pass' if ok else 'FAIL', w.checksum.iloc[-1][:30] + '...' if len(w) else 'missing'))
else:
    print('no provenance yet: run tools/phase0_copy_from_moho.sh run'); checks.append(('PhaseNet+ weights = Haenam frozen weights', 'not run', '-'))
for b in ['hypoDD', 'hyp1.40', 'ph2dt', 'ncsn2pha']:
    p = f'{EXT}/bin/{b}'
    print(f'{b:9s}', run(f'md5sum {p} 2>/dev/null; head -c 300 {p} | grep -a exec')[1].strip().replace(chr(10), ' | ') if os.path.exists(p) else 'missing')
""")

md(r"""
## Decision log

Every setting that differs from Haenam, and every open choice, is listed in `docs/decisions.csv` with its default.
The **rule** rows change a value by a stated rule, for example the same box half-widths centred on Imsil.
The **open** rows wait for evidence from the stage notebooks and for your decision.
""")
co(r"""
dec = pd.read_csv(f'{DL}/docs/decisions.csv')
display(dec[['id', 'setting', 'haenam_value', 'imsil_default', 'status', 'evidence']])
n_open = int((dec.status == 'open').sum())
checks.append(('Decision log', 'info', f"{len(dec)} rows: {int((dec.status == 'rule').sum())} rule, {int((dec.status == 'decided').sum())} decided, {int((dec.status == 'stated').sum())} stated, {n_open} open"))
""")

md(r"""
## Comprehensive summary
""")
co(r"""
summary = pd.DataFrame(checks, columns=['check', 'status', 'detail'])
display(summary)
ready = all(s in ('pass', 'info') for s in summary.status)
print('Ready for stage S1 (stations and data mirror):', 'yes' if ready else 'no, fix the rows above that do not pass')
print()
st = dict(zip(summary.check, summary.status))
smoke_txt = open(logs[-1]).read() if logs else ''
print('Take-home points (from the checks above)')
print('  - Port check:', 'the Imsil code equals the Haenam code apart from the whitelisted region values, paths and labels in docs/decisions.csv.'
      if st.get('Port check (identical core, whitelisted diffs only)') == 'pass' else 'NOT clean; see logs/port_diff_vs_haenam.txt.')
print('  - Environments:', 'key package versions equal moho.' if st.get('Environments match moho (key packages)') == 'pass' else 'not yet equal to moho (see the environment table).')
s4 = re.search(r'=== S4.*?\n    (PASS|FAIL)', smoke_txt, re.S); s5 = re.search(r'=== S5.*?\n    (PASS|FAIL)', smoke_txt, re.S)
print('  - ML 1.9 HYPOINVERSE solution reproduced on PAGO (S4):', s4.group(1) if s4 else 'not run',
      '| hypoDD example = moho within 1 m (S5):', s5.group(1) if s5 else 'not run')
print(f'  - Decision log: {n_open} open decisions; they are settled in the stage notebooks 01-06.')
print()
print('Reproduce:  tools/phase0_copy_from_moho.sh run; envs/build_envs.sh all run; tools/smoke_envs.sh; then re-run this notebook.')
print('Next:       ./run_imsil.sh stations run, then the mirror (notebook 01).')
""")

nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
          metadata=dict(kernelspec=dict(name="dl-base", display_name="dl-base (Imsil DL)", language="python"),
                        language_info=dict(name="python")))
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:02d}"
json.dump(nb, open(OUT, "w"), indent=1)
print(f"wrote {OUT}: {len(cells)} cells ({sum(c['cell_type'] == 'code' for c in cells)} code, all compiled)")
