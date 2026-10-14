# Imsil DL catalog (port of the Haenam pipeline)

This folder rebuilds the 2025 Imsil catalog with the frozen pipeline adopted for the 2020–2026 Haenam sequences:
PhaseNet+ picks → PyOcto association → HYPOINVERSE → HypoDD relocation (waveform cross-correlation, 1,000-replica
bootstrap, 25 m reference rule) → local magnitude → FMF template matching.

- **Source:** the Haenam code, copied from the read-only reference `~/works/_moho_reference/27.Haenam_sequence/DL_detection` (moho original: `27.Haenam_sequence/DL_detection`).
- **Period:** 2025-09-01 to 2026-01-31 (2025 DOY 244–365, 2026 DOY 1–31).
- **Network:** KS stations within 100 km of Imsil. There are no KG data for 2025, and the JU stations near Imsil start on 2026-09-01.

## Rules of the port
- `pipeline/core.py`, `pick_augmentation.py`, the velocity models, the SAC export, the magnitude engine and the FMF scan code are byte-identical to Haenam's. `../tools/port_check.sh` checks this.
- Region values, paths and labels changed only by stated rules. They are marked `IMSIL PORT` in the code and listed in `docs/decisions.csv`.
- Open choices (D1–D16) keep a Haenam-like default until the stage notebook shows the evidence and you decide.
- Every script is a dry run unless you add `run` (shell) or `--apply` (python).

## Run order
| Step | Command | Output | Check |
|---|---|---|---|
| S0 | `../tools/phase0_copy_from_moho.sh run` | `../_external/` (EQNet + weights, korea-cluster-relocation, fast_matched_filter, HypoDD, binaries), `../data/` | weights sha256 = Haenam's |
| S0 | `../envs/build_envs.sh all run`, then `~/miniforge3/bin/python ../envs/verify_envs.py` | envs `dl-base`, `pq-gpu`, `fmf` in `~/miniforge3/envs` | key versions = moho |
| S0 | `../tools/smoke_envs.sh`; notebook `00` | `logs/smoke_envs_*.txt` | 8/8 pass |
| S1 | `./run_imsil.sh stations run` | `station_table/stations.csv` (54 KS; KWJU dropped, elevations 0 m) | — |
| S1 | `./run_imsil.sh mirror list 2025-2026 run`, `mirror plan 2025-2026 run`, `mirror pull 2025 run`, `mirror pull 2026 run` | `../continuous_mara/` (~214 GB) | `mirror status 2025 --assert-complete`, same for 2026 |
| S1 | `./run_imsil.sh link run`, `./run_imsil.sh epochs run`; notebook `01` | `continuous/` symlinks, `station_table/station_epochs.csv` | coverage per station and day |
| S2 | `./run_imsil.sh detect 2025 306-307 run`, `locate 2025 run`, `catalog run`; notebook `02` | smoke catalog | 24/24 KMA events within 12 s; ML 1.9 within ~1 km of 35.5506/127.2951 |
| S2 | move the smoke outputs aside: `mv models/phasenet_plus models/phasenet_plus_smoke` | — | — |
| S3 | `./run_imsil.sh years 2025-2026 run`, `catalog run`, `status 2025-2026 run`; notebook `03` | `logs/full_run_params.json` (freeze), `catalogs/imsil_dl_catalog.csv` | frozen values = Haenam's except the box |
| S4 | `XCORR_TRACE_CACHE=30000 ./run_imsil.sh dl-reloc --years 2025-2026 --min-depth 10 --open-wdcc --n-boot 1000 --blocked-pairs 45 run` | `runs/hypodd_dl/`, `catalogs/imsil_dl_relocated.csv` | condition numbers 40–80 |
| S4 | `python analysis/add_reloc_origin_times.py`, `python catalogs/build_reference_25m.py --apply`; notebook `04` | `catalogs/reference_25m/DL_reference_25m.csv` | compare with the earlier 18-event dt.cc geometry |
| S5 | `python local_magnitudes/full_period/full_magnitudes.py --out local_magnitudes/full_period --workers 6`, then `validate_full_magnitudes.py --out local_magnitudes/full_period`; notebook `05` | `local_magnitudes/full_period/full_period_ML.csv` | ML − KMA ML |
| S5 | `full_magnitudes.py` only (no validator) with `IMSIL_RESPONSE_MODE=resp_current` and `--out local_magnitudes/full_period_resp_current`; notebook `05a` | D4 and D18 comparison | the literal validator quarantines every RESP component (D18), so it cannot run on resp_current |
| S6 | `python catalogs/build_template_catalog_ref25.py`, `python fmf/build_station_list.py --apply` | `catalogs/imsil_templates_ref25.csv`, `fmf/stations_imsil_5_20H.csv` | — |
| S1b | `python tools/repair_stub_records.py --days 2025.343 2025.356 --apply` (D19), then re-run detect/locate for those days | `../continuous_repaired/` (cleaned copies + `repair_manifest.csv`), relinked `continuous/` | ObsPy reads every repaired file; kept records byte-identical |
| S6 | frozen comparison: `cd fmf && ./run_imsil_5_20H.sh templates`, `check 2025 306`, `tools/fmf_scan_days.sh fmf run`, `catalog`. FINAL (D20, D23): `cd fmf_aligned && FMF_RUN=imsil_5_20H_aligned_mad10 FMF_MAD_K=10 ./run_imsil_5_20H.sh templates`, `check 2025 306`; `FMF_RUN=imsil_5_20H_aligned_mad10 FMF_MAD_K=10 tools/fmf_scan_days.sh fmf_aligned run`; `FMF_CC_FLOOR=0 FMF_CATALOG_OUT=... ./run_imsil_5_20H.sh catalog`; notebooks `06`-`09` | `catalogs/fmf_final/imsil_fmf_catalog.csv` (235 detections: 28 reference, 207 new) + `settings.json` | every template detects itself (self CC 1.000) |
| S7 | `python reloc/export_fmf_detections.py --apply`; `PATH=../_external/bin:$PATH` + `imsil.env`, then pq-gpu `python reloc/run_fmf_hypodd.py --n-boot 1000`; `python catalogs/build_reference_25m.py --run-root runs/hypodd_fmf --relocated catalogs/imsil_fmf_relocated.csv --outdir catalogs/reference_25m_fmf --apply`; notebook `10` | `runs/hypodd_fmf/`, `catalogs/imsil_fmf_relocated.csv`, `catalogs/reference_25m_fmf/` (101 events: 28 references + 73 detections) | detections carry no catalog links (D22); references move ~1 m |
| A1 | notebook `11` (builder `notebooks/build_11_migration.py`) | diffusion front and back front on the 101 events of `reference_25m_fmf` | time-scale settings are D24 (open) |
| S8 | long FMF scan (D25, D26): `cd fmf_aligned && ./run_imsil_5_20H_long.sh templates`, `check 2025 306`; then `setsid nohup python tools/fmf_long_rolling.py run --apply &` (pull a year chunk from MARA, scan one day per process, delete; the adopted period is the control and is never deleted); `python tools/fmf_long_rolling.py status`; `FMF_CC_FLOOR=0 ./run_imsil_5_20H_long.sh catalog` | `fmf_aligned/catalogs/daily_shelly_imsil_5_20H_aligned_mad10_long_seg/`, log `logs/fmf_long_rolling.log`, state `../continuous_mara/_manifest_long/state.json` | every template self-detects (CC 1.000); control recovers the adopted catalog |
| A3 | `python tools/link_keep_long.py --fetch 2018.041,2018.065,2018.173 --apply` (kept hit days linked into `continuous/`), then notebook `13` (builder `notebooks/build_13_long_fmf_candidates.py`, kernel fmf-imsil, ~20 min) | waveforms of the 22 long-scan candidates + one noise-burst reference; `catalogs/fmf_long_candidates_evidence.csv`, `notebooks/figs_13/` | evidence only; which candidates are genuine is the user's call |
| A4 | notebook `14` (follow-up: why KMA missed 2019-04-15; group B = source near JASA; why only 400005) and `tools/run_fmf_reloc_2019A.sh [TAG]` (A1-A3 relocated with all 235 swarm events, D22 settings) + notebook `15` | `catalogs/imsil_fmf_relocated_2019A{,_full}.csv`, `catalogs/reference_25m_fmf_2019A{,_full}/`, `runs/hypodd_fmf_2019A{,_full}`, `runs/hypodd_fmf_2019A_full/solo_A1A3/` (written and run by notebook 15, section 5) | 2019A_full = 2019A by construction (links limited to the 8-9 template-picked stations). The solo run (A1-A3 alone, SVD, cc only) keeps their starting centroid by construction; only the swarm links measure the offset |
| A5 | notebook `16` (builder `notebooks/build_16_report_figures.py`, kernel dl-base, ~6 min) and `docs/reports/imsil_2025_summary_20261014/report.tex` (`tectonic -X compile report.tex`) | summary report: 7 figures (PDF + PNG), `facts.json`/`facts.tex` (every number of the report), `report.pdf` | report numbers are macros; values recomputed by notebook 16 are asserted equal to the measuring notebook |
| A2 | notebook `12` (builder `notebooks/build_12_park2026_comparison.py`; ~6 min, reads continuous 2025.306–307) | comparison with Park et al. (2026, PEPI 381, 107613; `../papers/`): timing, magnitudes on their scale, absolute location, geometry, migration, mechanism, waveform similarity, b value | comparison only: their magnitude and migration recipes re-implemented; nothing in the pipeline changed |

Python commands run in the `dl-base` env (`~/miniforge3/envs/dl-base/bin/python`); `run_imsil.sh` and the FMF drivers pick their own envs.
Never run detection, cross-correlation and FMF on the GPU at the same time.

## Traps carried over from Haenam
- **Caches keyed on file existence:** SAC export skips existing event folders, `bootstrap_errors.csv` is reused if present, and `--xcorr-resume` reuses pair files. After any change, use a fresh run root or `FMF_RUN`.
- **Whole-second origin times:** the relocation framework writes `origin_time` to whole seconds. Run `analysis/add_reloc_origin_times.py` before anything that needs sub-second origins.
- **Single-cell reruns** leave tables stale; run each script in full.
- **Never edit a shell script while it runs:** bash reads `run_imsil.sh` as it executes. An edit during the 2025 mirror pull (2026-10-07) made the script resume at a shifted position and fail after that pull, so the chained 2026 pull never started.
- **FMF batch on PAGO:** the loader's forked worker processes die (BrokenProcessPool) when forked from a process that has already scanned a day, so `run_full_period.sh batch` (30-day chunks) loses every day after the first of a chunk. Use `tools/fmf_scan_days.sh` (one fresh process per day; identical results). In a notebook, give `scan` forkserver workers (see notebook 07). The exact trigger is not identified (not CUDA initialisation alone, not memory, not BLAS threads).
- **Repaired day files (D19):** `tools/link_continuous.py` restores the original links to the MARA files; re-run `tools/repair_stub_records.py --apply` after any `link run`.
- **Two FMF codes (D20):** `fmf/` is the frozen Haenam code (byte-identical, checked by port_check); `fmf_aligned/` differs only in `scan._to_grid` (every day trace shifted onto the exact 10 ms grid) used by `templates._process_day` and `scan._load_channel`.
- **Fortran binaries on PAGO:** `hyp1.40`, `ph2dt` and `ncsn2pha` need the `libgfortran.so.3` shim, so `run_imsil.sh` puts `../_external/bin` (wrappers plus hypoDD v2.1beta) first on `PATH`.
