# MACE-H-LR — Refactor and Repository Migration Plan

**Status:** implemented in the local working tree on 2026-08-17; standalone
repository publication, the `v0.1.0` tag, and public push remain intentionally
pending group approval.
**Scope:** restructure `MACE-H-mg` for readability and handoff, then migrate it out of the fork into a standalone repository.
**Upstream:** MACE-H (`maurergroup/MACE-H`, arXiv:2508.15108) → DeepH-E3 (`Xiaoxun-Gong/DeepH-E3`, arXiv:2210.13955). MIT licensed.

Implementation verification: local `pre-refactor` tag at
`2bad8021ca7581207de0fdb817ed3f6266c16a55`; 289 default CPU tests pass and
4 environment-dependent tests skip; all 75 curated result artifacts have
sibling checksum/provenance manifests. No remote was added or pushed.

---

## 1. Goals

1. A newcomer can install the project, run a smoke test, and regenerate one figure without asking anyone a question.
2. Library code, campaign code, generated data, and curated results are separable and separately versioned.
3. The provenance of every inherited line of code is legible and legally attributed.
4. Nothing about the science changes. Numerical outputs before and after must match.

### Non-goals

- Improving model accuracy, adding features, or refactoring algorithms. Behaviour-preserving moves only.
- Reorganising the 134 GB / 65 GB data trees. Those are documented, not restructured.
- Merge compatibility with upstream MACE-H. Abandoned deliberately (see §11).

---

## 2. The invariant

> **`workflows → maceh`, never `maceh → workflows`.**

This is the whole point of the refactor and the criterion for every judgement call below. The library must be importable and testable with no campaign code present.

The current tree violates this: `maceh/data.py` and `maceh/epc/mgo_long_range.py` import from the root-level `mgo_lr` package. Moving `mgo_lr` wholesale into `workflows/` would harden that inversion rather than fix it — which is why §7 splits it instead of moving it.

The invariant is enforced by a test (§9), not by discipline.

---

## 3. Target layout

```text
mace-h-lr/
├── README.md                  # what it is, install, smoke test, reproduce one figure
├── LICENSE                    # MIT, stacked copyright (see §6)
├── NOTICE                     # lineage, prose
├── CITATION.cff
├── pyproject.toml
├── environment.yml
├── licenses/                  # third-party license texts
├── .github/workflows/ci.yml
│
├── src/maceh/                 # LIBRARY — importable, no campaign logic
│   ├── paths.py               # sole resolver of $MACEH_DATA_ROOT / $MACEH_RUNS_ROOT
│   ├── data/
│   │   ├── graph.py
│   │   ├── preprocessing.py
│   │   ├── structures.py      # supercells, displacements
│   │   └── io/
│   │       ├── abacus.py
│   │       └── blocks.py      # Hamiltonian block conversion
│   ├── models/                # e3nn layers, readouts
│   ├── training/              # reusable train/eval logic
│   ├── response/              # Z*, eps_inf, long-range Hamiltonian math
│   ├── epc/                   # supercells, finite differences, tensor assembly
│   ├── analysis/              # reusable plotting/numerics helpers
│   ├── external/              # adapters over installed third-party packages
│   ├── _vendor/               # copied third-party source (was from_*)
│   │   └── <name>/ORIGIN      # upstream URL, revision, license
│   ├── default_configs/       # packaged resources
│   └── cli/                   # thin command implementations
│
├── workflows/                 # CAMPAIGNS — everything that runs something
│   ├── mgo_dataset/           # stages, workspace ops, generation, validation, export
│   ├── training/              # INIs, campaign launchers, monitoring, reports
│   ├── epc/                   # benchmark and reference campaigns
│   └── analysis/              # figure and deck generators for specific outputs
│
├── configs/                   # example/template INIs
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── smoke/
│   ├── data/                  # tiny committed fixtures only
│   └── conftest.py
├── results/                   # curated, publication-grade outputs + manifests
├── provenance/                # frozen dataset record (keep as-is)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA.md
│   ├── design/
│   └── archive/               # was instructions/
│
├── data/                      # gitignored; convenience pointer to $MACEH_DATA_ROOT
└── runs/                      # gitignored; convenience pointer to $MACEH_RUNS_ROOT
```

**`external/` vs `_vendor/`.** `external/` holds adapters — thin interfaces over packages installed from PyPI. `_vendor/` holds copied source that cannot be replaced by a dependency. The current `from_nequip/`, `from_dimenet/`, `from_mfn/`, `from_pymatgen/`, `from_schnetpack/` are copied source, so they go to `_vendor/`, each with an `ORIGIN` file recording upstream URL, revision, and license.

**`cli/` in the library is fine** provided it contains thin command implementations only. The console script comes from `pyproject.toml`:

```toml
[project.scripts]
maceh = "maceh.cli:main"
```

---

## 4. Phase overview

| Phase | Work | Est. | Blocking? |
|---|---|---|---|
| 0 | Freeze and safety net | 0.5 d | yes |
| 1 | Pre-flight audit | 0.5 d | yes |
| 2 | Prune | 0.5 d | no |
| 3 | Packaging: `pyproject` + `src/` move, internals untouched | 1 d | yes |
| 4 | Split `mgo_lr` | 1–1.5 d | yes |
| 5 | Split `training/` and `epc/` | 1 d | no |
| 6 | Tests | 0.5 d | no |
| 7 | Analysis and results | 0.5 d | no |
| 8 | Data and runs contract | 0.5 d | no |
| 9 | Docs | 0.5 d | no |
| 10 | Cutover and CI | 0.5 d | no |

Roughly 7–8 working days. Each phase is one PR ending on a green smoke test. **Never mix a `git mv` with a content edit in the same commit** — it makes the diff unreviewable and the revert impossible.

---

## 5. Phase 0 — Freeze and safety net

Nothing else in this document is safe without this phase.

1. Commit or revert the dirty tree (`README.md`, `.gitignore` are modified). Delete `CRASH` and `input_tmp.in`. Move the audience-feedback file to `docs/archive/`. Add `.claude/` to `.gitignore`.
2. `git tag pre-refactor` and push it.
3. **Write the smoke test.** Tiny config, ~2 structures, 2 epochs, exercising preprocess → train → eval → one EPC tensor element. Target under 2 minutes. This is the only thing standing between you and a silent breakage.
4. **Record golden numbers.** Commit `tests/smoke/golden.json` with a handful of reference values and tolerances — an SR test MAE, one EPC tensor component, one Z\* element, one ε∞ element:

```json
{
  "sr_test_mae_meV":      {"value": 0.0, "rtol": 1e-6},
  "epc_xx_meV_per_ang":   {"value": 0.0, "rtol": 1e-6},
  "born_Mg_xx":           {"value": 0.0, "rtol": 1e-6},
  "eps_inf_xx":           {"value": 0.0, "rtol": 1e-6}
}
```

The golden test is the acceptance criterion for every later phase. If it drifts, the move was not behaviour-preserving.

---

## 6. Phase 1 — Pre-flight audit

Three mechanical scans. Do them before pushing anything anywhere.

### 6.1 Licensing

Upstream is MIT (`Copyright (c) 2023 Xiaoxun-Gong`), which permits everything planned here on one condition: the copyright and permission notice must travel with the code.

The copyright line names the DeepH-E3 author, not the Maurer group — MACE-H kept the inherited LICENSE without adding their own line. Ordinary reading is that they distribute their contributions under the same terms. A one-line email asking them to add their copyright upstream costs nothing and is not blocking.

`LICENSE` in the new repo — same MIT text, stacked:

```text
MIT License

Copyright (c) 2023 Xiaoxun-Gong                        (DeepH-E3)
Copyright (c) 2025 Maurer Group, University of Warwick (MACE-H)
Copyright (c) 2026 <your full name>                    (MACE-H-LR)

Permission is hereby granted, free of charge, ...
[remainder verbatim, unchanged]
```

`NOTICE` — prose lineage, not legally required but it is what makes the repo legible:

```text
This project derives from MACE-H (github.com/maurergroup/MACE-H, arXiv:2508.15108),
forked at commit <SHA>, which derives from DeepH-E3
(github.com/Xiaoxun-Gong/DeepH-E3, arXiv:2210.13955). Both are MIT licensed.

Inherited largely unchanged:  src/maceh/{data,models,training}, configs/
New in this project:          src/maceh/response, src/maceh/epc,
                              workflows/mgo_dataset, workflows/epc

Third-party source under src/maceh/_vendor/ retains its original license;
see licenses/ and each _vendor/<name>/ORIGIN.
```

Capture the fork point while you still have the remote: `git merge-base HEAD upstream/main`.

**Vendored code is the remaining license question.** Each `from_*` directory is governed by its own upstream license, not by DeepH-E3's MIT. NequIP, SchNetPack, and pymatgen are MIT; verify DimeNet and MFN rather than assume. First find out what is even reachable:

```bash
grep -rn "from_dimenet\|from_mfn\|from_nequip\|from_schnetpack\|from_pymatgen" \
  --include='*.py' . | grep -v "^\./maceh/from_"
```

Anything with no live import is deleted in Phase 2 and the question dies with it. Whatever survives gets its license text into `licenses/` and an `ORIGIN` file.

### 6.2 Secrets and machine-specific paths

```bash
gitleaks detect --source . --log-opts="--all"
git grep -nI -e "DFGPT" -e "/mnt/c/Users/Main" -e "tailscale" $(git rev-list --all) | head -50
```

Cluster hostnames, Tailscale names, and absolute Windows paths should not survive into a public repo.

### 6.3 Large blobs

```bash
git count-objects -vH
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1=="blob"' | sort -k3 -n -r | head -30
```

Any stray checkpoint or `.h5` gets removed with `git filter-repo` **before** the first push to the new remote, never after.

---

## 7. Phase 2 — Prune

With no upstream to track, dead code is pure liability. Git remembers everything; reference the `pre-refactor` tag in the commit message.

- `visual_tools/` — legacy plotting, superseded.
- `test_tools/` — legacy diagnostics, superseded by the pytest suite.
- `inference_tools/` — keep the current implementation, delete the old one.
- `data-reproductivity/` — exists to reproduce someone else's paper. Delete unless you actually re-run the Au/2D checks. (Also: the name is a typo.)
- Unreachable `from_*` directories, per §6.1.
- `instructions/` → `docs/archive/`.

---

## 8. Phase 3 — Packaging

**Move the package. Do not reorganise its internals.** Splitting the move from the restructure is what makes both reviewable.

1. Add `pyproject.toml`.
2. `git mv maceh src/maceh` — nothing else.
3. `pip install -e .`, confirm the smoke test and golden numbers still pass.
4. Add the `maceh` console script with subcommands (`preprocess`, `train`, `eval`, `epc`) delegating to `src/maceh/cli/`.
5. Keep `deephe3-*.py` as thin wrappers **for the duration of the refactor** — they are a useful canary that the CLI actually works. Delete them at cutover (§14); nothing outside the repo consumes them. Note the breaking change in the README.

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "maceh-lr"                 # distribution name
requires-python = ">=3.9"
dynamic = ["version", "dependencies"]

[project.scripts]
maceh = "maceh.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
maceh = ["default_configs/*.ini", "**/*.json", "**/*.jl"]

[tool.pytest.ini_options]
addopts   = "--import-mode=importlib"
testpaths = ["tests"]
markers = [
  "slow: long-running",
  "gpu: requires CUDA",
  "dft: requires ABACUS/QE",
  "bigdata: requires $MACEH_DATA_ROOT",
]
```

### 8.1 Keep the import name `maceh`

Distribution name and import name are independent. Repo `mace-h-lr`, distribution `maceh-lr`, **`import maceh`**. Renaming the Python package would invalidate any checkpoint or graph cache pickled with `maceh.*` module paths — and there are 65 GB of those.

### 8.2 Packaged resources

Once inside `src/`, source-tree path assumptions break. Package the JSON files, Julia script, default INIs, and license files explicitly (above), and replace `os.path.dirname(__file__)` lookups with:

```python
from importlib.resources import files
default_ini = files("maceh.default_configs") / "train_default.ini"
```

---

## 9. Phase 4 — Split `mgo_lr`

The highest-risk phase, and the one that actually fixes the dependency inversion. `mgo_lr` currently holds two different kinds of code:

| Destination | Content |
|---|---|
| `src/maceh/data/`, `src/maceh/response/` | Supercells, long-range Hamiltonian assembly, displacement handling, ABACUS parsing, block conversion |
| `workflows/mgo_dataset/` | Stage definitions, workspace management, dataset generation, validation, export |

Procedure:

1. Inventory what the library already imports: `git grep -n "mgo_lr" src/maceh/`. Those symbols define the minimum that must move up.
2. Move reusable modules into `src/maceh/` one at a time, running the golden test after each.
3. Keep `mgo_lr/__init__.py` as a temporary compatibility shim re-exporting from the new locations, with a `DeprecationWarning`. Remove it at the end of Phase 5.
4. Move the remainder into `workflows/mgo_dataset/`.
5. `git grep -n "mgo_lr" src/maceh/` must return nothing.

### 9.1 Pickle trap — check before you move

If any cached artefact was pickled with classes defined in `mgo_lr.*`, moving those classes breaks unpickling and invalidates the caches. Check first:

```bash
strings <one-graph-cache-file> | grep -oE '^(maceh|mgo_lr)[.A-Za-z_]*' | sort -u | head
```

If `mgo_lr.*` appears, add module aliases alongside the shim so old artefacts still load:

```python
# src/maceh/_compat.py — imported from maceh/__init__.py during transition
import sys
from maceh.data import structures as _structures
sys.modules.setdefault("mgo_lr.supercell", _structures)   # one line per moved module
```

Add an integration test that loads one pre-refactor cache file. Retire the shim only once caches are regenerated.

### 9.2 The boundary test

Add this in Phase 4 and never remove it:

```python
# tests/unit/test_import_boundary.py
import ast, pathlib, pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "maceh"
FORBIDDEN = {"workflows", "mgo_lr", "plots", "plots2", "analysis_scripts"}

def _roots(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]

@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=str)
def test_library_never_imports_workflows(path):
    bad = FORBIDDEN & set(_roots(path))
    assert not bad, f"{path} imports workflow-layer module(s): {sorted(bad)}"
```

---

## 10. Phase 5 — Split `training/` and `epc/`

Same treatment, lower risk.

- `training/` → reusable train/eval/checkpoint logic to `src/maceh/training/`; INIs, campaign launchers, GPU scripts, monitoring, and provenance checks to `workflows/training/`.
- `epc/` → reusable supercell construction, finite differences, and tensor assembly are already in `maceh/epc/`; the top-level `epc/` becomes `workflows/epc/` (reference DFT campaigns, response scans, run outputs). The name collision resolves itself once the path prefix disambiguates.

Delete the `mgo_lr` shim at the end of this phase.

---

## 11. Phase 6 — Tests

One collection root, differentiated inside it. Merge `tests/`, `maceh/tests/`, and anything salvageable from `test_tools/` into:

```text
tests/
├── unit/          # fast, no I/O
├── integration/   # multi-component, small fixtures
├── smoke/         # the Phase 0 end-to-end test + golden numbers
├── data/          # tiny committed fixtures only
└── conftest.py
```

- Mark anything needing GPU, ABACUS/QE, or `$MACEH_DATA_ROOT`, so the default command stays fast:
  `pytest -m "not slow and not gpu and not dft and not bigdata"`
- Run against the **editable install with `--import-mode=importlib`**. This is what verifies the `src/` boundary — without it, the repo root silently makes imports succeed and the layout proves nothing.
- Remove every `sys.path` modification in the codebase. `git grep -n "sys.path"` must come back empty.

---

## 12. Phase 7 — Analysis and results

`plots/` and `plots2.0/` collapse. The version number in a folder name is the problem: nobody can tell which is current.

- Reusable plotting and numerics helpers → `src/maceh/analysis/`.
- Generators for specific figures and decks → `workflows/analysis/`.
- Curated, publication-grade outputs only → `results/`. Everything else is regenerable and gets ignored.

Note that `plots2.0/generate_plots.py` imports directly from root-level `epc` and `training`. Packaging alone does not fix this — it is fixed by Phases 4 and 5, and the boundary test keeps it fixed.

Every artefact in `results/` gets a sibling manifest:

```yaml
figure: epc_sr_vs_full_mgo.pdf
commit: <sha>
command: python workflows/analysis/epc_comparison.py --config configs/epc_mgo.ini
inputs:
  config:  workflows/epc/configs/epc_mgo.ini
  run_id:  <training run id>
  dataset: <provenance/ checksum>
environment: environment.lock.yml
sha256: <output checksum>
```

Those five fields — commit, command, input provenance, environment, checksum — are the difference between a figure someone can regenerate and a figure someone has to trust.

---

## 13. Phase 8 — Data and runs contract

Reference data and execution outputs have different lifecycles and must not share one ambiguous root.

```
$MACEH_DATA_ROOT   reference DFT/DFPT data, snapshots, loader splits   (~134 GB)
$MACEH_RUNS_ROOT   graph caches, checkpoints, logs, eval outputs        (~65 GB)
```

- Resolved in exactly one module, `src/maceh/paths.py`, which raises a clear, actionable error when unset.
- In-repo `data/` and `runs/` are gitignored convenience pointers, not architecture. The existing machine-specific `run` and `training_runs` symlinks are replaced by them.
- Tiny committed fixtures live in `tests/data/` and nowhere else.
- `docs/DATA.md` documents the tree, its size, and — most importantly — **which parts are cheap to regenerate and which represent weeks of DFT.**

**One deviation from the reviewed draft:** do not impose a generic `raw/interim/processed` convention on the data tree. Your existing structure (`reference/`, `pilot/`, `main/`, `test_large_cell/`, `loader_splits/`, `rejected/`) already describes what the data *is*, in the vocabulary of the project. A generic convention would add a translation layer for no benefit, and renaming directories inside a 134 GB tree is real risk for zero gain. Document the existing structure; don't rewrite it.

---

## 14. Phase 9 — Docs

- **`README.md`** — one page, linear: what this is → install → run the smoke test → reproduce one named figure. If a new person can follow it start to finish without asking a question, it is done.
- **`docs/ARCHITECTURE.md`** — the structural walkthrough you already wrote, updated to the new tree, plus a statement of the §2 invariant.
- **`docs/DATA.md`** — per §13.
- **`docs/design/`** — the current MgO-LR specification.
- **`docs/archive/`** — superseded plans, clearly marked as historical.

---

## 15. Phase 10 — Cutover and CI

A GitHub fork relationship is repository *metadata*, not git data. No need to ask GitHub to detach anything — create a fresh empty repository and push into it:

```bash
git remote add newrepo git@github.com:<user>/mace-h-lr.git
git push newrepo --all --tags
```

Full history preserved, no fork badge, upstream attribution intact in `git log` for free. Only squash to a fresh `init` if history contains blobs or secrets you cannot scrub — and prefer `git filter-repo` over squashing, since discarding your own development history makes the work look less credible, not more.

At cutover:

- Delete the `deephe3-*.py` wrappers.
- Add `.github/workflows/ci.yml` running `pytest -m "not slow and not gpu and not dft and not bigdata"` on push and PR. This is the single best guard against the next person silently breaking things.
- Add `CITATION.cff`.
- Tag `v0.1.0` at whatever state your SURF report describes.
- Confirm with Bernardi before making it public — it is group work, and the MgO dataset may not be ready to be seen.

---

## 16. Traps

| Trap | Mitigation |
|---|---|
| Renaming the import package invalidates 65 GB of caches | Keep `import maceh`; rename only repo and distribution (§8.1) |
| Moving `mgo_lr` classes breaks pickled artefacts | Check with `strings`; add `sys.modules` aliases (§9.1) |
| Repo root makes imports succeed, hiding layout violations | Editable install + `--import-mode=importlib` (§11) |
| Source-tree path assumptions break inside `src/` | `importlib.resources` + explicit `package-data` (§8.2) |
| Mixed move/edit commits | One `git mv` per commit; edits separately |
| `git blame` ruined by mass moves | Add move commits to `.git-blame-ignore-revs`, set `blame.ignoreRevsFile` |
| Secrets or large blobs discovered after the push | Phase 1 runs before Phase 10, always |

---

## 17. Minimum viable version

If time runs out, do **Phase 0 → Phase 1 → Phase 2 → Phase 9 → Phase 10** and stop.

A correct README, a runnable smoke test, a documented data contract, and four fewer dead directories help the next person more than a perfect `src/` layout with stale docs. Folder structure is the part they can fix themselves. Knowing which of `plots` and `plots2.0` is real, and how to regenerate 134 GB, is not.

Phase 1 is the only item that gets *harder* to fix after the repository goes up.
