# Architecture

MACE-H-LR has one directional dependency rule:

> `workflows -> maceh`, never `maceh -> workflows`.

`src/maceh/` is the installed library. It owns graph/model code, reusable
ABACUS and block I/O, structures, analytic long-range response, EPC numerics,
packaged defaults, and path resolution. It must import and run without any
campaign package present. `tests/unit/test_import_boundary.py` enforces this
rule by parsing every library import.

`workflows/` owns operations tied to this MgO project: dataset state machines,
DFT/DFPT preparation and collection, campaign launchers, monitoring, reference
EPC calculations, response scans, and figure/deck generation. Workflows may
compose any public library module.

`results/` contains curated outputs. `provenance/` identifies external input
data but is not the data itself. Generated graphs, checkpoints, logs, and DFT
trees live outside Git under the roots documented in `DATA.md`.

## Library areas

- `maceh.data`: graph dataset, preprocessing, structures, ABACUS and HDF5 I/O.
- `maceh.response`: long-range Hamiltonian math and provenance checks.
- `maceh.epc`: finite differences, supercells, derivative storage, assembly.
- `maceh.analysis`: reusable distance/locality numerics.
- `maceh._vendor`: copied source with explicit origins and licenses.
- `maceh.cli`: thin command dispatch to the existing kernel/EPC interfaces.

The import name remains `maceh`; only the distribution name is `maceh-lr`.
This preserves serialized module paths used by existing graph/model artifacts.
