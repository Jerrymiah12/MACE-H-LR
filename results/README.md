# Curated results and what their manifests claim

Every artifact in this tree has a sibling `<artifact>.manifest.yaml` recording
the commit, the command, the primary inputs, the environment file, and a
SHA-256 of the artifact.

## Scope of the checksum

**These manifests are for regression detection, not reviewer verification.**

The checksum answers one question: *did re-running this command on this machine,
in this environment, change the artifact?* That is the question that catches an
accidental change to a figure during a refactor, and it is the one the manifests
can actually answer.

The checksum does **not** promise that a stranger who clones this repository will
reproduce the same bytes. Matplotlib output depends on the FreeType build used
for font subsetting and on platform rasterisation, so figures generated on
another machine are expected to differ byte-for-byte while being visually and
numerically identical. A mismatch across machines is not evidence of a problem;
a mismatch on the *same* machine is.

`maceh.analysis.figures` removes the avoidable sources of drift — the PDF
`/CreationDate` and `/Producer` keys, the PNG `Software` key, and randomised SVG
element ids — so that a same-machine re-run is byte-stable. What remains is the
toolchain itself.

Upgrading to reviewer verification would mean recording the resolved Python,
matplotlib, FreeType, and NumPy versions in each manifest header and shipping a
container that pins them. That has not been done; do not read the current
manifests as though it had.

## Regenerating

Artifacts and manifests must be regenerated **together, from one run**, so that
every checksum in the tree describes the same environment. Regenerating a single
artifact and hand-editing its manifest produces a tree whose entries were made
under different conditions with nothing recording that fact.

Verification procedure before trusting a regeneration:

1. Run the generator twice on the same machine into different output
   directories and diff them. They must be identical.
2. Run it once on a second machine and diff against the first. Differences here
   are expected and are what confines these manifests to regression detection —
   if you need them to agree, you need the pinned container described above.

See `provenance/` for notes on specific regeneration events.
