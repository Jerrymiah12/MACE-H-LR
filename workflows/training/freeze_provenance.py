"""Freeze the dataset provenance of an `mgo_lr` workspace into `provenance/`.

The 150 GB workspace lives outside the repo and is not archival.  What the
repo needs to keep, so that a checkpoint trained months from now can still be
tied to the data it saw, is small: the splits, the resolved configuration that
produced them, the workspace metadata (LR definition, DFT settings, artefact
hashes), the validation and locality verdicts, and a checksum over all of it.

    python -m workflows.training.freeze_provenance --workspace DIR   # write provenance/
    python -m workflows.training.freeze_provenance --verify          # check SHA256SUMS

`--verify` needs no workspace: it re-hashes the committed files and compares
them to the manifest, which is what CI and a GPU box should run.

The expected counts below are asserted on every freeze.  They are the frozen
dataset's identity -- if a re-`organize` moves them, the freeze fails loudly
rather than silently publishing a different dataset under the same name.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys


from workflows.training import paths

MANIFEST = "SHA256SUMS"

#: the frozen dataset's identity, asserted on every freeze
EXPECTED_COUNTS = {
    "main": {"train": 330, "validation": 37, "test": 37},
    "main_validated": 404,
    "main_rejected": 0,
    "large_test": 44,
    "large_validated": 44,
    "large_rejected": 0,
    "pilot": 20,
    "pilot_validated": 20,
    "pilot_rejected": 0,
}

#: (source path relative to the workspace, name in provenance/)
FILES = [
    ("splits.json", "splits.json"),
    ("metadata.yaml", "metadata.yaml"),
    ("generation_logs/validation_main.json", "validation_main.json"),
    ("generation_logs/validation_pilot.json", "validation_pilot.json"),
    ("generation_logs/validation_large.json", "validation_large.json"),
    ("generation_logs/locality/locality_main.json", "locality_main.json"),
    ("generation_logs/locality/locality_pilot.json", "locality_pilot.json"),
    ("generation_logs/locality/locality_large.json", "locality_large.json"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_resolved_config(workspace):
    """The newest `organize` config snapshot -- the run that made the splits."""
    logs = os.path.join(workspace, "generation_logs")
    candidates = sorted(f for f in os.listdir(logs)
                        if f.startswith("config-organize-")
                        and f.endswith(".yaml"))
    if not candidates:
        raise SystemExit(f"{logs}: no config-organize-*.yaml to freeze")
    return os.path.join(logs, candidates[-1])


def check_counts(workspace):
    """Assert the workspace still holds exactly the frozen dataset."""
    with open(os.path.join(workspace, "splits.json")) as f:
        splits = json.load(f)
    problems = []

    for key, want in EXPECTED_COUNTS["main"].items():
        got = len(splits["main"][key])
        if got != want:
            problems.append(f"main/{key}: {got} snapshots, expected {want}")
    for key, subset in (("large_test", "large_test"), ("pilot", "pilot")):
        got = len(splits[subset])
        want = EXPECTED_COUNTS[key]
        if got != want:
            problems.append(f"{subset}: {got} snapshots, expected {want}")

    for subset in ("main", "pilot", "large"):
        report = os.path.join(workspace, "generation_logs",
                              f"validation_{subset}.json")
        with open(report) as f:
            counts = json.load(f)["counts"]
        for kind in ("validated", "rejected"):
            want = EXPECTED_COUNTS[f"{subset}_{kind}"]
            got = int(counts[kind])
            if got != want:
                problems.append(
                    f"validation_{subset}.json {kind}: {got}, expected {want}")

    # the splits must partition the validated main set with nothing shared
    main = splits["main"]
    total = sum(len(main[k]) for k in ("train", "validation", "test"))
    if total != EXPECTED_COUNTS["main_validated"]:
        problems.append(f"main splits hold {total} snapshots, expected "
                        f"{EXPECTED_COUNTS['main_validated']}")
    for a, b in (("train", "validation"), ("train", "test"),
                 ("validation", "test")):
        shared = set(main[a]) & set(main[b])
        if shared:
            problems.append(f"main {a}/{b} overlap: {sorted(shared)}")

    if problems:
        raise SystemExit("the workspace is not the frozen dataset:\n  "
                         + "\n  ".join(problems))
    print(f"counts OK: main {EXPECTED_COUNTS['main']['train']}/"
          f"{EXPECTED_COUNTS['main']['validation']}/"
          f"{EXPECTED_COUNTS['main']['test']}, "
          f"large {EXPECTED_COUNTS['large_test']}, "
          f"pilot {EXPECTED_COUNTS['pilot']}, rejected 0 across all three sets")


def freeze(workspace):
    os.makedirs(paths.PROVENANCE_DIR, exist_ok=True)
    check_counts(workspace)

    written = []
    for src_rel, dst_name in FILES:
        src = os.path.join(workspace, src_rel)
        if not os.path.isfile(src):
            raise SystemExit(f"{src}: missing, cannot freeze provenance")
        shutil.copy2(src, os.path.join(paths.PROVENANCE_DIR, dst_name))
        written.append(dst_name)

    resolved = latest_resolved_config(workspace)
    shutil.copy2(resolved,
                 os.path.join(paths.PROVENANCE_DIR, "config.resolved.yaml"))
    written.append("config.resolved.yaml")
    print(f"resolved config from {os.path.basename(resolved)}")

    lines = [f"{sha256(os.path.join(paths.PROVENANCE_DIR, n))}  {n}\n"
             for n in sorted(written)]
    with open(os.path.join(paths.PROVENANCE_DIR, MANIFEST), "w") as f:
        f.writelines(lines)
    print(f"{paths.PROVENANCE_DIR}: {len(written)} files + {MANIFEST}")
    return 0


def verify():
    manifest = os.path.join(paths.PROVENANCE_DIR, MANIFEST)
    if not os.path.isfile(manifest):
        raise SystemExit(f"{manifest}: missing; run --workspace DIR first")
    bad = []
    with open(manifest) as f:
        entries = [line.split(None, 1) for line in f if line.strip()]
    for digest, name in entries:
        name = name.strip()
        path = os.path.join(paths.PROVENANCE_DIR, name)
        if not os.path.isfile(path):
            bad.append(f"{name}: missing")
        elif sha256(path) != digest:
            bad.append(f"{name}: checksum mismatch")
    listed = {n.strip() for _, n in entries}
    for name in sorted(os.listdir(paths.PROVENANCE_DIR)):
        if name not in listed and name not in (MANIFEST, "README.md"):
            bad.append(f"{name}: present but not in {MANIFEST}")
    if bad:
        raise SystemExit("provenance verification FAILED:\n  "
                         + "\n  ".join(bad))
    print(f"provenance OK: {len(entries)} files match {MANIFEST}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default=None,
                        help="mgo_lr run workspace to freeze from")
    parser.add_argument("--verify", action="store_true",
                        help="re-hash provenance/ against SHA256SUMS and exit")
    args = parser.parse_args()
    if args.verify:
        return verify()
    return freeze(paths.resolve("workspace", args.workspace))


if __name__ == "__main__":
    sys.exit(main())
