"""Record and verify what a graph cache actually contains.

`export-target` is workspace-wide: it repoints every snapshot's
`hamiltonians.h5` at either the SR or the full-H labels and records the choice
in `data/metadata.yaml` as `training_target`. A graph cache built while the
workspace is in one state holds *those* labels forever after, and nothing in
the .pkl says which.

That is fine as long as the cache is never rebuilt -- but if it is deleted, or
`dataset_name` changes, `AijData` silently rebuilds from whatever is exported
at that moment. A cache named `mgo404sr` would then hold full-H labels, both
runs would train on the same target, and the comparison would quietly become
meaningless.

So: freeze the cache right after building it, while the export state is known.

    python -m workflows.training.freeze_cache --target sr     # after building graphs_sr
    python -m workflows.training.freeze_cache --verify        # re-hash both manifests

The freeze refuses if the workspace's `training_target` does not match the
target being frozen -- which is exactly the mistake this guards against.

Hashing 8+ GB takes a couple of minutes.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import sys


from workflows.training import paths

MANIFEST = "CACHE_MANIFEST.json"


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    size = os.path.getsize(path)
    # progress only on a terminal -- \r into a log file is thousands of lines
    show = size > (1 << 30) and sys.stdout.isatty()
    done = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
            done += len(block)
            if show:
                print(f"\r    hashing {100.0 * done / size:5.1f}%",
                      end="", flush=True)
    if show:
        print("\r    hashing 100.0%")
    return h.hexdigest()


def workspace_target(workspace):
    """`training_target` from the workspace metadata, without a yaml import."""
    path = os.path.join(workspace, "metadata.yaml")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            if line.startswith("training_target:"):
                return line.split(":", 1)[1].strip()
    return None


def cache_file(target, training_root, workspace):
    """The .pkl a production config would read, found by its dataset name.

    `save_graph_dir` interpolates `%(training_root)s`, so the resolved values
    have to be substituted in before reading it -- otherwise this looks under
    the config's own default rather than where the caller actually built.
    """
    config = paths.SR_CONFIG if target == "sr" else paths.FULL_CONFIG
    cp = paths.read_config(config)
    cp["DEFAULT"]["training_root"] = training_root
    cp["DEFAULT"]["workspace"] = workspace
    graph_dir = cp.get("data", "save_graph_dir")
    dataset_name = cp.get("data", "dataset_name")
    hits = sorted(glob.glob(os.path.join(graph_dir,
                                         f"HGraph-{dataset_name}-*.pkl")))
    if not hits:
        raise SystemExit(f"no cache for dataset_name={dataset_name} under "
                         f"{graph_dir}; build it first")
    if len(hits) > 1:
        raise SystemExit(f"{graph_dir}: {len(hits)} caches match "
                         f"{dataset_name}; refusing to guess: {hits}")
    return graph_dir, dataset_name, hits[0]


def freeze(target, workspace, training_root):
    graph_dir, dataset_name, path = cache_file(target, training_root,
                                               workspace)
    exported = workspace_target(workspace)
    if exported != target:
        raise SystemExit(
            f"refusing to freeze the '{target}' cache: the workspace is "
            f"exported as '{exported}'. The cache in {graph_dir} may hold "
            f"'{exported}' labels under the name '{dataset_name}'. Re-export "
            f"(`mgo_lr export-target --target {target}`), rebuild the cache, "
            "then freeze.")

    splits_sha = sha256(os.path.join(paths.PROVENANCE_DIR, "splits.json"))
    print(f"  {os.path.basename(path)} "
          f"({os.path.getsize(path) / 2**30:.2f} GiB)")
    manifest = {
        "target": target,
        "dataset_name": dataset_name,
        "file": os.path.basename(path),
        "size_bytes": os.path.getsize(path),
        "sha256": sha256(path),
        "workspace_training_target_at_build": exported,
        "provenance_splits_sha256": splits_sha,
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc)
                              .replace(microsecond=0).isoformat(),
    }
    out = os.path.join(graph_dir, MANIFEST)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(f"  frozen -> {out}")
    return manifest


def verify_one(target, training_root, workspace):
    graph_dir, dataset_name, path = cache_file(target, training_root,
                                               workspace)
    manifest_path = os.path.join(graph_dir, MANIFEST)
    if not os.path.isfile(manifest_path):
        print(f"[{target}] NO MANIFEST at {manifest_path} -- this cache's "
              "label provenance is unknown")
        return False
    with open(manifest_path) as f:
        manifest = json.load(f)
    problems = []
    if manifest.get("file") != os.path.basename(path):
        problems.append(f"file is {os.path.basename(path)}, manifest says "
                        f"{manifest.get('file')}")
    if manifest.get("size_bytes") != os.path.getsize(path):
        problems.append(f"size is {os.path.getsize(path)}, manifest says "
                        f"{manifest.get('size_bytes')}")
    else:
        print(f"[{target}] {os.path.basename(path)} "
              f"({os.path.getsize(path) / 2**30:.2f} GiB), built from export "
              f"'{manifest.get('workspace_training_target_at_build')}' on "
              f"{manifest.get('frozen_utc')}")
        if sha256(path) != manifest.get("sha256"):
            problems.append("sha256 mismatch -- the cache changed since it "
                            "was frozen")
    splits_sha = sha256(os.path.join(paths.PROVENANCE_DIR, "splits.json"))
    if manifest.get("provenance_splits_sha256") != splits_sha:
        problems.append("provenance/splits.json changed since this cache was "
                        "frozen; the cache may not match the current splits")
    if problems:
        print(f"[{target}] FAILED:\n  " + "\n  ".join(problems))
        return False
    print(f"[{target}] OK")
    return True


def main():
    parser = paths.add_path_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--target", choices=("sr", "full"),
                        help="freeze this cache")
    parser.add_argument("--verify", action="store_true",
                        help="verify frozen caches instead of freezing")
    args = parser.parse_args()
    workspace = paths.resolve("workspace", args.workspace)
    training_root = paths.resolve("training_root", args.training_root)

    if args.verify:
        targets = [args.target] if args.target else ["sr", "full"]
        ok = True
        for target in targets:
            try:
                ok &= verify_one(target, training_root, workspace)
            except SystemExit as exc:
                print(f"[{target}] {exc}")
                ok = False
        return 0 if ok else 1

    if not args.target:
        raise SystemExit("give --target sr|full, or --verify")
    freeze(args.target, workspace, training_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
