# === PREFLIGHT + MIRROR =======================================================
# Paste into Colab right after the environment cell. Run preflight() BEFORE any
# generation cell. Then use mirror() / checkpoint() instead of bare open().
#
# Why this exists (narrative.md §18d): a whole session's work was lost because
# the HF token was read-only and nobody found out until the upload at the end.
# The first save attempt of the session was also the last chance to save
# anything. This module makes the token failure happen in second 1, not hour 3.
#
# Design rules:
#   1. preflight() RAISES. It never prints a warning and lets you continue.
#   2. It does a real round-trip (upload -> read back -> delete), not a role
#      string check. Role strings have been wrong before.
#   3. mirror() pushes immediately, with retries. Call it the moment a file
#      exists, not at the end of the session.
# ==============================================================================
import os, io, json, time, glob, hashlib

CORPUS_REPO = "mild-rgb/bert_cot_em"

# Colab secret names, in probe order. Case-sensitive.
_TOKEN_NAMES = ("hf_write_token", "HF_WRITE_TOKEN", "HF_TOKEN_WRITE",
                "HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN")

_STATE = {"token": None, "name": None, "user": None, "repo": CORPUS_REPO}
TOKEN = None   # set by preflight()


def _secret(name):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


def preflight(repo=CORPUS_REPO, need_openrouter=True):
    """Verify we can actually WRITE to HF before spending any GPU time.

    Raises RuntimeError on any failure. Returns the verified token on success.
    """
    from huggingface_hub import HfApi

    print("=== preflight ===", flush=True)

    # ---- 1. find a token ------------------------------------------------------
    token = name = None
    for n in _TOKEN_NAMES:
        v = _secret(n)
        if v:
            token, name = v, n
            break
    if not token:
        raise RuntimeError(
            "No HF token secret found. Tried: " + ", ".join(_TOKEN_NAMES) +
            "\nAdd a WRITE-scoped token in Colab -> Secrets. Names are "
            "case-sensitive.")
    print(f"  token secret : {name}", flush=True)

    # ---- 2. identity and claimed role ----------------------------------------
    api = HfApi(token=token)
    who = api.whoami()
    role = who.get("auth", {}).get("accessToken", {}).get("role", "?")
    print(f"  user         : {who['name']}", flush=True)
    print(f"  claimed role : {role}", flush=True)
    if role == "read":
        raise RuntimeError(
            f"Secret {name!r} is a READ-ONLY token. This is the exact §18d "
            "failure. Replace it with a write-scoped token before generating "
            "anything.")

    # ---- 3. real round-trip: upload, read back, delete ------------------------
    # A claimed role is not proof. Repo-level permission can still deny writes.
    probe_path = "_preflight/write_probe.txt"
    stamp = hashlib.sha256(repr(time.time()).encode()).hexdigest()[:16]
    try:
        api.upload_file(
            path_or_fileobj=io.BytesIO(stamp.encode()),
            path_in_repo=probe_path, repo_id=repo, repo_type="dataset",
            token=token, commit_message="preflight write probe")
    except Exception as e:
        raise RuntimeError(
            f"Write probe FAILED on {repo}. Token {name!r} cannot write here.\n"
            f"  {type(e).__name__}: {e}\n"
            "Do not start generating. Fix the token first.") from e

    from huggingface_hub import hf_hub_download
    got = open(hf_hub_download(repo_id=repo, filename=probe_path,
                               repo_type="dataset", token=token,
                               force_download=True)).read().strip()
    if got != stamp:
        raise RuntimeError(f"Write probe round-trip mismatch: {got!r} != {stamp!r}")

    api.delete_file(path_in_repo=probe_path, repo_id=repo, repo_type="dataset",
                    token=token, commit_message="preflight probe cleanup")
    print(f"  write probe  : OK (round-tripped to {repo})", flush=True)

    # ---- 4. judge key --------------------------------------------------------
    if need_openrouter:
        if _secret("OPENROUTER_API_KEY"):
            print("  OPENROUTER   : present", flush=True)
        else:
            print("  OPENROUTER   : MISSING (uppercase 'OPENROUTER_API_KEY'). "
                  "Generation can proceed; the judge pass cannot.", flush=True)

    # ---- 5. GPU sanity -------------------------------------------------------
    try:
        import torch
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU          : {torch.cuda.get_device_name(0)} | {vram:.0f} GB",
              flush=True)
        if vram <= 70:
            print("  !! <=70 GB: the engine cell will silently fall back to "
                  "4-bit. The absolute 63% gate is only valid in bf16.", flush=True)
    except Exception as e:
        print(f"  GPU          : could not query ({e})", flush=True)

    _STATE.update(token=token, name=name, user=who["name"], repo=repo)

    # Export as a module/notebook global. Later cells (the HF pushes, the
    # restore step) expect a bare TOKEN, and after a kernel restart the only
    # thing that recreates it is re-running preflight().
    global TOKEN
    TOKEN = token

    print("=== preflight PASSED — safe to generate ===", flush=True)
    return token


def _token():
    if not _STATE["token"]:
        raise RuntimeError("preflight() has not been run (or did not pass). "
                           "Run it before mirroring anything.")
    return _STATE["token"]


def mirror(path, subdir="results", retries=4, repo=None):
    """Push one file to HF right now. Call this the moment the file exists.

    §18d lesson 2: the raw generation files were mirrored; the assembled file
    every result was computed from never was.
    """
    from huggingface_hub import HfApi
    repo = repo or _STATE["repo"]
    api = HfApi(token=_token())
    dest = f"{subdir}/{os.path.basename(path)}"
    size = os.path.getsize(path) / 1e6
    for attempt in range(1, retries + 1):
        try:
            api.upload_file(path_or_fileobj=path, path_in_repo=dest,
                            repo_id=repo, repo_type="dataset", token=_token(),
                            commit_message=f"mirror {os.path.basename(path)}")
            print(f"  mirrored {os.path.basename(path)} ({size:.1f} MB) -> "
                  f"{repo}:{dest}", flush=True)
            return dest
        except Exception as e:
            wait = 2 ** attempt
            print(f"  mirror attempt {attempt}/{retries} failed "
                  f"({type(e).__name__}: {e}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(
        f"Could not mirror {path} after {retries} attempts. STOP and fix this "
        "before generating more — unmirrored work is work you are about to lose.")


def checkpoint(rows, name, subdir="results", mirror_every=True):
    """Write rows to <name>.jsonl locally AND push to HF in one call.

    Drop-in replacement for the notebook's save(): use it inside the cell-32
    chunk loop so an interrupt or a reclaimed runtime costs one chunk, not
    everything.
    """
    path = name if name.endswith(".jsonl") else f"{name}.jsonl"
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {path} ({len(rows)} rows)", flush=True)
    if mirror_every:
        mirror(path, subdir=subdir)
    return path


def unmirrored(subdir="results", repo=None, min_mb=0.5):
    """List local .jsonl/.npy/.json files that are NOT on HF yet.

    Run before the session ends, and any time you step away. Anything it lists
    is what you lose if the runtime is reclaimed.
    """
    from huggingface_hub import HfApi
    repo = repo or _STATE["repo"]
    api = HfApi(token=_token())
    remote = {os.path.basename(f) for f in api.list_repo_files(
        repo_id=repo, repo_type="dataset")}
    local = sorted(set(glob.glob("*.jsonl") + glob.glob("*.json") +
                       glob.glob("*.npy")))
    missing = [f for f in local
               if os.path.basename(f) not in remote
               and os.path.getsize(f) / 1e6 >= min_mb]
    if not missing:
        print("everything on disk is mirrored ✓", flush=True)
    else:
        print(f"!! {len(missing)} local file(s) NOT on {repo}:", flush=True)
        for f in missing:
            print(f"   {os.path.getsize(f)/1e6:8.1f} MB  {f}", flush=True)
        print("   -> mirror() each of these NOW.", flush=True)
    return missing
