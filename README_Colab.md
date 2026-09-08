# README_Colab.md — Run Chatterbox TTS Server on Google Colab (T4 GPU)

This guide walks through **`DELIGHTFUL_Chatterbox_simplified.ipynb`**, a Colab notebook that clones
your fork, installs it into an isolated micromamba environment, and runs the server with a single
committed voice and a set of pre-tuned generation defaults.

Unlike the original upstream Colab demo, this notebook:
- Clones **your own fork** (`michtai/chatterbox`) instead of the upstream repo, so all the patches
  in this repo (sentence-splitting fixes, custom `output_filename` support, etc.) are already baked in —
  no runtime patching of `utils.py`/`server.py` is needed.
- Uses a **single committed voice** (`voices/delightful_really_soft.wav`) instead of the stock demo
  voice set, which has been removed from this fork.
- Mounts **Google Drive** and writes all generated audio to `MyDrive/chatterbox/outputs`, so output
  persists across Colab sessions instead of disappearing when the runtime resets.
- Applies pre-tuned **generation defaults** (temperature, exaggeration, CFG weight, seed, chunk size)
  instead of the app's stock defaults.
- **Pulls your latest commits automatically** every time you (re-)run the server, so pushing a fix to
  your fork mid-session doesn't require redoing setup.

You will open the Web UI via Colab's built-in port proxy: the notebook prints a `*.colab.*` URL once
the server is ready.

---

## Before you start: add a `GITHUB_TOKEN` secret

The notebook clones `michtai/chatterbox` using a Colab secret called `GITHUB_TOKEN`. If the fork is
private, this token is required (a GitHub [personal access token](https://github.com/settings/tokens)
with `repo` read access is enough). Add it via the 🔑 **Secrets** tab in the left sidebar of Colab,
name it exactly `GITHUB_TOKEN`, and toggle "Notebook access" on. If the fork is public, you can leave
the secret unset — `userdata.get("GITHUB_TOKEN")` will just resolve to `None` and the clone URL will
still work, since a `None` token embedded in the URL is harmless for a public repo.

---

## Notebook layout: 3 cells + a stop cell

The notebook has exactly three cells that matter, plus one to stop the server:

| # | Cell | Run it... |
|---|------|-----------|
| 1 | Environment + packages (micromamba, PyTorch, Chatterbox) | **Once** per Colab runtime |
| 2 | Drive, clone your fork, server deps, voice + config | **Once** per Colab runtime |
| 3 | Run server | **Every time** you want to (re-)start the server — including after pushing new commits |
| — | Stop server | Whenever you want to free port 8004 |

Cells 1 and 2 install things and don't need to be re-run just to restart the server or pick up code
changes — that's what Cell 3 is for. Cell 3 runs `git fetch` + `git reset --hard origin/HEAD` against
the clone made in Cell 2 before launching, so anything you've pushed to your fork since Cell 2 ran
(or since the last time you ran Cell 3) is picked up automatically. It then re-applies your voice and
`config.yaml` settings — `config.yaml` is a tracked file, so without this the hard reset would wipe
out the local voice/output/generation-defaults it just set.

---

## Important rules (read first)

- **Do not use "Run all."** Cell 3 runs the server in the foreground and keeps running while the
  server is up, so "Run all" will hang at that cell.
- Run cells **in order** the first time. After that, Cell 3 is the only one you'll normally touch.
- Keep the notebook tab open while using the Web UI; if the runtime disconnects, the server stops.
- The first time you mount Google Drive (part of Cell 2), Colab will ask you to authorize access —
  approve it.

---

## Cell-by-cell

### Cell 1 — Environment + packages

Creates the isolated `cb311` micromamba environment (skips creation if it already exists), installs
PyTorch 2.5.1 (cu121) and `chatterbox-tts` from the `devnen/chatterbox-v2` fork, installs
`s3tokenizer`/`onnx` with `--no-deps`, force-upgrades `protobuf` to resolve the conflict between
`descript-audiotools` (pins `protobuf<3.20`) and `onnx` (needs `protobuf>=3.20.2`), then verifies CUDA
is visible and that the installed Turbo build won't demand a Hugging Face auth token.

### Cell 2 — Drive, clone, server deps, voice + config

Defines all the settings the notebook uses (`REPO_OWNER`/`REPO_NAME`, `PORT`, the Drive output
folder, the committed voice filename, chunk size, generation defaults) at the top — edit these if you
want a different voice, chunk size, or defaults. Then:
- Mounts Google Drive at `/content/drive` and creates `MyDrive/chatterbox/outputs`.
- Clones your fork fresh into `/content/chatterbox`.
- Installs `requirements-nvidia.txt` (or a fallback dependency list if that file is missing) and
  force-upgrades `protobuf` again.
- Patches the Chatterbox watermarker to fail open instead of crashing the server with
  `'NoneType' object is not callable` on environments where Perth can't initialize (the same patch
  `start.py` applies automatically outside Colab).
- Defines `apply_voice_and_config()` (confirms the committed voice file exists, copies it into
  `reference_audio/` if needed, and writes `config.yaml`: active model, default voice, output
  directory, generation defaults, and the `ui_state` fields the Web UI reads on load — including
  `last_chunk_size`, which is what the chunk-size slider, and therefore every `/tts` request, actually
  uses; a top-level `chunking` key does **not** exist in the app's schema and is silently ignored) and
  calls it once.

### Cell 3 — Run server

`git fetch` + `git reset --hard origin/HEAD` against the Cell 2 clone, re-runs
`apply_voice_and_config()`, then launches `server.py` in the foreground inside `cb311`. Streams logs
live (collapsing noisy `Sampling: NN%|...` progress bars into compact dot progress), writes the full
log to `/content/chatterbox_server_stdout.log`, and prints the Colab proxy link plus
`/api/model-info` once the server is reachable.

**First run note:** model downloads can take a while; watch the progress output.

### Stop cell — free port 8004

Run this any time you want to stop the server process.

---

## What to run when…

### Start it the first time
Run: **Cell 1 → Cell 2 → Cell 3** (in order, one-by-one).

### Stop the server
Run: **the stop cell**.

### Restart the server, or pick up new commits you pushed to your fork
Run: **the stop cell**, then **Cell 3**. You don't need to re-run Cells 1 or 2.
If you get "address already in use" without stopping first, run the stop cell and then rerun Cell 3.

### Change the voice, chunk size, or generation defaults
Edit the values at the top of **Cell 2**, then re-run **Cell 2** (it's safe to re-run — cloning is a
fresh `rm -rf` + `git clone` each time), then **Cell 3**.

### After changing / updating Python packages
Run: **the stop cell** → **Cell 1** → **Cell 2** → **Cell 3**.

---

## Troubleshooting

### Web UI opens but TTS doesn't work
The server can be reachable even if the model didn't load; always check `/api/model-info` (Cell 3
prints it) to confirm `loaded: True`.

### "Token is required (`token=True`), but no token found"
Something in your installed Turbo code is still requesting `huggingface_hub` to read a local token
(`token=True` semantics).
Fix: re-run **Cell 1** and confirm it doesn't report any auth-forcing markers.

### "/content/bin/micromamba: No such file or directory"
You likely restarted the runtime or Cell 1 didn't complete; rerun **Cell 1**.

### `git clone`/`git fetch` fails with "Repository not found" or an auth error
Your `GITHUB_TOKEN` secret is missing, expired, or doesn't have read access to the fork. Check the
🔑 Secrets tab and make sure "Notebook access" is enabled for it.

### `Expected voices/<file> in the cloned repo but it's missing`
Cell 2 or Cell 3 couldn't find `VOICE_FILENAME` (set at the top of Cell 2) under `voices/` in the
cloned repo. Confirm the voice file is actually committed to your fork's `voices/` directory.

### Where is generated audio saved?
`MyDrive/chatterbox/outputs` in your Google Drive (set by `DRIVE_OUTPUTS_DIR` in Cell 2 and written
into `config.yaml`'s `paths.output`). Each generation writes a new file — if a name collision would
occur (e.g. you send a repeated custom `output_filename`), the server appends `_2`, `_3`, etc. rather
than overwriting the earlier file.

### Where are model files cached?
During Cell 3, downloads are stored under `/content/hf_home` (set by `HF_HOME` in Cell 3).

---

## Notes
- For bug reports, attach `/content/chatterbox_server_stdout.log` and the `/api/model-info` output.
- This notebook and fork are set up for one person's own voice and defaults; if you want the stock
  multi-voice demo experience instead, use the upstream
  [`devnen/Chatterbox-TTS-Server`](https://github.com/devnen/Chatterbox-TTS-Server) notebook instead.
