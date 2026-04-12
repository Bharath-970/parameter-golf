import modal
import os

app = modal.App("parameter-golf")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "numpy",
        "tiktoken",
        "sentencepiece",
        "huggingface_hub",
        "zstandard",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
)

volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

BRANCH = "sp4096-depth-recur-parallel-resid"
SUBMISSION = "records/track_10min_16mb/2026-04-12_SP4096_DepthRecur_ParallelResid_MuonEqR/train_gpt.py"
N_GPUS = 8


@app.function(
    image=image,
    cpu=4,
    timeout=600,
    volumes={"/data": volume},
)
def check_data():
    """CPU-only check of what's cached in the volume. Costs ~$0.02."""
    import os
    base = "/data/datasets"
    if os.path.exists(base):
        for d in sorted(os.listdir(base)):
            path = os.path.join(base, d)
            files = os.listdir(path) if os.path.isdir(path) else []
            sizes = [os.path.getsize(os.path.join(path, f)) for f in files]
            total_gb = sum(sizes) / 1e9
            print(f"  {d}: {len(files)} files, {total_gb:.1f} GB")
    else:
        print("No datasets dir found")
    tok = "/data/tokenizers"
    if os.path.exists(tok):
        print("tokenizers:", os.listdir(tok))
    else:
        print("No tokenizers dir found")


@app.function(
    image=image,
    cpu=4,
    timeout=3600,
    volumes={"/data": volume},
)
def download_data():
    """Download SP4096 data into the volume once. CPU-only, no GPU cost.
    Points HF cache at /data so downloaded files persist across runs."""
    import subprocess
    import sys
    import os

    subprocess.run([
        "git", "clone", "--branch", BRANCH,
        "https://github.com/Bharath-970/parameter-golf.git",
        "/repo"
    ], check=True)

    os.chdir("/repo")

    # Point HF cache to the volume so it persists
    os.makedirs("/data/hf_cache", exist_ok=True)
    env = {
        **os.environ,
        "HOME": "/root",
        "HF_HOME": "/data/hf_cache",
        "HUGGINGFACE_HUB_CACHE": "/data/hf_cache",
    }

    # SP4096 data is hosted on kevclark/parameter-golf, not the default repo.
    # Must delete cached manifest.json first so the script fetches the right one.
    for stale in ["data/manifest.json", "data/datasets/manifest.json"]:
        if os.path.exists(stale):
            os.remove(stale)

    print("Downloading SP4096 data from kevclark/parameter-golf (takes ~5-10 min)...")
    env["MATCHED_FINEWEB_REPO_ID"] = "kevclark/parameter-golf"
    subprocess.run([
        sys.executable, "data/cached_challenge_fineweb.py",
        "--variant", "sp4096", "--train-shards", "5",
    ], check=True, env=env)

    # Copy downloaded files into volume for direct access
    import shutil
    src_datasets = "/repo/data/datasets/fineweb10B_sp4096"
    src_tokenizer_dir = "/repo/data/tokenizers"
    dst_datasets = "/data/datasets/fineweb10B_sp4096"
    dst_tokenizers = "/data/tokenizers"

    if os.path.exists(src_datasets):
        print(f"Copying {len(os.listdir(src_datasets))} dataset files to volume...")
        os.makedirs(dst_datasets, exist_ok=True)
        for f in os.listdir(src_datasets):
            shutil.copy2(os.path.join(src_datasets, f), os.path.join(dst_datasets, f))

    if os.path.exists(src_tokenizer_dir):
        os.makedirs(dst_tokenizers, exist_ok=True)
        for f in os.listdir(src_tokenizer_dir):
            shutil.copy2(os.path.join(src_tokenizer_dir, f), os.path.join(dst_tokenizers, f))
        print(f"Copied tokenizers: {os.listdir(dst_tokenizers)}")

    volume.commit()
    print("Done — data committed to volume.")


@app.function(
    image=image,
    gpu=f"H100:{N_GPUS}",
    timeout=3600,
    volumes={"/data": volume},
)
def train(seed: int = 42):
    import subprocess
    import sys
    import os

    subprocess.run([
        "git", "clone", "--branch", BRANCH,
        "https://github.com/Bharath-970/parameter-golf.git",
        "/repo"
    ], check=True)

    os.chdir("/repo")

    # Wire up cached data from volume
    sp4096_dir = "/data/datasets/fineweb10B_sp4096"
    tok_src = "/data/tokenizers/fineweb_4096_bpe.model"

    if os.path.exists(sp4096_dir) and len(os.listdir(sp4096_dir)) >= 3:
        print(f"SP4096 data cached ({len(os.listdir(sp4096_dir))} files), symlinking...")
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/tokenizers", exist_ok=True)
        os.symlink(sp4096_dir, "data/datasets/fineweb10B_sp4096")
        if os.path.exists(tok_src):
            os.symlink(tok_src, "data/tokenizers/fineweb_4096_bpe.model")
    else:
        print("SP4096 data not in volume — downloading now (adds ~5-10 min)...")
        for stale in ["data/manifest.json", "data/datasets/manifest.json"]:
            if os.path.exists(stale):
                os.remove(stale)
        subprocess.run([
            sys.executable, "data/cached_challenge_fineweb.py",
            "--variant", "sp4096", "--train-shards", "5",
        ], check=True, env={
            **os.environ,
            "HOME": "/root",
            "MATCHED_FINEWEB_REPO_ID": "kevclark/parameter-golf",
        })

    env = {
        **os.environ,
        "HOME": "/root",
        "SEED": str(seed),
        "DATA_PATH": "./data/datasets/fineweb10B_sp4096",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_4096_bpe.model",
    }

    result = subprocess.run([
        "torchrun", "--standalone", f"--nproc_per_node={N_GPUS}",
        SUBMISSION,
    ], capture_output=False, env=env)

    return result.returncode


@app.local_entrypoint()
def check():
    """Check what data is cached in the volume (~$0.02)."""
    print("Checking cached data...")
    check_data.remote()


@app.local_entrypoint()
def download():
    """Download SP4096 data into the volume. CPU-only, run once before training."""
    print("Downloading SP4096 data into volume (CPU, no GPU cost)...")
    download_data.remote()
    print("Done. Run 'python3 -m modal run run_modal.py' to train.")


@app.local_entrypoint()
def main(seed: int = 42):
    """Run training on 8xH100. Usage: python3 -m modal run run_modal.py --seed 42"""
    print(f"Launching seed={seed} on {N_GPUS}xH100, branch={BRANCH}")
    returncode = train.remote(seed=seed)
    print(f"Training finished with return code: {returncode}")
