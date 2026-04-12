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
SEED = int(os.environ.get("SEED", "42"))
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
    gpu=modal.gpu.H100(count=N_GPUS),
    timeout=3600,
    volumes={"/data": volume},
)
def train(seed: int = 42):
    import subprocess
    import sys
    import os

    # Clone repo on the submission branch
    subprocess.run([
        "git", "clone", "--branch", BRANCH,
        "https://github.com/Bharath-970/parameter-golf.git",
        "/repo"
    ], check=True)

    os.chdir("/repo")

    # Download SP4096 data only if not already cached in the volume
    sp4096_dir = "/data/datasets/fineweb10B_sp4096"
    if not os.path.exists(sp4096_dir) or len(os.listdir(sp4096_dir)) < 3:
        print("SP4096 data not cached — downloading...")
        subprocess.run([
            sys.executable, "data/cached_challenge_fineweb.py",
            "--variant", "sp4096", "--train-shards", "5"
        ], check=True, env={**os.environ, "HOME": "/root"})
    else:
        print(f"SP4096 data already cached ({len(os.listdir(sp4096_dir))} files), skipping download")
        os.makedirs("data/datasets", exist_ok=True)
        os.makedirs("data/tokenizers", exist_ok=True)
        if not os.path.exists("data/datasets/fineweb10B_sp4096"):
            os.symlink(sp4096_dir, "data/datasets/fineweb10B_sp4096")
        tok_src = "/data/tokenizers/fineweb_4096_bpe.model"
        tok_dst = "data/tokenizers/fineweb_4096_bpe.model"
        if os.path.exists(tok_src) and not os.path.exists(tok_dst):
            os.symlink(tok_src, tok_dst)

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
def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "check":
        print("Checking cached data (CPU only, ~$0.02)...")
        check_data.remote()
    else:
        seed = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 42
        print(f"Launching seed={seed} on {N_GPUS}×H100, branch={BRANCH}")
        returncode = train.remote(seed=seed)
        print(f"Training finished with return code: {returncode}")
