#!/usr/bin/env python3
"""
FML-bench unified setup.

Idempotent: clones task repositories, downloads datasets, and creates conda
environments needed to run agents on FML-bench.

Usage:
    python setup.py                              # set up everything
    python setup.py --task Causality_causalml  # set up a single task
    python setup.py --list                       # list available tasks
    python setup.py --skip-envs                  # skip conda env creation
    python setup.py --skip-data                  # skip dataset downloads
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE = PROJECT_ROOT / "workspace"
ML_TASKS = PROJECT_ROOT / "ml_tasks"


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

def info(msg: str) -> None:
    print(f"[setup] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd, cwd=None, check=True, env=None, capture=False):
    """Run a shell command. cmd may be a list or string."""
    if isinstance(cmd, str):
        printable = cmd
        shell = True
    else:
        printable = " ".join(shlex.quote(str(c)) for c in cmd)
        shell = False
    info(f"$ {printable}" + (f"  (cwd={cwd})" if cwd else ""))
    return subprocess.run(
        cmd, cwd=cwd, check=check, env=env, shell=shell,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )


def have_command(name: str) -> bool:
    return shutil.which(name) is not None


def conda_env_exists(name: str) -> bool:
    res = run(["conda", "env", "list"], capture=True, check=True)
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.split()[0] == name:
            return True
    return False


def conda_run(env: str, script: str, cwd=None) -> None:
    """Run a Python snippet inside a conda env."""
    run(["conda", "run", "--no-capture-output", "-n", env, "python", "-c", script], cwd=cwd)


def conda_create(name: str, python_version: str, pip_steps: list) -> None:
    """Create a conda env and run pip/conda install steps inside it.

    Each step in pip_steps is a string; it is executed with `bash -lc` after
    activating the env, so it can use `pip`, `conda install`, etc.
    """
    if conda_env_exists(name):
        info(f"conda env '{name}' already exists, skipping")
        return
    info(f"creating conda env '{name}' (python={python_version})")
    run(["conda", "create", "-n", name, f"python={python_version}", "-y"])
    for step in pip_steps:
        full = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {name} && {step}"
        run(["bash", "-lc", full])


# ---------------------------------------------------------------------------
# Repo / dataset helpers
# ---------------------------------------------------------------------------

def clone_repo(parent_dir: Path, repo_name: str, url: str, commit: str,
               setup_files: list | None = None, sparse: str | None = None) -> Path:
    """Clone <url>@<commit> into parent_dir/repo_name, copy setup files, commit them.

    setup_files: list of (src_relative_to_PROJECT_ROOT, dst_relative_to_repo) pairs.
    sparse: if set, do a sparse-checkout including only this subdirectory.
    """
    parent_dir.mkdir(parents=True, exist_ok=True)
    repo_path = parent_dir / repo_name
    if not (repo_path / ".git").exists():
        info(f"cloning {url} -> {repo_path.relative_to(PROJECT_ROOT)}")
        if sparse:
            run(["git", "clone", "--filter=blob:none", "--sparse", url, str(repo_path)])
            run(["git", "sparse-checkout", "set", sparse], cwd=repo_path)
            run(["git", "checkout", commit], cwd=repo_path)
        else:
            run(["git", "clone", url, str(repo_path)])
            run(["git", "checkout", commit], cwd=repo_path)
    else:
        info(f"{repo_path.relative_to(PROJECT_ROOT)} already exists, skipping clone")

    if setup_files:
        for src_rel, dst_rel in setup_files:
            src = PROJECT_ROOT / src_rel
            dst = repo_path / dst_rel
            if not src.exists():
                die(f"setup file not found: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        run(["git", "add", "."], cwd=repo_path, check=False)
        diff = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=repo_path)
        if diff.returncode != 0:
            run(["git", "commit", "-m", "setup for agent benchmark"], cwd=repo_path, check=False)

    return repo_path


def git_commit_data(repo_path: Path, paths: list, message: str) -> None:
    """git add -f <paths> and commit if there is anything to commit."""
    cmd = ["git", "add", "-f", *paths]
    run(cmd, cwd=repo_path, check=False)
    diff = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=repo_path)
    if diff.returncode != 0:
        run(["git", "commit", "-m", message], cwd=repo_path, check=False)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
# Each task entry: a function `setup_<name>(args)` that does everything for it.
# The TASKS registry maps task name -> setup function.

def setup_generalization_domainbed(args):
    repo = clone_repo(WORKSPACE / "Generalization_domainbed", "DomainBed",
                      "https://github.com/facebookresearch/DomainBed.git",
                      "b93c22a1cfc3b2428398272c1a116c8de1f4139e")
    if args.skip_data:
        return
    template = WORKSPACE / "Generalization_domainbed"
    if not (template / "data" / "MNIST" / "raw").exists():
        info("downloading MNIST for domainbed")
        conda_run("domainbed", f"""
import torchvision
torchvision.datasets.MNIST(root=r'{template/"data"}', train=True,  download=True)
torchvision.datasets.MNIST(root=r'{template/"data"}', train=False, download=True)
""")


def setup_data_efficiency_easyfsl(args):
    repo = clone_repo(WORKSPACE / "Data_Efficiency_easyfsl", "easy-few-shot-learning",
                      "https://github.com/sicara/easy-few-shot-learning.git",
                      "8023ff49a02a68830c10a21b8eb908cb33bdf1b9")
    if args.skip_data:
        return
    if not (repo / "data" / "mini_imagenet" / "images").exists():
        info("downloading Mini-ImageNet")
        d = repo / "data" / "mini_imagenet"
        d.mkdir(parents=True, exist_ok=True)
        run(["curl", "-L", "-o", "miniimagenet.zip",
             "https://www.kaggle.com/api/v1/datasets/download/arjunashok33/miniimagenet"], cwd=d)
        run(["unzip", "miniimagenet.zip", "-d", "images"], cwd=d)
        (d / "miniimagenet.zip").unlink(missing_ok=True)
    model = repo / "data" / "models" / "feat_resnet12_mini_imagenet.pth"
    embeds = repo / "data" / "mini_imagenet" / "feat_resnet12"
    if not model.exists() or not embeds.exists():
        info("downloading ResNet12 model and embeddings for easyfsl")
        (repo / "data" / "models").mkdir(parents=True, exist_ok=True)
        run(["bash", "-lc",
             "source $(conda info --base)/etc/profile.d/conda.sh && conda activate easyfsl && "
             f"gdown --id 1ixqw1l9XVxl3lh1m5VXkctw6JssahGbQ -O {shlex.quote(str(model))} && "
             "python -m scripts.predict_embeddings feat_resnet12 "
             "data/models/feat_resnet12_mini_imagenet.pth mini_imagenet "
             "--device=cuda --num-workers=12 --batch-size=1024"], cwd=repo)


def setup_representation_learning_lightly(args):
    repo = clone_repo(WORKSPACE / "Representation_Learning_lightly", "lightly",
                      "https://github.com/lightly-ai/lightly.git",
                      "3d371ee3699e2b6d20adc4c79ac2a0fee52009ac",
                      setup_files=[
                          ("ml_tasks/Representation_Learning_lightly/model.py", "model.py"),
                          ("ml_tasks/Representation_Learning_lightly/transform.py", "transform.py"),
                          ("ml_tasks/Representation_Learning_lightly/split_config.json", "split_config.json"),
                      ])
    if args.skip_data:
        return
    cifar = repo / "datasets" / "cifar10" / "cifar-10-batches-py"
    if not cifar.exists():
        info("downloading CIFAR-10 for lightly")
        target = repo / "datasets" / "cifar10"
        conda_run("lightly", f"""
import torchvision
torchvision.datasets.CIFAR10(root=r'{target}', train=True,  download=True)
torchvision.datasets.CIFAR10(root=r'{target}', train=False, download=True)
""")
        git_commit_data(repo, ["datasets/"], "add cifar-10 dataset for agent benchmark")


def setup_continual_learning(args):
    repo = clone_repo(WORKSPACE / "Continual_Learning_continual_learning", "continual-learning",
                      "https://github.com/GMvandeVen/continual-learning.git",
                      "7cb0ef5a85c928c3cbae3f876f71640251f9dc79")
    if args.skip_data:
        return
    raw = repo / "store" / "datasets" / "MNIST" / "MNIST" / "raw"
    if not raw.exists():
        info("downloading MNIST for continual_learning")
        target = repo / "store" / "datasets" / "MNIST"
        conda_run("continual_learning", f"""
import torchvision
torchvision.datasets.MNIST(root=r'{target}', train=True,  download=True)
torchvision.datasets.MNIST(root=r'{target}', train=False, download=True)
""")
        git_commit_data(repo, ["store/datasets/"], "add mnist dataset for agent benchmark")


def setup_causality_causalml(args):
    clone_repo(WORKSPACE / "Causality_causalml", "causalml",
               "https://github.com/uber/causalml.git",
               "1a96e01f67496c3846d0e8146e5dd90ae9eb21a6",
               setup_files=[("ml_tasks/Causality_causalml/split_config.json", "split_config.json")])


def setup_robustness_art(args):
    clone_repo(WORKSPACE / "Robustness_and_Reliability_art", "adversarial-robustness-toolbox",
               "https://github.com/Trusted-AI/adversarial-robustness-toolbox.git",
               "5ddd8ef204a8352d19d4bd212a4de5d4b7ca6fa9",
               setup_files=[
                   ("ml_tasks/Robustness_and_Reliability_art/model.py", "model.py"),
                   ("ml_tasks/Robustness_and_Reliability_art/trainer.py", "trainer.py"),
                   ("ml_tasks/Robustness_and_Reliability_art/split_config.json", "split_config.json"),
               ])


def setup_privacy_privacymeter(args):
    repo = clone_repo(WORKSPACE / "Privacy_privacymeter", "ml_privacy_meter",
                      "https://github.com/privacytrustlab/ml_privacy_meter.git",
                      "e384af8fd9319b8eeb1303aa82474df1441e3c59")
    if args.skip_data:
        return
    pkl = repo / "data" / "cifar10.pkl"
    if not pkl.exists():
        info("downloading CIFAR-10 + generating pkl files for privacymeter")
        conda_run("privacy_meter", f"""
import os, pickle, torchvision
from torchvision import transforms
path = r'{repo / "data" / "cifar10"}'
os.makedirs(os.path.dirname(path), exist_ok=True)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
all_data = torchvision.datasets.CIFAR10(root=path, train=True,  download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
with open(path + '.pkl', 'wb') as f:
    pickle.dump(all_data, f)
with open(path + '_population.pkl', 'wb') as f:
    pickle.dump(test_data, f)
""")
        git_commit_data(repo, ["data/"], "add cifar-10 dataset and pkl files for agent benchmark")


def setup_fairness_aif360(args):
    repo = clone_repo(WORKSPACE / "Fairness_and_Bias_aif360", "AIF360",
                      "https://github.com/Trusted-AI/AIF360.git",
                      "cd7e2138b7919e0796db7e7902bf49b20065f4f8",
                      setup_files=[("ml_tasks/Fairness_and_Bias_aif360/algorithm.py", "algorithm.py")])
    if args.skip_data:
        return
    raw = repo / "aif360" / "data" / "raw"
    if not (raw / "adult" / "adult.data").exists():
        info("downloading adult dataset")
        (raw / "adult").mkdir(parents=True, exist_ok=True)
        run(["wget", "-q", "https://archive.ics.uci.edu/static/public/2/adult.zip"], cwd=raw / "adult")
        run(["unzip", "-o", "adult.zip"], cwd=raw / "adult")
        (raw / "adult" / "adult.zip").unlink(missing_ok=True)
    if not (raw / "compas" / "compas-scores-two-years.csv").exists():
        info("downloading compas dataset")
        (raw / "compas").mkdir(parents=True, exist_ok=True)
        run(["wget", "-q", "-O", str(raw / "compas" / "compas-scores-two-years.csv"),
             "https://raw.githubusercontent.com/propublica/compas-analysis/refs/heads/master/compas-scores-two-years.csv"])
    if not (raw / "german" / "german.data").exists():
        info("downloading german dataset")
        (raw / "german").mkdir(parents=True, exist_ok=True)
        run(["wget", "-q", "-O", str(raw / "german" / "german.data"),
             "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"])
        run(["wget", "-q", "-O", str(raw / "german" / "german.doc"),
             "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc"])


def setup_generalization_officehome(args):
    clone_repo(WORKSPACE / "Generalization_domainbed_officehome", "DomainBed",
               "https://github.com/facebookresearch/DomainBed.git",
               "b93c22a1cfc3b2428398272c1a116c8de1f4139e")
    if args.skip_data:
        return
    out = WORKSPACE / "Generalization_domainbed_officehome" / "data"
    if not (out / "office_home").exists():
        info("downloading OfficeHome dataset (~2.4GB)")
        out.mkdir(parents=True, exist_ok=True)
        run(["pip", "install", "gdown", "-q"], check=False)
        run(["python", "-c",
             "import gdown; gdown.download(id='1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa', "
             "output='office_home.zip', quiet=False)"], cwd=out)
        run(["unzip", "-q", "office_home.zip"], cwd=out)
        run(["mv", "office_home_dg", "office_home"], cwd=out)
        (out / "office_home.zip").unlink(missing_ok=True)


def setup_fairness_fairlearn(args):
    clone_repo(WORKSPACE / "Fairness_fairlearn", "fairlearn",
               "https://github.com/fairlearn/fairlearn.git", "0173555",
               setup_files=[
                   ("ml_tasks/Fairness_fairlearn/algorithm.py", "algorithm.py"),
                   ("ml_tasks/Fairness_fairlearn/split_config.json", "split_config.json"),
               ])


def setup_causality_gcastle(args):
    parent = WORKSPACE / "Causality_gcastle"
    parent.mkdir(parents=True, exist_ok=True)
    repo = parent / "trustworthyAI"
    if not (repo / ".git").exists():
        info("cloning trustworthyAI (sparse) for gcastle")
        run(["git", "clone", "--filter=blob:none", "--sparse",
             "https://github.com/huawei-noah/trustworthyAI.git", str(repo)])
        run(["git", "sparse-checkout", "set", "gcastle"], cwd=repo)
        run(["git", "checkout", "58abc35"], cwd=repo)
    inner = repo / "gcastle"
    shutil.copy2(PROJECT_ROOT / "ml_tasks/Causality_gcastle/algorithm.py", inner / "algorithm.py")
    if not (inner / ".git").exists():
        run(["git", "init"], cwd=inner)
        run(["git", "add", "-A"], cwd=inner)
        run(["git", "commit", "-m", "setup for agent benchmark"], cwd=inner, check=False)


def setup_continual_learning_pycil(args):
    repo = clone_repo(WORKSPACE / "Continual_Learning_pycil", "PyCIL",
                      "https://github.com/LAMDA-CL/PyCIL.git", "f3509b8",
                      setup_files=[("ml_tasks/Continual_Learning_pycil/algorithm.py", "algorithm.py")])
    if args.skip_data:
        return
    if not (repo / "data" / "cifar-100-python").exists():
        info("downloading CIFAR-100 for pycil")
        conda_run("pycil", f"""
import torchvision
torchvision.datasets.CIFAR100(root=r'{repo/"data"}', train=True,  download=True)
torchvision.datasets.CIFAR100(root=r'{repo/"data"}', train=False, download=True)
""")
        git_commit_data(repo, ["data/"], "add cifar-100 dataset for agent benchmark")


def setup_privacy_opacus(args):
    repo = clone_repo(WORKSPACE / "Privacy_opacus", "opacus",
                      "https://github.com/pytorch/opacus.git", "6dc0a27",
                      setup_files=[
                          ("ml_tasks/Privacy_opacus/algorithm.py", "algorithm.py"),
                          ("ml_tasks/Privacy_opacus/split_config.json", "split_config.json"),
                      ])
    if args.skip_data:
        return
    if not (repo / "data" / "cifar-10-batches-py").exists():
        info("downloading CIFAR-10 for opacus")
        # cd to PROJECT_ROOT to avoid python importing the local opacus/ folder
        conda_run("opacus", f"""
import torchvision
torchvision.datasets.CIFAR10(root=r'{repo/"data"}', train=True,  download=True)
torchvision.datasets.CIFAR10(root=r'{repo/"data"}', train=False, download=True)
""", cwd=PROJECT_ROOT)
        git_commit_data(repo, ["data/"], "add cifar-10 dataset for agent benchmark")


def setup_data_efficiency_usb(args):
    repo = clone_repo(WORKSPACE / "Data_Efficiency_usb", "Semi-supervised-learning",
                      "https://github.com/microsoft/Semi-supervised-learning.git", "1ef4cbe",
                      setup_files=[("ml_tasks/Data_Efficiency_usb/algorithm.py", "algorithm.py")])
    if args.skip_data:
        return
    if not (repo / "data" / "cifar-100-python").exists():
        info("downloading CIFAR-100 for usb")
        conda_run("usb", f"""
import torchvision
torchvision.datasets.CIFAR100(root=r'{repo/"data"}', train=True,  download=True)
torchvision.datasets.CIFAR100(root=r'{repo/"data"}', train=False, download=True)
""")
        git_commit_data(repo, ["data/"], "add cifar-100 dataset for agent benchmark")


def setup_representation_learning_solo(args):
    repo = clone_repo(WORKSPACE / "Representation_Learning_solo_learn", "solo-learn",
                      "https://github.com/vturrisi/solo-learn.git", "b69b4bd")
    shutil.copy2(PROJECT_ROOT / "ml_tasks/Representation_Learning_solo_learn/algorithm.py", repo / "algorithm.py")
    shutil.copy2(PROJECT_ROOT / "ml_tasks/Representation_Learning_solo_learn/split_config.json", repo / "split_config.json")
    git_commit_data(repo, ["algorithm.py", "split_config.json"], "setup for agent benchmark")
    if args.skip_data:
        return
    if not (repo / "data" / "cifar-100-python").exists():
        info("downloading CIFAR-100 for solo-learn")
        conda_run("sololearn", f"""
import torchvision
torchvision.datasets.CIFAR100(root=r'{repo/"data"}', train=True,  download=True)
torchvision.datasets.CIFAR100(root=r'{repo/"data"}', train=False, download=True)
""")
        git_commit_data(repo, ["data/"], "add cifar-100 dataset for agent benchmark")


def setup_robustness_openood(args):
    repo = clone_repo(WORKSPACE / "Robustness_openood", "OpenOOD",
                      "https://github.com/Jingkang50/OpenOOD.git", "3c35632",
                      setup_files=[
                          ("ml_tasks/Robustness_openood/algorithm.py", "algorithm.py"),
                          ("ml_tasks/Robustness_openood/split_config.json", "split_config.json"),
                      ])
    if args.skip_data:
        return
    needs = (not (repo / "data" / "cifar-10-batches-py").exists() or
             not (repo / "data" / "cifar-100-python").exists() or
             not (repo / "data" / "svhn").exists())
    if needs:
        info("downloading CIFAR-10/100 + SVHN for openood")
        conda_run("openood", f"""
import torchvision
torchvision.datasets.CIFAR10(root=r'{repo/"data"}', train=True,  download=True)
torchvision.datasets.CIFAR10(root=r'{repo/"data"}', train=False, download=True)
torchvision.datasets.CIFAR100(root=r'{repo/"data"}', train=False, download=True)
torchvision.datasets.SVHN(root=r'{repo/"data"/"svhn"}', split='test', download=True)
""")
        git_commit_data(repo, ["data/"], "add cifar-10/100 and svhn datasets for agent benchmark")


def setup_federated_learning_pfllib(args):
    repo = clone_repo(WORKSPACE / "Federated_Learning_PFLlib", "PFLlib",
                      "https://github.com/TsingZ0/PFLlib.git", "0169ba7")
    shutil.copy2(PROJECT_ROOT / "ml_tasks/Federated_Learning_PFLlib/algorithm.py", repo / "algorithm.py")
    shutil.copy2(PROJECT_ROOT / "ml_tasks/Federated_Learning_PFLlib/split_config.json", repo / "split_config.json")
    git_commit_data(repo, ["algorithm.py", "split_config.json", "dataset/"], "setup for agent benchmark")
    if args.skip_data:
        return
    if not (repo / "dataset" / "Cifar10").exists():
        info("generating CIFAR-10 federated dataset for PFLlib")
        run(["bash", "-lc",
             "source $(conda info --base)/etc/profile.d/conda.sh && conda activate pfllib && "
             "python generate_Cifar10.py noniid - dir"], cwd=repo / "dataset")
        git_commit_data(repo, ["dataset/"], "add cifar10 federated dataset")


def setup_unlearning(args):
    clone_repo(WORKSPACE / "Unlearning_open_unlearning", "open-unlearning",
               "https://github.com/locuslab/open-unlearning.git", "4ad738a",
               setup_files=[("ml_tasks/Unlearning_open_unlearning/algorithm.py", "algorithm.py")])


# ---------------------------------------------------------------------------
# Conda environments
# ---------------------------------------------------------------------------

def setup_conda_env_fmlbench():
    """Harness env used by run_agent_benchmark.py."""
    conda_create("fmlbench", "3.11", [
        "pip install aider-chat==0.85.1",
        "pip install anthropic==0.84.0 backoff==2.2.1 openai==1.91.0 google-generativeai==0.8.5 "
        "matplotlib==3.10.8 pypdf==6.9.0 pymupdf4llm==1.27.2.1 torch==2.10.0 numpy==1.26.4 "
        "transformers==5.3.0 datasets==4.7.0 tiktoken==0.9.0 wandb==0.25.1 tqdm==4.67.1",
        "pip install gpytorch==1.15.2 umap-learn==0.5.11 scikit-learn==1.8.0",
        "pip install jupyterlab",
        "conda install -c conda-forge nodejs -y",
    ])


CONDA_ENVS = {
    "fmlbench": setup_conda_env_fmlbench,
    "domainbed": lambda: conda_create("domainbed", "3.10", [
        "conda install mkl==2023.1.0 mkl-service==2.4.0 pytorch==1.12.1 torchvision==0.13.1 "
        "torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y",
        "pip install mkl-fft==1.3.11 mkl-random==1.2.8 backpack-for-pytorch==1.3.0 numpy==1.22.4 "
        "wilds==2.0.0 tqdm==4.66.4 imageio==2.9.0 gdown==3.13.0 parameterized==0.9.0 "
        "Pillow==10.3.0 timm==0.9.16",
        "pip install weco==0.3.0",
        'pip install "setuptools<81"',
    ]),
    "easyfsl": lambda: conda_create("easyfsl", "3.10", [
        'pip install "matplotlib>=3.0.0" "pandas>=1.5.0" "torch==2.9.1" "torchvision==0.24.1" '
        '"tqdm>=4.1.0" tensorboard',
        "pip install gdown typer loguru pyarrow fastparquet",
        "pip install weco==0.3.0",
        'pip install "setuptools<81"',
    ]),
    "lightly": lambda: conda_create("lightly", "3.10", [
        "conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 "
        "-c pytorch -y",
        'conda install -c conda-forge torchmetrics==0.11.4 "numpy<2" lightning==2.1.2 scipy '
        "einops tqdm scikit-learn hydra-core -y",
        "pip install timm==1.0.25",
        "pip install aenum",
        "pip install weco==0.3.0",
        'pip install "setuptools<81"',
    ]),
    "continual_learning": lambda: conda_create("continual_learning", "3.10.4", [
        "pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 "
        "--index-url https://download.pytorch.org/whl/cu113",
        'pip install matplotlib "numpy<2" scipy pandas tqdm scikit-learn',
        "pip install weco==0.3.0",
        'pip install "setuptools<81"',
    ]),
    "causalml": lambda: conda_create("causalml", "3.10", [
        'pip install "cython<3" "scikit-learn>=1.6.0,<1.9.2" "scipy>=1.8,<1.9.2" "packaging>=21.3" '
        '"typing-extensions<4.6.0" "platformdirs>=2" "tomli>=1.1.0" "contourpy>=1.0.1" '
        '"cycler>=0.10" "fonttools>=4.22.0" "kiwisolver>=1.3.1" "pillow>=8" "python-dateutil>=2.7" '
        '"numpy>=1.18.5,<1.25.0" "pytz>=2020.1" "tzdata>=2022.7" "llvmlite>=0.44.0,<0.45"',
        'pip install "optree<0.16.0" pandas matplotlib seaborn xgboost==3.2.0 lightgbm==4.6.0 '
        'sphinx sphinx_rtd_theme "sphinxcontrib-bibtex<2.0.0" nbsphinx statsmodels shap pathos',
        "pip install tensorflow==2.10.0",
        "pip install weco==0.3.0",
        'pip install "setuptools<81"',
    ]),
    "art": lambda: conda_create("art", "3.10", [
        "pip install tensorflow-gpu==2.10.1",
        "pip install numpy==1.23.5",
        "pip install matplotlib tqdm",
        'pip install "scipy>=1.4.1" scikit-learn==1.7.2',
        "pip install weco==0.3.0",
        'pip install "setuptools<81"',
    ]),
    "privacy_meter": lambda: conda_create("privacy_meter", "3.12", [
        f"pip install -r {WORKSPACE/'Privacy_privacymeter/ml_privacy_meter/requirements.txt'}",
        "pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 "
        "--extra-index-url https://download.pytorch.org/whl/cu118",
        "pip install numpy==1.26.3 scipy==1.14.1 scikit-learn==1.5.2 opacus==1.5.2 "
        "matplotlib==3.9.2",
        "pip install weco==0.3.0",
    ]),
    "aif360": lambda: conda_create("aif360", "3.10", [
        'conda install "numpy<2" scipy pandas scikit-learn==1.1.3 "matplotlib==3.8.4" tqdm seaborn -y',
        'pip install lightgbm==3.1.1 "igraph[plotting]==0.9.8"',
        "pip install lime adversarial-robustness-toolbox==1.13.0 BlackBoxAuditing "
        "tensorflow-gpu==2.10.1 cvxpy==1.7.5 fairlearn==0.9.0 skorch==0.11.0 inFairness==0.2.3 "
        "pot==0.9 mlxtend colorama",
        'pip install "pytest>=3.5.0" "pytest-cov>=2.8.1"',
        "pip install weco==0.3.0",
        "conda install -c pytorch torchvision==0.13.1 torchaudio==0.12.1 --no-deps -y",
        "pip install torch==2.10.0",
        'pip install "setuptools<81"',
    ]),
    "fairlearn": lambda: conda_create("fairlearn", "3.10", [
        "pip install fairlearn==0.13.0 scikit-learn==1.7.2 pandas numpy==2.2.6 scipy==1.15.3",
        'pip install "setuptools<81"',
    ]),
    "gcastle": lambda: conda_create("gcastle", "3.9", [
        "pip install gcastle==1.0.4 numpy==2.0.2 scipy==1.13.1 scikit-learn==1.6.1 tqdm==4.67.3",
        "pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu",
    ]),
    "pycil": lambda: conda_create("pycil", "3.10", [
        "pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 "
        "--index-url https://download.pytorch.org/whl/cu121",
        "pip install tqdm==4.67.3 numpy==2.2.6 scipy==1.15.3 quadprog==0.1.13 POT==0.9.6.post1 "
        "scikit-learn==1.7.2 pillow==12.1.1",
        'pip install "setuptools<81"',
    ]),
    "opacus": lambda: conda_create("opacus", "3.10", [
        "pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 "
        "--index-url https://download.pytorch.org/whl/cu121",
        "pip install opacus==1.5.4 numpy==2.2.6 scipy==1.15.3",
        'pip install "setuptools<81"',
    ]),
    "usb": lambda: conda_create("usb", "3.10", [
        "pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 "
        "--index-url https://download.pytorch.org/whl/cu121",
        "pip install numpy==2.2.6 pillow==12.1.1",
        'pip install "setuptools<81"',
    ]),
    "sololearn": lambda: conda_create("sololearn", "3.10", [
        "pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 "
        "--index-url https://download.pytorch.org/whl/cu121",
        'pip install "numpy==1.26.4" pillow==12.1.1',
        'pip install "setuptools<81"',
    ]),
    "openood": lambda: conda_create("openood", "3.9", [
        "pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 "
        "--index-url https://download.pytorch.org/whl/cu121",
        "pip install scikit-learn==1.6.1 numpy==2.0.2 scipy==1.13.1",
    ]),
    "pfllib": lambda: conda_create("pfllib", "3.11", [
        "pip install torch==2.0.1 torchvision==0.15.2",
        "pip install numpy==1.26.4 scipy==1.17.1 scikit-learn==1.8.0 tqdm cvxpy ujson==5.12.0",
        'pip install "setuptools<81"',
    ]),
    "open_unlearning": lambda: conda_create("open_unlearning", "3.11", [
        "pip install torch==2.4.1",
        "pip install transformers==4.51.3 datasets==3.0.1 accelerate==0.34.2 numpy==2.2.3 "
        "scipy==1.14.1 scikit-learn==1.5.2 tqdm==4.67.3 hydra-core==1.3.0 omegaconf==2.3.0",
        'pip install "setuptools<81"',
    ]),
}


# Each task → (workspace setup function, conda envs required for data/exec).
# The runner uses `conda_env` from ml_tasks/<task>/config.json — we mirror that here so
# `--task X` only installs envs strictly needed.
TASKS = {
    "Generalization_domainbed":                  (setup_generalization_domainbed,    ["domainbed"]),
    "Data_Efficiency_easyfsl":                   (setup_data_efficiency_easyfsl,     ["easyfsl"]),
    "Representation_Learning_lightly":           (setup_representation_learning_lightly, ["lightly"]),
    "Continual_Learning_continual_learning":     (setup_continual_learning,          ["continual_learning"]),
    "Causality_causalml":                   (setup_causality_causalml,     ["causalml"]),
    "Robustness_and_Reliability_art": (setup_robustness_art,            ["art"]),
    "Privacy_privacymeter":            (setup_privacy_privacymeter,        ["privacy_meter"]),
    "Fairness_and_Bias_aif360": (setup_fairness_aif360,             ["aif360"]),
    "Generalization_domainbed_officehome":       (setup_generalization_officehome,   ["domainbed"]),
    "Fairness_fairlearn":                        (setup_fairness_fairlearn,          ["fairlearn"]),
    "Causality_gcastle":                         (setup_causality_gcastle,           ["gcastle"]),
    "Continual_Learning_pycil":                  (setup_continual_learning_pycil,    ["pycil"]),
    "Privacy_opacus":                            (setup_privacy_opacus,              ["opacus"]),
    "Data_Efficiency_usb":                       (setup_data_efficiency_usb,         ["usb"]),
    "Representation_Learning_solo_learn":        (setup_representation_learning_solo, ["sololearn"]),
    "Robustness_openood":                        (setup_robustness_openood,          ["openood"]),
    "Federated_Learning_PFLlib":                 (setup_federated_learning_pfllib,   ["pfllib"]),
    "Unlearning_open_unlearning":                (setup_unlearning,                  ["open_unlearning"]),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", action="append", default=[],
                   help="Set up only the given task(s). Repeatable. Default: all tasks.")
    p.add_argument("--list", action="store_true", help="List available tasks and exit.")
    p.add_argument("--skip-envs", action="store_true", help="Skip conda env creation.")
    p.add_argument("--skip-data", action="store_true",
                   help="Skip dataset downloads (clones repos and creates envs only).")
    p.add_argument("--skip-workspaces", action="store_true",
                   help="Skip workspace setup (clones, datasets); useful with --skip-data --skip-envs=false.")
    return p.parse_args()


def check_prereqs():
    missing = [c for c in ("conda", "git", "pip", "wget", "unzip", "curl") if not have_command(c)]
    if missing:
        die(f"missing required commands: {', '.join(missing)}")
    if not ML_TASKS.exists():
        die("ml_tasks/ not found — run from project root.")


def main():
    args = parse_args()

    if args.list:
        print("Available tasks:")
        for name in TASKS:
            print(f"  - {name}")
        return

    check_prereqs()

    selected = args.task or list(TASKS.keys())
    unknown = [t for t in selected if t not in TASKS]
    if unknown:
        die(f"unknown task(s): {unknown}. Use --list to see available tasks.")

    # Conda envs needed = harness env + envs of selected tasks
    envs_needed = ["fmlbench"]
    for t in selected:
        for e in TASKS[t][1]:
            if e not in envs_needed:
                envs_needed.append(e)

    if not args.skip_envs:
        info("=" * 60)
        info("Stage 1/2: Conda environments")
        info("=" * 60)
        for env_name in envs_needed:
            CONDA_ENVS[env_name]()
    else:
        info("Skipping conda env creation (--skip-envs)")

    if not args.skip_workspaces:
        info("=" * 60)
        info("Stage 2/2: Task workspaces" + (" (datasets skipped)" if args.skip_data else ""))
        info("=" * 60)
        for t in selected:
            info(f"--- {t} ---")
            TASKS[t][0](args)
    else:
        info("Skipping workspace setup (--skip-workspaces)")

    info("=" * 60)
    info("Setup complete.")
    info("=" * 60)
    info("Next:")
    info("  conda activate fmlbench")
    info("  python run_agent_benchmark.py "
         "--agent-config configs/agents/ai_scientist_v2.yaml "
         "--task-config configs/tasks/causality_causalml.yaml "
         "--model gpt-5 --provider OpenAI")


if __name__ == "__main__":
    main()
