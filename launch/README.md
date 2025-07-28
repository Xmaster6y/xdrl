# Cluster Launch

Opinionated guide to launch scripts on a cluster that uses Slurm and Conda, e.g. JeanZay.

## Setup Git

Init a git repo on the cluster:

```bash
mkdir xdrl
cd xdrl
git init
```

Add a cluster remote branch:

```bash
git remote add jeanzay <remote_url>
```

Dedicate a branch for the cluster:

```bash
git branch main-jz
git push --set-upstream jeanzay main-jz
```

## Setup `uv`

Simply follow [the official guide](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer), e.g. using `curl`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:

```bash
uv sync
```

## Launch Interactive

Get an interactive session:

```bash
salloc --gpus 1 -A nwq@v100
```

Run a job:

```bash
srun uv run --no-sync -m scripts.demo
```

## Launch Batched

Launch a batched job:

```bash
sbatch launch/jz/demo.sh
```

## Sync `wandb` Logs

Use the following command:

```bash
make wandb-sync
```

You'll need to either export `WAND_API_KEY` environment variable or paste it when prompted.


## JeanZay Tricks

### Setup

First JeanZay doesn't allow for direct use of remote vscode server. You'll need to use alternative tools like `rsync` or `sshfs` see [this page](https://code.visualstudio.com/docs/remote/troubleshooting#_using-sshfs-to-access-files-on-your-remote-host).

Then, the nodes do not have an internet connection. You'll need to use the head node to setup the project and download the dependencies and other assets. For `uv` you can use the `--no-sync` flag to skip the sync step when running a script.

### Monitor Disk Usage

In order to monitor disk usage and inode usage, you can use `idr_quota_user` for a user and `idr_quota_project` for a project, cf [this page](http://www.idris.fr/eng/jean-zay/cpu/jean-zay-quota-eng.html).

### Monitor Allocation

In order to monitor your allocation usage, you can use `idracct` to monitor the allocation usage. You can separate per project (which you can get with `idrproj`) and different accounts, cf [this page](http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-doc_account-eng.html). You can also access the fair share ratio with `idr_compuse`.

It is more likely that you'll run out of inodes before you run out of disk space, especially if you're multiplying projects. To track where your inodes are you can use the following script ([cf](https://unix.stackexchange.com/questions/4105/how-do-i-count-all-the-files-recursively-through-directories)):

```bash
find . ! -name . -prune -type d -exec sh -c '
  for dir do
    dir="${dir#./}"
    printf "%s:\t" "$dir"
    find ".//$dir" -type f | grep -c //
  done' sh {} +
```

### Monitor Job

In addition to interactive sessions (`srun`), you can also monitor your job by connecting to your allocated node(s) with `ssh`.

Find your jobs using `squeue`:

```bash
squeue -u <user>
```

Then, you can connect to the node with `ssh` and use various commands like `top`, `htop`, `nvidia-smi`, etc, cf [this page](http://www.idris.fr/eng/jean-zay/jean-zay-connexion_ssh_noeud_calcul-eng.html).

### Additional Resources

- [JeanZay Slurm Partitions](http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm-eng.html)
- [BigScience "How To"](https://github.com/bigscience-workshop/bigscience/blob/master/jz/slurm/README.md)
