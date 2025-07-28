# Cluster JupyterHub

Opinionated guide to run notebooks on a cluster that hosts a JupyterHub, e.g. JeanZay.

## Connect to JupyterHub

See the [official guide](http://www.idris.fr/eng/jean-zay/pre-post/jean-zay-jupyterhub-eng.html) for more details.

Port forwarding:

```bash
ssh -N -D 9090 <proxy>
```

Chrome with Proxy (for mac):

```
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome  \
    --user-data-dir="$HOME/proxy-profile" \
    --proxy-server="socks5://localhost:9090"
```

-> [https://jupyterhub.idris.fr/](https://jupyterhub.idris.fr/)

## Create a Custom Kernel

See the [user kernel guide](https://idris-cnrs.gitlab.io/jupyter/jupyter-documentation/jupyterlab/jupyterlab#user-kernels), for more details.

Contrary to the guide I propose to base the kernel on the `.venv` managed by `uv` instead of using conda.

In order to install your project's kernel, simply run:

```bash
uv run -m ipykernel install --user --name=xrl --display-name "🔬 XRL"
```

The kernel will be installed in `.local/share/jupyter/kernels`, which you can see using:

```bash
uv run jupyter kernelspec list
```

To test your kernel run:

```bash
uv run jupyter console --kernel=xrl # --debug
```

As mentionned in the guide, kernels should be installed to `.ipython/kernels`. For that, modify the [`create-kernel.sh`](./create-kernel.sh) script to suite your project and run it:

```bash
source notebooks/create-kernel.sh
```

Finally remove the kernel when done (and the `.venv` if you want to save inodes):

```
uv run jupyter kernelspec uninstall xrl
```
