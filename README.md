<img src="https://raw.githubusercontent.com/Xmaster6y/xrl/refs/heads/main/docs/source/_static/images/xrl-logo.png" alt="logo" width="200"/>

# XRL 🔍

[![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/yp-edu/xrl/blob/main/LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![python versions](https://img.shields.io/badge/python-3.11%20|%203.12-blue)](https://www.python.org/downloads/)
![ci](https://github.com/yp-edu/xrl/actions/workflows/ci.yml/badge.svg)
![publish](https://github.com/Xmaster6y/xrl/actions/workflows/publish.yml/badge.svg)

<a href="https://xrl.readthedocs.io"><img src="https://img.shields.io/badge/-Read%20the%20Docs%20Here-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white"></img></a>

Explainable RL for TorchRL.

## Python Config

Using `uv` to manage python dependencies and run scripts.

## Scripts

This project uses [Just](https://github.com/casey/just) to manage scripts, refer to their instructions for installation.

## Cluster Config

This project is cluster-ready.

- See [`launch`](./launch/) to launch scripts with slurm.
- See [`notebooks`](./notebooks/) to run notebooks on a cluster's JupyterHub.
