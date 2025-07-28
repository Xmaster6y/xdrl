export PROJECT_NAME="xdrl"
export DISPLAY_NAME="🔬 XDRL"

mkdir -p $HOME/.ipython/kernels/${PROJECT_NAME}
cp -r ${WORK}/${PROJECT_NAME}/.venv/share/jupyter/kernels/python3/* $HOME/.ipython/kernels/${PROJECT_NAME}/

echo '{
"argv": [
 "'"${WORK}/${PROJECT_NAME}"'/.venv/bin/python3",
 "-Xfrozen_modules=off",
 "-m",
 "ipykernel_launcher",
 "-f",
 "{connection_file}"
],
"display_name": "'"$DISPLAY_NAME"'",
"language": "python",
"metadata": {
"debugger": true
 }
}' > $HOME/.ipython/kernels/${PROJECT_NAME}/kernel.json
