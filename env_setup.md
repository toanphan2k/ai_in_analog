# WSL/Linux Environment Setup

These steps target Ubuntu 22.04/24.04 on WSL 2 or native Linux. Run the EDA build
commands inside the Linux shell, not PowerShell. Keep the repository and build trees in
the Linux filesystem (for example `~/src`), rather than under `/mnt/c`, for reliable and
faster builds.

## 1. Install System Packages

```bash
sudo apt update
sudo apt install -y \
  build-essential git ca-certificates curl \
  autoconf automake libtool pkg-config m4 flex bison \
  tcsh csh \
  libx11-dev tcl-dev tk-dev libcairo2-dev libglu1-mesa-dev \
  libncurses-dev libreadline-dev \
  python3 python3-dev python3-pip python3-venv \
  ngspice
```

`ngspice` is the simulator used by the project. Magic is installed from source below;
its Tcl/Tk, X11, Cairo, and GLU development libraries are included above.

## 2. Build Magic

Open PDKs uses Magic's technology files and setup during PDK installation, so install
Magic before building the PDK.

```bash
mkdir -p ~/src
cd ~/src
git clone https://github.com/RTimothyEdwards/magic.git
cd magic
./configure
make -j"$(nproc)"
sudo make install
```

Check that the executable is on `PATH`:

```bash
command -v magic
```

## 3. Build and Install SKY130A

Open PDKs downloads the SKY130 source repositories during configuration/build. The
following creates only the `sky130A` variant and installs to the default
`/usr/local/share/pdk` path expected by this project's `netlists/strongarm/netlist.spice`.

```bash
cd ~/src
git clone https://github.com/fossi-foundation/open_pdks.git
cd open_pdks
./configure --enable-sky130-pdk --with-sky130-variants=A
make -j"$(nproc)"
sudo make install
```

Set `PDK_ROOT` for tools that use the environment variable:

```bash
grep -qxF 'export PDK_ROOT=/usr/local/share/pdk' ~/.bashrc || \
  echo 'export PDK_ROOT=/usr/local/share/pdk' >> ~/.bashrc
export PDK_ROOT=/usr/local/share/pdk
```

Confirm the model library used by the StrongARM netlist exists:

```bash
test -r "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice" && \
  echo "SKY130A ngspice models found under $PDK_ROOT"
```

If you install Open PDKs with a non-default prefix, update the `.lib` path in
`netlists/strongarm/netlist.spice` as well; SPICE does not expand `PDK_ROOT` inside that
absolute path.

## 4. Set Up the Project Python Environment

From the repository root in WSL/Linux:

```bash
cd ~/src/ai_in_analog
bash setup_env.sh
source .venv/bin/activate
```

The setup script installs Python requirements, including Stable-Baselines3, Gymnasium,
and PyTorch for PPO/SAC. If your organization provides a pip mirror, pass its
pip-compatible simple index URL, for example:

```bash
PIP_INDEX_URL=https://<artifactory-host>/artifactory/api/pypi/<repository>/simple \
  bash setup_env.sh
```

## 5. Check the Toolchain

```bash
ngspice --version
magic --version
printf 'PDK_ROOT=%s\n' "$PDK_ROOT"
test -r "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice"
```

Then run a small real simulation from the repository root:

```bash
python3 optimize_sizing.py \
  --runner ngspice \
  --netlist netlists/strongarm/ \
  --algorithm PPO \
  --samples 2 \
  --population-size 4 \
  --rl-timesteps 64 \
  --episode-length 4 \
  --seed 0
```

For off-policy SAC, replace `--algorithm PPO` with `--algorithm SAC`. To exercise the
pipeline without ngspice, use `--runner mock`; those metrics are synthetic and do not
validate the PDK or simulator installation.
