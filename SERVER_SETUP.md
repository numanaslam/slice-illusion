# Exact Server Setup — VCF3D on **Windows** (PowerShell, PyTorch + MONAI, CUDA GPU)

Run every command in **PowerShell** (not cmd). Right-click PowerShell → *Run as
Administrator* only where noted. Windows 10/11 or Windows Server 2019/2022.

> Prefer Linux tooling? A **WSL2** path is at the bottom (often smoother for DL). The steps
> below are native Windows.

## 0. Check the GPU and driver first

```powershell
nvidia-smi        # confirm the GPU is listed + note "CUDA Version: XX.X" (top-right)
```

The **driver** CUDA version must be **>=** the wheel you install in step 3 (cu121 needs
driver CUDA >= 12.1). You do **not** need to install the CUDA Toolkit separately — the pip
torch wheel bundles the CUDA runtime. If `nvidia-smi` isn't found, install the NVIDIA driver
first and reboot.

## 1. Enable long paths (medical datasets have deep folder trees)

Run once, **as Administrator**, then reboot:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
git config --system core.longpaths true    # if using git
```

## 2. Install Miniforge (conda) and get the code

Download + install Miniforge (per-user, no admin needed):

```powershell
curl.exe -L -o Miniforge3.exe https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe
Start-Process .\Miniforge3.exe -Wait      # accept defaults; "Add to PATH" optional
# open a NEW PowerShell so conda is on PATH, then:
conda init powershell
```

Put the project on the box (copy the `vcf3d` folder via RDP clipboard/share, or clone):

```powershell
cd $HOME\vcf3d
```

## 3. Create the environment + install CUDA-matched PyTorch

```powershell
conda env create -f environment.yml
conda activate vcf3d

# Install torch matched to your driver (cu121 is standard). If environment.yml's
# torch already matches your driver, skip this line.
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
# Older driver? use cu118:
# pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
```

### venv alternative (no conda)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## 3a. Activating the environment in a new terminal (READ THIS if `conda` is "not recognized")

Every new PowerShell should just need:
```powershell
conda activate vcf3d
```
The prompt then shows `(vcf3d)`. If instead you see **`conda : The term 'conda' is not recognized`** and/or **`running scripts is disabled on this system`**, PowerShell's *execution policy* is blocking conda's activation script. Fix it once (no admin needed), answer `Y`:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Then load conda into this session and activate (adjust the path to wherever Miniforge
was installed — default `$HOME\miniforge3`, or `C:\Users\Public\Miniforge3` if you
installed to a space-free path because the Windows username contains a space):
```powershell
(& "C:\Users\Public\Miniforge3\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate vcf3d
```
Make it permanent so future terminals only need `conda activate vcf3d`:
```powershell
conda init powershell     # then open a NEW terminal
```

**If group policy blocks `-Scope CurrentUser`** (error: "overridden by a policy defined
at a more specific scope"), set it for just the current session instead:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
then re-run the hook + `conda activate vcf3d`.

**Bulletproof fallback — no activation needed.** If the policy stays locked, call the
environment's Python by full path; it uses the exact same packages without activating:
```powershell
& C:\Users\Public\Miniforge3\envs\vcf3d\python.exe -m pytest -q tests\test_smoke.py
```
i.e. replace `python` with `& C:\Users\Public\Miniforge3\envs\vcf3d\python.exe` in any
command (`python -m scripts.s04_run_faithfulness ...`, etc.).

## 4. Verify (must pass before proceeding)

```powershell
python -c "import torch, monai; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('monai', monai.__version__)"
```

Expect `cuda True` + your GPU name. If `cuda False`, the wheel doesn't match the driver —
reinstall torch with the correct index URL from step 3.

## 5. Smoke test the pipeline (no dataset needed)

```powershell
pytest -q tests\test_smoke.py
```

Green = code + GPU wiring OK. (Windows note: the config defaults `data.num_workers` to 0 to
avoid multiprocessing-spawn issues; keep it 0 unless you know you need more.)

## 6. Datasets — download to `data\raw\`

`curl.exe` and `tar` ship with Windows 10/11. Fastest no-DUA start is the MSD BraTS tarball:

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
cd data\raw
curl.exe -L -o Task01_BrainTumour.tar https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
tar -xf Task01_BrainTumour.tar
cd ..\..
```

| Dataset | How to get it on Windows |
|---|---|
| **BraTS** (flagship) | MSD mirror above (easiest). Full BraTS: register at https://www.synapse.org/brats then `pip install synapseclient; synapse get -r <synID>` |
| **LIDC-IDRI** (CT) | Install the **NBIA Data Retriever** Windows app from https://www.cancerimagingarchive.net/collection/lidc-idri/ and load the manifest |
| **AMOS22** | `curl.exe -L -o amos22.zip https://zenodo.org/records/7262581/files/amos22.zip; Expand-Archive amos22.zip` |
| **TotalSegmentator** (organ labels → tissue bank) | `curl.exe -L -o totalseg.zip https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip; Expand-Archive totalseg.zip` |
| **CT-RATE** (optional FM) | `pip install huggingface_hub; huggingface-cli download ibrahimhamamci/CT-RATE --repo-type dataset` |

## 7. Configure + preprocess + run

```powershell
# edit configs\default.yaml: data.root, data.dataset, model.*
python -m scripts.s01_preprocess        --config configs\default.yaml
python -m scripts.s03_train_cavs        --config configs\default.yaml   # Piece 1
python -m scripts.s04_run_faithfulness  --config configs\default.yaml   # Piece 2 + slice illusion
```

Results land in `experiments\results\`.

## 8. Long runs that survive RDP disconnects

Windows has no `tmux`. Two reliable options:

- **Disconnect, don't sign out.** Closing the RDP window (or *Disconnect*) keeps your
  processes running; *Sign out* kills them. Reconnect later to the same session.
- **Detached process with a log file:**

  ```powershell
  Start-Process -NoNewWindow python `
    -ArgumentList "-m","scripts.s04_run_faithfulness","--config","configs\default.yaml" `
    -RedirectStandardOutput "experiments\results\run.log" `
    -RedirectStandardError  "experiments\results\run.err"
  Get-Content experiments\results\run.log -Wait     # tail the log
  ```

  For jobs that must outlive sign-out, wrap the command with **Task Scheduler**
  (`Register-ScheduledTask`) or NSSM.

## Backbone note (why Windows is fine here)
`vcf3d\models\backbone.py` loads a native PyTorch `.pt` checkpoint (`model.source: checkpoint`).
No ONNX round-trip — the reason PyTorch removed the MATLAB friction. Drop a 3D foundation-model
checkpoint at `data\processed\backbone.pt` and set `model.bottleneck` in the config.

---

## Alternative: WSL2 (Ubuntu inside Windows) — often smoother for DL

```powershell
wsl --install -d Ubuntu     # as Administrator, then reboot; set up a Linux user
```

Inside the Ubuntu shell, GPU passthrough works with a recent NVIDIA Windows driver
(no driver install inside WSL). Then follow the **Linux** flow: miniforge, `conda env create`,
`pip install torch ... --index-url .../cu121`, `pytest`. `num_workers>0` and long paths are
non-issues under WSL2. Datasets: put them under the Linux home (`~/vcf3d/data`), not
`/mnt/c/...`, for I/O speed.
