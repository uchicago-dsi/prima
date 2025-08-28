# prima
Polygenic Risk and Imaging Multimodal Assessment

# chimec list
/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/List_ChiMEC_priority_2025July30.csv

Micromamba install:
```
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -r ~/micromamba 
source ~/.bashrc
micromamba config append channels conda-forge
```

Installation recipe:
```bash
micromamba config append channels conda-forge
micromamba create -y --name prima python=3.11
micromamba activate prima
git clone git@github.com:uchicago-dsi/prima.git
cd prima
micromamba install -y pytorch torchvision pytorch-cuda=12.1 selenium firefox geckodriver ipykernel -c pytorch -c nvidia
pip install -e . # if developing
# pip install . # if not developing
pip install -r requirements.txt
```





## HIRO → ChiMEC Data Transfer Instructions (macOS client → GPFS)

This document describes the tested steps for moving data from a HIRO CIFS share to our ChiMEC GPFS storage.

Goal: one-way, resumable, safe copy of files. Files are deleted from HIRO after successful transfer; nothing is ever deleted from ChiMEC.

---
### 1. Mount the HIRO share on macOS

- Open Finder → Go → Connect to Server...
- Enter the server string (use your UCHAD username):
  `smb://UCHAD;annawoodard@cifs01uchadccd.uchad.uchospitals.edu/radiology/HIRO/16352A`
- Authenticate with your UCHAD password.
- Finder mounts the share under `/Volumes/16352a` (the exact name may vary if you remount multiple times). This mounted path is the source for rsync.

---
### 2. Install a modern rsync on macOS

The Apple-provided rsync is outdated. Use **Homebrew** to install a modern version.

```bash
brew install rsync
```

This installs modern rsync as `/opt/homebrew/bin/rsync` (Apple Silicon) or `/usr/local/bin/rsync` (Intel).

---
### 3. Set source and destination paths

```bash
SRC="/Volumes/16352a/"  # mounted HIRO subdir; note trailing slash
DST="annawoodard@cri-ksysappdsp3.cri.uchicago.edu:/gpfs/data/huo-lab/Image/ChiMEC/"
```

---
### 4. Dry run (safe preview)

This command performs a dry run, showing what will be copied without actually moving any files.

```bash
/opt/homebrew/bin/rsync -aHvn --itemize-changes --append-verify --partial \
  --remove-source-files \
  "$SRC" "$DST"
```
- `-n`: dry run, nothing is copied or removed
- `--itemize-changes`: shows a detailed list of what would happen
- `--remove-source-files`: is inert in a dry-run but will delete files from HIRO after a successful copy in a real run

Check that the file mapping is correct (`/Volumes/16352a/...` → `/gpfs/.../ChiMEC/...`).

---
### 5. Real transfer (resumable, safe delete-on-success)

This command performs the actual data transfer and can be stopped and restarted as needed.

```bash
LOG="$HOME/hiro2chimec-$(date +%F).log"
caffeinate -dims /opt/homebrew/bin/rsync \
  -aH --info=progress2 --human-readable \
  --append-verify --partial --remove-source-files \
  --log-file="$LOG" \
  "$SRC" "$DST"
```
- `caffeinate`: prevents the Mac from sleeping during the transfer
- `--append-verify`: enables robust resume on partial files and ensures checksum verification
- `--partial`: keeps partially copied files to allow for a restart
- `--remove-source-files`: deletes files from HIRO only after a successful copy to the destination
- `--log-file`: records all transfers to a log file

---
### 6. Cleanup (optional)

Directories on HIRO will remain after the files are moved. They can be pruned manually later if desired, but this is not required.

---
### Performance note

Over VPN, throughput was measured at ~3 MB/s (approximately 4 days per terabyte).

### check how many exams have transferred

```bash
grep '<f++++++++++' hiro2chimec-2025-08-21.log  | awk '{print $5}' | cut -d/ -f1-2 | sort -u | wc -l
```