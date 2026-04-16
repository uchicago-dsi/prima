#!/bin/bash
# Set /gpfs/data/karczmar-lab/CAPS/MRI1.0/ to:
# - Group: cri-karczmar_lab_phi
# - Readable by group
# - Writable only by owner

set -e
ROOT="/gpfs/data/karczmar-lab/CAPS/MRI1.0"
GROUP="cri-karczmar_lab_phi"

echo "Setting group ownership to ${GROUP}..."
chgrp -R "${GROUP}" "${ROOT}"

echo "Setting permissions: dirs=2750 (rwxr-x---), files=640 (rw-r-----)..."
find "${ROOT}" -type d -exec chmod 2750 {} \;
find "${ROOT}" -type f -exec chmod 640 {} \;

echo "Done. Verifying top-level..."
ls -la "${ROOT}" | head -15
