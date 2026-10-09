#!/bin/sh
# Fetch the retained wheelhouse for local builds.
# Wheel bytes are retained in Forgejo generic packages
# (JEFF7712/solubility-gnn-wheels, version 1). Every file is verified
# against wheelhouse-manifest.json by sha256 after download.
# CI assembles releases from the same retained supply pinned in the
# homelab application catalog; keep both manifests in sync on dep bumps.
set -eu
FORGEJO="${FORGEJO:-http://10.0.30.20:3000}"
: "${FORGEJO_TOKEN:?set FORGEJO_TOKEN to a Forgejo read token}"
mkdir -p wheelhouse
python3 - "$FORGEJO" <<'EOF'
import hashlib
import json
import os
import sys
import urllib.request

forgejo = sys.argv[1]
token = os.environ["FORGEJO_TOKEN"]
manifest = json.load(open("wheelhouse-manifest.json"))
for entry in manifest:
    name, expected = entry["filename"], entry["sha256"]
    target = os.path.join("wheelhouse", name)
    url = (
        f"{forgejo}/api/packages/JEFF7712/generic/"
        f"solubility-gnn-wheels/1/{name}"
    )
    request = urllib.request.Request(url, headers={"Authorization": f"token {token}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = response.read()
    actual = "sha256:" + hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"digest mismatch for {name}")
    with open(target, "wb") as handle:
        handle.write(data)
    print(f"verified {name}")
EOF
