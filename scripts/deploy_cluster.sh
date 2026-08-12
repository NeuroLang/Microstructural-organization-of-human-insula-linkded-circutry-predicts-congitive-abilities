#!/usr/bin/env bash
# Sync this checkout to a cluster and prepare the environment there.
#
#   DEPLOY_HOST=mycluster DEPLOY_ROOT=/work/me/menon_et_al \
#       ./scripts/deploy_cluster.sh                   # sync + uv sync + check
#   DEPLOY_HOST=... DEPLOY_ROOT=... ./scripts/deploy_cluster.sh --no-sync-env
#
# DEPLOY_HOST is anything `ssh` accepts, including a ~/.ssh/config alias with a
# ProxyJump. DEPLOY_SITE names the site config to print commands for; it does
# not have to match the host.
#
# rsync rather than git because this repository may have no remote. Add one and
# this script becomes `git push` + `git pull` on the cluster; nothing else
# changes.
#
# Derivatives are NOT written into the checkout -- they go to the paths in
# src/insula_rtop/pipeline/site/<site>.yaml. A checkout is disposable; anything
# expensive written inside one is a deletion away from being lost.

set -euo pipefail

HOST="${DEPLOY_HOST:-}"
REMOTE="${DEPLOY_ROOT:-}"
SITE="${DEPLOY_SITE:-mysite}"

if [ -z "${HOST}" ] || [ -z "${REMOTE}" ]; then
    echo "set DEPLOY_HOST and DEPLOY_ROOT, e.g." >&2
    echo "  DEPLOY_HOST=mycluster DEPLOY_ROOT=/work/me/menon_et_al $0" >&2
    exit 2
fi
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYNC_ENV=1

for arg in "$@"; do
    case "$arg" in
        --no-sync-env) SYNC_ENV=0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

echo "=== syncing ${LOCAL} -> ${HOST}:${REMOTE} ==="
ssh "${HOST}" "mkdir -p '${REMOTE}'"
rsync -az --delete \
    --exclude '.venv/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    --exclude 'outputs/' \
    --exclude 'data/atlas_cache/' \
    "${LOCAL}/" "${HOST}:${REMOTE}/"

if [ "${SYNC_ENV}" -eq 1 ]; then
    echo "=== creating the environment on ${HOST} ==="
    # uv.lock is not committed, so this resolves dependency versions on the
    # cluster rather than mirroring the local ones.
    ssh "${HOST}" "cd '${REMOTE}' && uv sync"

    echo "=== verifying the install ==="
    ssh "${HOST}" "cd '${REMOTE}' && uv run python -c '
import insula_rtop
from insula_rtop.rtop.mapl import build_mapl_model
from insula_rtop.atlases.glasser import area_index
print(\"insula_rtop\", insula_rtop.__version__, \"OK\")
print(\"HCP-MMP L_AVI index:\", area_index(\"L\", \"AVI\"))
'"
fi

cat <<EOF

Deployed to ${HOST}:${REMOTE}

Next steps (on ${HOST}, from ${REMOTE}):

  # 1. cohort + BIDS view (minutes)
  uv run python -m insula_rtop.pipeline site=${SITE} steps=[cohort,hcp2bids]

  # 2. time ONE subject's MAPL fit before committing the cohort to SLURM
  uv run python -m insula_rtop.pipeline site=${SITE} \\
      steps=[rtop_volume] subjects=["100206"] rtop_volume.skip_existing=false

  # 3. the full cohort, on SLURM, sized from that timing
  uv run python -m insula_rtop.pipeline site=${SITE} slurm.use=true \\
      steps=[rtop_volume,rtop_surface,atlas_labels,analysis,figures]
EOF
