#!/bin/bash
# Build the Zenodo reproducibility archive of the paper's data.
#
#   bash experiments/mi/paper_gecco2027/zenodo/build_archive.sh [OUT]
#
# Writes to OUT (default <first two levels of the repo path>/zenodo-blade-mi):
#   blade-mi-results.zip               README, all run logs, labels, tables
#   blade-mi-activations-exp01.zip     residual-stream activations (probing)
#   blade-mi-activations-exp03-05.zip  SAE codes, corpus/transfer/Bragg acts
#   SHA256SUMS
# Paths inside the zips are relative to the repository root, so unzipping
# all three in the root restores experiments/mi/*/data as the scripts
# expect. Absolute paths of the machine that ran the experiments are
# replaced by <BLADE_ROOT> / <HOME> in text files (anonymisation).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)        # BLADE repository root
MI=experiments/mi
HOMEDIR=$(echo "$ROOT" | cut -d/ -f1-3)       # e.g. /local/<name>
NAME=$(basename "$HOMEDIR")
OUT=${1:-$HOMEDIR/zenodo-blade-mi}
STAGE=$OUT/stage
mkdir -p "$OUT"
rm -rf "$STAGE" && mkdir -p "$STAGE"
cd "$ROOT"

# Activation arrays go to the activation zips; per-query activation files of
# exp01 (mi_activations/*.npz) are omitted because processed/*/
# activations.npz stacks exactly the same arrays (02_build_dataset.py).
# Smoke campaigns are pipeline checks only.
echo "$(date -u) staging text data"
rsync -a --prune-empty-dirs \
  --exclude='*smoke*' --exclude='__pycache__' \
  --exclude='mi_activations/*.npz' \
  --exclude='activations.npz' --exclude='acts.npz' \
  --exclude='k100/shard-*.npz' \
  --include='*/' --include="$MI/*/data/**" --exclude='*' \
  ./ "$STAGE/"
# result files that are too large for git
for f in "$MI"/exp05_steering/results/llamea-confirm/solutions.csv \
  "$MI"/exp05_steering/results/llamea-combo/solutions.csv; do
  mkdir -p "$STAGE/$(dirname "$f")" && cp "$f" "$STAGE/$f"
done
echo "$(date -u) anonymising paths"
grep -rlIZ -e "$ROOT" -e "$HOMEDIR" "$STAGE" |
  xargs -0 -r sed -i -e "s#$ROOT#<BLADE_ROOT>#g" -e "s#$HOMEDIR#<HOME>#g"
if grep -rqI -e "$NAME" -e "$USER" -e "$(hostname -s)" "$STAGE"; then
  echo "identifying strings left:" >&2
  grep -rlI -e "$NAME" -e "$USER" -e "$(hostname -s)" "$STAGE" | head >&2
  exit 1
fi
cp "$HERE/README_zenodo.md" "$STAGE/README.md"

act_exp01() {
  find "$MI"/exp01_probing/data/processed -name activations.npz ! -path '*smoke*'
}
act_rest() {
  find "$MI"/exp03_concept_features/data/processed/k100 -name 'shard-*.npz'
  find "$MI"/exp05_steering/data/processed -name acts.npz ! -path '*smoke*'
  find "$MI"/exp05_steering/data/raw/prescreen-bragg -path '*mi_activations/*.npz'
}
{
  echo "# blade-mi-results.zip"
  (cd "$STAGE" && find . -type f ! -name FILES.txt -printf '%s\t%P\n' | sort -k2)
  echo "# blade-mi-activations-exp01.zip"
  act_exp01 | xargs stat -c '%s	%n' | sort -k2
  echo "# blade-mi-activations-exp03-05.zip"
  act_rest | xargs stat -c '%s	%n' | sort -k2
} > "$STAGE/FILES.txt"

echo "$(date -u) zipping"
rm -f "$OUT"/*.zip
(cd "$STAGE" && zip -q -r -9 "$OUT/blade-mi-results.zip" .)
act_exp01 | zip -q -0 "$OUT/blade-mi-activations-exp01.zip" -@
act_rest | zip -q -0 "$OUT/blade-mi-activations-exp03-05.zip" -@
(cd "$OUT" && sha256sum *.zip > SHA256SUMS)
rm -rf "$STAGE"
ls -l "$OUT"
echo "$(date -u) done"
