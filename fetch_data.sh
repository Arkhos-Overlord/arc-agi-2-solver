#!/usr/bin/env bash
# Fetch the public ARC-AGI-2 dataset from the official source.
#
# The data is deliberately NOT committed to this repo: it is 1,120 JSON task
# files, already public at github.com/arcprize/ARC-AGI-2, and regenerable in
# under a minute. This script is the tracked, reproducible way to get it.
set -euo pipefail

REPO="https://github.com/arcprize/ARC-AGI-2.git"
DEST="$(cd "$(dirname "$0")" && pwd)"
TMP="${TMPDIR:-/tmp}/arcagi2-fetch"

mkdir -p "$DEST/data"

if [ -d "$DEST/data/training" ] && [ -d "$DEST/data/evaluation" ]; then
  echo "data already present:"
  echo "  training   : $(ls "$DEST/data/training"/*.json 2>/dev/null | wc -l)"
  echo "  evaluation : $(ls "$DEST/data/evaluation"/*.json 2>/dev/null | wc -l)"
  echo "delete data/ to force a re-fetch."
  exit 0
fi

echo "cloning official ARC-AGI-2 (public, no Kaggle access needed)..."
rm -rf "$TMP"
git clone --depth 1 "$REPO" "$TMP"

for split in training evaluation; do
  n=$(ls "$TMP/data/$split"/*.json 2>/dev/null | wc -l)
  echo "  $split: $n tasks"
  cp -r "$TMP/data/$split" "$DEST/data/"
done

[ -f "$TMP/readme.md" ] && cp "$TMP/readme.md" "$DEST/ARC-AGI-2-readme.md"
rm -rf "$TMP"

echo
echo "done:"
echo "  training   : $(ls "$DEST/data/training"/*.json 2>/dev/null | wc -l)"
echo "  evaluation : $(ls "$DEST/data/evaluation"/*.json 2>/dev/null | wc -l)"
