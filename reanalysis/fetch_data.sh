#!/usr/bin/env sh
# Download the CSAS 2026 Data Challenge CSVs into ../data (repo root /data)
set -e
dir="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$dir"
for f in Competition Competitors Ends Games Stones Teams; do
  curl -sSL -o "$dir/$f.csv" "https://raw.githubusercontent.com/CSAS-Data-Challenge/2026/HEAD/$f.csv"
done
echo "Downloaded to $dir"
