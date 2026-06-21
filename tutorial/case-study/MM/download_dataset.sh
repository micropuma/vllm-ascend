mkdir -p /root/datasets/vision-arena-bench-v0.1

export http_proxy=http://127.0.0.1:10808
export https_proxy=http://127.0.0.1:10808
export HTTP_PROXY=http://127.0.0.1:10808
export HTTPS_PROXY=http://127.0.0.1:10808

hf download lmarena-ai/vision-arena-bench-v0.1 \
  --repo-type dataset \
  --include "data/*.parquet" \
  --local-dir /root/datasets/vision-arena-bench-v0.1