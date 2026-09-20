#!/usr/bin/env bash
# Fetch model weights through the hf-mirror endpoint using plain curl.
# The huggingface_hub python client is unusable here because its temporary-file
# cleanup trips the sandbox's safe-delete guard.
#
# We load the encoders with plain `transformers` (CLS pooling + L2 norm) rather
# than sentence-transformers, so only the transformer weights and the tokenizer
# are needed -- no sentence-transformers meta files.
set -u
export MSYS_NO_PATHCONV=1

ROOT="E:/paper/emotional-rag-repro/models"
MIRROR="https://hf-mirror.com"

fetch () {  # fetch <repo> <file> <dst_subdir>
  local repo="$1" file="$2" sub="$3"
  local dst="$ROOT/$sub"
  mkdir -p "$dst"
  local out="$dst/$(basename "$file")"
  if [ -s "$out" ]; then
    echo "[skip] $sub/$(basename "$file") $(du -h "$out" | cut -f1)"
    return 0
  fi
  echo "[get ] $repo/$file"
  curl -sL --retry 3 --retry-delay 2 --max-time 1200 \
       -o "$out" "$MIRROR/$repo/resolve/main/$file"
  if [ $? -ne 0 ] || [ ! -s "$out" ]; then
    echo "[fail] $repo/$file"
    rm -f "$out"
    return 1
  fi
  echo "[ok  ] $sub/$(basename "$file") $(du -h "$out" | cut -f1)"
}

# ---------------- bge-base-zh-v1.5 ----------------
for f in config.json special_tokens_map.json tokenizer.json tokenizer_config.json \
         vocab.txt pytorch_model.bin ; do
  fetch "BAAI/bge-base-zh-v1.5" "$f" "bge-base-zh-v1.5"
done

# ---------------- bge-base-en-v1.5 ----------------
for f in config.json special_tokens_map.json tokenizer.json tokenizer_config.json \
         vocab.txt pytorch_model.bin ; do
  fetch "BAAI/bge-base-en-v1.5" "$f" "bge-base-en-v1.5"
done

# ---------------- Qwen2.5-1.5B-Instruct ----------------
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         vocab.json merges.txt model.safetensors ; do
  fetch "Qwen/Qwen2.5-1.5B-Instruct" "$f" "Qwen2.5-1.5B-Instruct"
done

echo "ALLDONE"
