#!/bin/bash
# Re-scrape Gemini API docs from native .md.txt endpoints
set -e
DIR="$(cd "$(dirname "$0")/pages" && pwd)"

declare -A PAGES=(
  ["generate-content.md"]="https://ai.google.dev/api/generate-content.md.txt"
  ["embeddings.md"]="https://ai.google.dev/api/embeddings.md.txt"
  ["models.md"]="https://ai.google.dev/api/models.md.txt"
  ["tokens.md"]="https://ai.google.dev/api/tokens.md.txt"
  ["caching.md"]="https://ai.google.dev/api/caching.md.txt"
  ["files.md"]="https://ai.google.dev/api/files.md.txt"
  ["live.md"]="https://ai.google.dev/api/live.md.txt"
  ["batch-mode.md"]="https://ai.google.dev/api/batch-mode.md.txt"
  ["interactions-api.md"]="https://ai.google.dev/api/interactions-api.md.txt"
  ["troubleshooting.md"]="https://ai.google.dev/gemini-api/docs/troubleshooting.md.txt"
  ["api-versions.md"]="https://ai.google.dev/gemini-api/docs/api-versions.md.txt"
  ["models-gemini.md"]="https://ai.google.dev/gemini-api/docs/models/gemini.md.txt"
  ["rate-limits.md"]="https://ai.google.dev/gemini-api/docs/rate-limits.md.txt"
  ["tokens-guide.md"]="https://ai.google.dev/gemini-api/docs/tokens.md.txt"
)

for file in "${!PAGES[@]}"; do
  url="${PAGES[$file]}"
  echo -n "  ${file} ... "
  curl -sL "$url" > "$DIR/$file"
  echo "$(wc -l < "$DIR/$file") lines"
  sleep 0.3
done

echo "---"
echo "$(ls "$DIR"/*.md | wc -l) pages, $(du -sh "$DIR" | cut -f1)"
