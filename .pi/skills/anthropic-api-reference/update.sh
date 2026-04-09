#!/bin/bash
# Re-scrape Anthropic API docs from native .md endpoints
set -e
DIR="$(cd "$(dirname "$0")/pages" && pwd)"

declare -A PAGES=(
  ["overview.md"]="https://platform.claude.com/docs/en/api/overview.md"
  ["messages--create.md"]="https://platform.claude.com/docs/en/api/messages/create.md"
  ["messages--streaming.md"]="https://platform.claude.com/docs/en/api/messages/streaming.md"
  ["creating-message-batches.md"]="https://platform.claude.com/docs/en/api/creating-message-batches.md"
  ["retrieving-message-batches.md"]="https://platform.claude.com/docs/en/api/retrieving-message-batches.md"
  ["listing-message-batches.md"]="https://platform.claude.com/docs/en/api/listing-message-batches.md"
  ["canceling-message-batches.md"]="https://platform.claude.com/docs/en/api/canceling-message-batches.md"
  ["message-batch-results.md"]="https://platform.claude.com/docs/en/api/message-batch-results.md"
  ["deleting-message-batches.md"]="https://platform.claude.com/docs/en/api/deleting-message-batches.md"
  ["messages-count-tokens.md"]="https://platform.claude.com/docs/en/api/messages-count-tokens.md"
  ["models-list.md"]="https://platform.claude.com/docs/en/api/models-list.md"
  ["models-get.md"]="https://platform.claude.com/docs/en/api/models-get.md"
  ["files-create.md"]="https://platform.claude.com/docs/en/api/files-create.md"
  ["files-list.md"]="https://platform.claude.com/docs/en/api/files-list.md"
  ["files-get.md"]="https://platform.claude.com/docs/en/api/files-get.md"
  ["files-delete.md"]="https://platform.claude.com/docs/en/api/files-delete.md"
  ["files-get-content.md"]="https://platform.claude.com/docs/en/api/files-get-content.md"
  ["errors.md"]="https://platform.claude.com/docs/en/api/errors.md"
  ["rate-limits.md"]="https://platform.claude.com/docs/en/api/rate-limits.md"
  ["service-tiers.md"]="https://platform.claude.com/docs/en/api/service-tiers.md"
  ["versioning.md"]="https://platform.claude.com/docs/en/api/versioning.md"
  ["beta-headers.md"]="https://platform.claude.com/docs/en/api/beta-headers.md"
  ["client-sdks.md"]="https://platform.claude.com/docs/en/api/client-sdks.md"
  ["supported-regions.md"]="https://platform.claude.com/docs/en/api/supported-regions.md"
  ["ip-addresses.md"]="https://platform.claude.com/docs/en/api/ip-addresses.md"
  ["getting-help.md"]="https://platform.claude.com/docs/en/api/getting-help.md"
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
