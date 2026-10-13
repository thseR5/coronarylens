#!/usr/bin/env bash
# Checks to run before pushing the repository anywhere public. Exit 1 if any check fails.
# Usage: tools/prepublish_check.sh [branch]   (default: main)
set -u
cd "$(dirname "$0")/.." || exit 1
branch="${1:-main}"
fail=0
ok() { echo "  ok   $1"; }
bad() { echo "  FAIL $1"; fail=1; }

echo "Pre-publish checks for '$branch'"

[ -z "$(git status --porcelain)" ] && ok "working tree clean" || bad "uncommitted changes (git status)"

grep -q "AUTHOR_NAME_TO_CONFIRM" LICENSE && bad "LICENSE still has the author placeholder" || ok "LICENSE author filled in"

word="$(printf '%s' 'Q2xhdWRl' | base64 --decode)"  # the assistant's product name, kept out of this file's text
n=$(git grep -i -I -l "$word" "$branch" -- . | wc -l | tr -d ' ')
[ "$n" = 0 ] && ok "no tracked file mentions the assistant" || bad "$n tracked files mention the assistant"
n=$(git log "$branch" --format='%B%n%an%n%ae%n%cn%n%ce' | grep -ic "$word")
[ "$n" = 0 ] && ok "no commit message or author mentions the assistant" || bad "$n commit lines mention the assistant"
n=0; for c in $(git rev-list "$branch"); do git grep -qi -I "$word" "$c" -- . && n=$((n + 1)); done
[ "$n" = 0 ] && ok "no historical commit contains the word" || bad "$n historical commits contain the word"

n=$(git ls-tree -r --name-only "$branch" | grep -c -i -E "\.bundle\$|(^|/)${word}\.md\$|(^|/)RESUME\.md\$|(^|/)\.${word}/")
[ "$n" = 0 ] && ok "no bundles or private notes tracked" || bad "$n private files tracked"

big=$(git rev-list --objects "$branch" | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' |
  awk '$1=="blob" && $2>50*1024*1024 {print $3}')
[ -z "$big" ] && ok "no blob over 50 MB in history" || bad "blobs over 50 MB: $big"

uv run python tools/check_numbers.py >/dev/null 2>&1 && ok "docs numbers match artifacts" || bad "tools/check_numbers.py failed"

exit $fail
