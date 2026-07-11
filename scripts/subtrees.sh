#!/usr/bin/env bash

set -euo pipefail

components=("report-source" "LION" "PaDIS-Lion-Compat")

usage() {
    cat <<'EOF'
Usage: ./scripts/subtrees.sh <status|fetch|pull|push> [component|all]

Components:
  report-source       mphil-dis-report/main
  LION                LION/feature/PaDIS_Implementation
  PaDIS-Lion-Compat   PaDIS/main

Examples:
  ./scripts/subtrees.sh status
  ./scripts/subtrees.sh fetch all
  ./scripts/subtrees.sh pull LION
  ./scripts/subtrees.sh push report-source

The push action requires one explicit component; it cannot target "all".
EOF
}

component_config() {
    case "$1" in
        report-source)
            prefix="report-source"
            remote="report-source"
            branch="main"
            ;;
        LION)
            prefix="LION"
            remote="lion"
            branch="feature/PaDIS_Implementation"
            ;;
        PaDIS-Lion-Compat)
            prefix="PaDIS-Lion-Compat"
            remote="padis-lion-compat"
            branch="main"
            ;;
        *)
            printf 'Unknown component: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
}

require_clean_tree() {
    if [[ -n "$(git status --porcelain)" ]]; then
        printf 'The working tree must be clean before %s.\n' "$action" >&2
        exit 1
    fi
}

run_for_component() {
    component_config "$1"

    if ! git remote get-url "$remote" >/dev/null 2>&1; then
        printf 'Required Git remote is missing: %s\n' "$remote" >&2
        exit 1
    fi

    case "$action" in
        status)
            printf '%-20s remote=%-20s branch=%s\n' "$prefix" "$remote" "$branch"
            git log -1 --oneline -- "$prefix"
            ;;
        fetch)
            git fetch "$remote" "$branch"
            ;;
        pull)
            git subtree pull --prefix="$prefix" "$remote" "$branch"
            ;;
        push)
            git subtree push --prefix="$prefix" "$remote" "$branch"
            ;;
    esac
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi

action="$1"
target="${2:-all}"

case "$action" in
    status|fetch|pull|push) ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        printf 'Unknown action: %s\n' "$action" >&2
        usage >&2
        exit 2
        ;;
esac

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ "$action" == "pull" || "$action" == "push" ]]; then
    require_clean_tree
fi

if [[ "$action" == "push" && "$target" == "all" ]]; then
    printf 'Push requires one explicit component name.\n' >&2
    exit 2
fi

if [[ "$target" == "all" ]]; then
    for component in "${components[@]}"; do
        run_for_component "$component"
    done
else
    run_for_component "$target"
fi
