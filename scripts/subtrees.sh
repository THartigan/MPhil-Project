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

Named Git remotes are optional. When an expected remote is absent, the script
uses the component's GitHub repository directly. Override those defaults with
REPORT_SOURCE_REMOTE_URL, LION_REMOTE_URL, or PADIS_REMOTE_URL.
EOF
}

component_config() {
    case "$1" in
        report-source)
            prefix="report-source"
            remote_name="report-source"
            remote_url="${REPORT_SOURCE_REMOTE_URL:-git@github.com:THartigan/mphil-dis-report.git}"
            branch="main"
            ;;
        LION)
            prefix="LION"
            remote_name="lion"
            remote_url="${LION_REMOTE_URL:-git@github.com:THartigan/LION.git}"
            branch="feature/PaDIS_Implementation"
            ;;
        PaDIS-Lion-Compat)
            prefix="PaDIS-Lion-Compat"
            remote_name="padis-lion-compat"
            remote_url="${PADIS_REMOTE_URL:-git@github.com:THartigan/PaDIS.git}"
            branch="main"
            ;;
        *)
            printf 'Unknown component: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
}

resolve_repository() {
    if repository_url="$(git remote get-url "$remote_name" 2>/dev/null)"; then
        repository="$remote_name"
    else
        repository="$remote_url"
        repository_url="$remote_url"
    fi
}

require_clean_tree() {
    if [[ -n "$(git status --porcelain)" ]]; then
        printf 'The working tree must be clean before %s.\n' "$action" >&2
        exit 1
    fi
}

run_for_component() {
    component_config "$1"
    resolve_repository

    if [[ ! -d "$prefix" ]]; then
        printf 'Subtree directory is missing: %s\n' "$prefix" >&2
        exit 1
    fi

    case "$action" in
        status)
            printf '%-20s source=%s branch=%s\n' "$prefix" "$repository_url" "$branch"
            git log -1 --oneline -- "$prefix"
            ;;
        fetch)
            git fetch "$repository" "$branch"
            ;;
        pull)
            git subtree pull --prefix="$prefix" "$repository" "$branch"
            ;;
        push)
            git subtree push --prefix="$prefix" "$repository" "$branch"
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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(git -C "$script_dir/.." rev-parse --show-toplevel)"
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
