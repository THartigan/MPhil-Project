# MPhil Project

This repository brings together the report source and the supporting software
for the MPhil project. Each component is included as a Git subtree with its full
upstream commit history.

## Repository layout

- `report-source/` — `THartigan/mphil-dis-report`, branch `main`
- `LION/` — `THartigan/LION`, branch `feature/PaDIS_Implementation`
- `PaDIS-Lion-Compat/` — `THartigan/PaDIS`, branch `main`

## Working with the subtrees

Edit files inside a subtree and commit them from this repository as usual. The
helper script can then pull upstream changes into this repository or push
committed subtree changes back to the component repository.

```sh
# Show the configured subtrees and their latest local commits.
./scripts/subtrees.sh status

# Pull one component, or all components.
./scripts/subtrees.sh pull LION
./scripts/subtrees.sh pull all

# Push committed changes back to one component's configured upstream branch.
./scripts/subtrees.sh push LION
```

Accepted component names are `report-source`, `LION`, and
`PaDIS-Lion-Compat`. Pulling and pushing require a clean working tree. Pushing
also requires write access to the corresponding GitHub repository.
