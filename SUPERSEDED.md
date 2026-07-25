# This repository is superseded. Do not work here.

Everything moved into the fork on 2026-07-25. There is now ONE project.

## Work here instead

    C:\Users\jaber\RightNow-Full\sglang-tree-MAIN

    branch : main
    remote : https://github.com/RightNow-AI/sglang.git   (PUBLIC, see push rule)

Layout:

    python/sglang/srt/tree/        the engine, 13 modules
    test/registered/unit/tree/     129 tests, all passing
    bench/e2e/acceptance.py        production gate, 11/11 against a live server
    autotree/                      everything that used to be this repo
      autotree/sdk/                the Python SDK
      autotree/thoughtbench/       the benchmark suite
      autotree/figures/            paper figures
      autotree/deploy/             helm, k8s, slurm
      autotree/core/               the original standalone prototype
      autotree/scheduler/          the Rust scheduler prototype
      autotree/AGENTS-GOALs/       measurement ledgers, GITIGNORED on purpose

## Why the fork became the home and not this repo

Both directions give one project. This one is safe and reversible:

- 280 files moved into 7,420, all under an `autotree/` prefix that upstream
  SGLang never touches, so `git fetch origin && git merge origin/main` still
  works. The fork's value depends on tracking a fast-moving upstream.
- The reverse, folding 7,420 upstream files into this repo, has no shared git
  history, needs --allow-unrelated-histories, and permanently destroys upstream
  mergeability. That is not undoable.

## Push rule, unchanged

The fork's remote is PUBLIC. Commit locally, do not push mechanism branches to
`RightNow-AI/sglang`. `autotree/AGENTS-GOALs/` is gitignored for this reason.
Currently 127 commits ahead of origin/main, unpushed by design.

## This directory

Kept as an archive so nothing is lost and old paths still resolve. Nothing new
should be written here. If you find yourself editing files in this directory,
you are in the wrong place.
