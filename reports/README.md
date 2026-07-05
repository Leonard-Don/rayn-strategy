# Reports Guide

This directory is for generated research artifacts.

## Evidence

`reports/evidence/` is for reproducible CSV, PNG, and Markdown outputs created
by local tools. Generated artifacts are ignored by default so the public source
tree stays small and reviewable.

## Runtime

`reports/runtime/` is for local scratch state, temporary reviews, and caches.
Do not commit machine-local state from this directory.

## Durable Notes

Keep durable methodology in reviewed source files such as `README.md`,
`docs/PROJECT_MAP.md`, or small Markdown notes that do not include local account
state or generated caches.
