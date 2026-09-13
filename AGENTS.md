# Repository guidance

- Keep planning documents and `todos.md` local-only. Do not add `docs/*PLAN*.md` or `todos.md` to Git.
- Comments and Javadocs must describe the current implementation and its active invariants only.
- Do not preserve implementation history, migration notes, completed phases, or superseded behavior in source comments. Git history owns that context.
- Do not reference internal plan steps, phase labels, milestone IDs, or numbered design-document sections from source comments.
- Prefer direct explanations of why the current code is required, especially API contracts, synchronization rules, units, and non-obvious constraints.
- Every option string must exist in every locale: a new option ships with its translations, and `CausticaLangTest` fails the build if a locale is missing one. English (`en_us.json`) is the reference for keys and order.
