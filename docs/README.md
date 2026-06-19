# docs/ — signpost only

The project spec / blueprint is **not** stored in this code repo, on purpose (one source of truth,
no drift). It lives in the Obsidian vault:

```
C:\Year3\Obsidian\Yuval\30_Projects\smart-traffic-rl\
```

Start there for anything design-related:

- `backlog.md` — the master, ordered task list (T-00-01 first).
- `decisions.md` + `notes/adr-*.md` — locked decisions and ADRs.
- `notes/kpis.md`, `notes/evaluation-methodology.md`, `notes/risks-and-mitigations.md`
- `final-glossary.md` — frozen vocabulary.
- `specs/movements.yaml`, `specs/data-schema.md` — the two formal specs.

The vault is its own git repo (tag `spec-frozen-v1`). The code in this repo conforms to that spec.
