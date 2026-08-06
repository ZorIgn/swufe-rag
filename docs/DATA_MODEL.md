# Data model

The canonical SQLite projection has distinct `sources`, `programs`,
`program_aliases`, `modules`, `module_aliases`, `courses`, `course_aliases`,
`program_courses`, `requirements`, and `source_sections` tables.

Every structured record retains `source_id`, page/chunk provenance, parser
version, extraction time, confidence, and review status. Sources include
authority, publication/effective dates, status, and an optional supersession
relationship. Aliases are data in SQLite plus `config/entity_aliases.json`, never
Python program-name conditionals.

`scripts.build_all` creates `data/academic.sqlite3` and an immutable manifest in
`artifacts/manifests/`. Both are local generated artifacts, not Git content.
