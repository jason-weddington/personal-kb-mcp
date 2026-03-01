# CHANGELOG

<!-- version list -->

## v0.14.0 (2026-02-28)

### Features

- Add eval baseline snapshot
  ([`25cf89a`](https://github.com/jason-weddington/personal-kb-mcp/commit/25cf89aebdb4800532729378cb7c52c72f1b8c4e))


## v0.13.0 (2026-02-28)

### Features

- Add search quality eval framework and update how_it_works.md
  ([`74d92ae`](https://github.com/jason-weddington/personal-kb-mcp/commit/74d92ae67482b0fbf1203d48b12601e96f1383de))


## v0.12.1 (2026-02-28)

### Bug Fixes

- Only reset confidence decay on explicit kb_get retrieval
  ([`2b7117a`](https://github.com/jason-weddington/personal-kb-mcp/commit/2b7117a158812437aaca4afeffb4843c33ba7319))

### Chores

- Ratchet coverage threshold to 77%
  ([`80bc430`](https://github.com/jason-weddington/personal-kb-mcp/commit/80bc4300ac5a1fa9667bff880657783fc2a7c875))


## v0.12.0 (2026-02-28)

### Features

- Add graph-hint annotations on sparse search results
  ([`f1a3bf5`](https://github.com/jason-weddington/personal-kb-mcp/commit/f1a3bf5885751cfd73869d0b21f6b1416d6c964b))


## v0.11.0 (2026-02-28)

### Features

- Add access-aware confidence decay
  ([`a5d567a`](https://github.com/jason-weddington/personal-kb-mcp/commit/a5d567a08d5adee1a5b8267203bf05be015761f3))


## v0.10.0 (2026-02-28)

### Chores

- Add agent graph traversal guidance to roadmap, update Done
  ([`891684b`](https://github.com/jason-weddington/personal-kb-mcp/commit/891684b37394c6750ff1578a24ed398823bf3288))

- Add timing output to test_dry_run.py
  ([`3b82cce`](https://github.com/jason-weddington/personal-kb-mcp/commit/3b82cce145681abbdb7b472ed725b598cdc679bb))

### Documentation

- Add positioning and research-grounded graph improvement plan to roadmap
  ([`acaceb0`](https://github.com/jason-weddington/personal-kb-mcp/commit/acaceb0591bc89d088c9d478e771af9a481045a0))

- Rewrite README with uvx install, correct Bedrock auth, all tools
  ([`3fdab2a`](https://github.com/jason-weddington/personal-kb-mcp/commit/3fdab2af99757f70fc9f42fdc9f0574b95165413))

### Features

- Add entity deduplication in graph enricher
  ([`9242e57`](https://github.com/jason-weddington/personal-kb-mcp/commit/9242e574191c77b00cdcac43c633e0a04ab9746f))


## v0.9.2 (2026-02-27)

### Bug Fixes

- Show long_title in compact search results for better discoverability
  ([`e3d434e`](https://github.com/jason-weddington/personal-kb-mcp/commit/e3d434e4eea5926cf04c6d5991c87d01402a3c23))


## v0.9.1 (2026-02-27)

### Bug Fixes

- Kb_get skips inactive entries
  ([`fff774f`](https://github.com/jason-weddington/personal-kb-mcp/commit/fff774fff7743375d654da28d5920b354b6454b0))

### Chores

- Update roadmap philosophy, add dogfooding note, fix test_dry_run provider support
  ([`6177814`](https://github.com/jason-weddington/personal-kb-mcp/commit/6177814de222fd1669fba9a5e6759f146e7e2078))


## v0.9.0 (2026-02-27)

### Chores

- Rename KB_LLM_MODEL to KB_OLLAMA_MODEL for consistency
  ([`d52499c`](https://github.com/jason-weddington/personal-kb-mcp/commit/d52499c87117f07e94e3085309ad4e58c843dd78))

- Rename KB_LLM_TIMEOUT to KB_OLLAMA_LLM_TIMEOUT
  ([`1e77124`](https://github.com/jason-weddington/personal-kb-mcp/commit/1e77124638da41835fc631729873a8b6e180d7b7))

### Features

- Compact output, kb_get two-phase retrieval, kb_store_batch
  ([`19546b1`](https://github.com/jason-weddington/personal-kb-mcp/commit/19546b1971269f37f803a6e6af981c0f08c49884))


## v0.8.1 (2026-02-27)

### Bug Fixes

- Pin smithy-json to fork and fix detect-secrets 1.5 compat
  ([`b870ca7`](https://github.com/jason-weddington/personal-kb-mcp/commit/b870ca72763acff726b1f230b278dc4a5a56d048))


## v0.8.0 (2026-02-27)

### Features

- Add Bedrock bearer token auth and remove smithy-json workaround
  ([`d0f535e`](https://github.com/jason-weddington/personal-kb-mcp/commit/d0f535e4732bbfbfa2da263a7033fe944da88ec6))


## v0.7.0 (2026-02-27)

### Features

- Ungate kb_ingest with glob support and improve tool descriptions
  ([`5059ef2`](https://github.com/jason-weddington/personal-kb-mcp/commit/5059ef2a87efee193f268a28682697c83dfbc3da))

### Refactoring

- Add audience framing to extraction prompts
  ([`32cc488`](https://github.com/jason-weddington/personal-kb-mcp/commit/32cc488f170beb627fb74faa2c9a415f31ba3d51))


## v0.6.0 (2026-02-26)

### Features

- Prose-specific extraction prompt for notes and documentation
  ([`cd72fca`](https://github.com/jason-weddington/personal-kb-mcp/commit/cd72fcaaacbf852db89b1f7d6e53bcdd3fa80491))


## v0.5.1 (2026-02-26)

### Bug Fixes

- Semantic-release push config and SSH remote URL
  ([`3e368a2`](https://github.com/jason-weddington/personal-kb-mcp/commit/3e368a2b3d5771f45c6cdeb065a172aa2135390c))

- Use ssh:// URL format for semantic-release remote
  ([`4bfb8ea`](https://github.com/jason-weddington/personal-kb-mcp/commit/4bfb8ea0763e98c586155a2c0978eb608cf5e15a))


## v0.5.0 (2026-02-26)

### Chores

- Add release workflow with recursion guard
  ([`a526541`](https://github.com/jason-weddington/personal-kb-mcp/commit/a526541c43fe78d82706f9f465a5a5b255262d9f))

- Fix semantic-release config and add as dev dep
  ([`8d28751`](https://github.com/jason-weddington/personal-kb-mcp/commit/8d28751ab97a820b7658a1bbd28c9a9f9a0912f7))

### Features

- Code-specific extraction prompt for file ingestion
  ([`284548a`](https://github.com/jason-weddington/personal-kb-mcp/commit/284548a03e9b8d6baaa05848b6902d2fa40aa696))


## v0.4.0 (2026-02-26)

### Chores

- Raise coverage threshold to 76%
  ([`2c4ea69`](https://github.com/jason-weddington/personal-kb-mcp/commit/2c4ea6989a6409ed613700f6133f8ccc271ca625))

### Documentation

- Add how_it_works.md technical documentation
  ([`862baa9`](https://github.com/jason-weddington/personal-kb-mcp/commit/862baa92ea9de8d4b826f5ebac30ee13d14fec45))

- Consolidate roadmap into dedicated ROADMAP.md
  ([`fcd21bd`](https://github.com/jason-weddington/personal-kb-mcp/commit/fcd21bd2675905df715e4cd4bf5a85eb47acdbff))

- Improve README for public release and add setup script
  ([`372273f`](https://github.com/jason-weddington/personal-kb-mcp/commit/372273fc6e85bd550d6e31b0cd206f66b3d4d371))

- Update README with kb_ingest tool and mark initial scope complete
  ([`982b86c`](https://github.com/jason-weddington/personal-kb-mcp/commit/982b86c2ffccb5870ddbf9f0da291e54d1e72ce7))

### Features

- Add AWS Bedrock LLM provider
  ([`b0ccc29`](https://github.com/jason-weddington/personal-kb-mcp/commit/b0ccc29017f414f2f02e2b1f75c9aa5803122c30))

- Add kb_ingest MCP tool for disk file ingestion
  ([`2ee5aba`](https://github.com/jason-weddington/personal-kb-mcp/commit/2ee5aba78c8ef378557d9656395dd2a67c093292))

- Add one-liner install script
  ([`c04020a`](https://github.com/jason-weddington/personal-kb-mcp/commit/c04020a7a629f0b02e99e7d73df3e99ba122a02e))

- **db**: Add ingested_files table schema
  ([`611bfb4`](https://github.com/jason-weddington/personal-kb-mcp/commit/611bfb4ddbc6a6f6a241e7ac586449ba6630b87e))

- **ingest**: Add file ingestion orchestrator
  ([`2796d9f`](https://github.com/jason-weddington/personal-kb-mcp/commit/2796d9f96bb16f6a4b4fb0d963b4a755247eadb5))

- **ingest**: Add LLM file summarization and entry extraction
  ([`33f6424`](https://github.com/jason-weddington/personal-kb-mcp/commit/33f642469833aa793e057daf85dad9048e0a951f))

- **ingest**: Add safety pipeline with detect-secrets and scrubadub
  ([`4d73461`](https://github.com/jason-weddington/personal-kb-mcp/commit/4d73461f508c43409787aae5b2a367f7c55875e4))


## v0.2.0 (2026-02-24)

### Chores

- Add auto-versioning with semantic-release and conventional commits
  ([`a624c00`](https://github.com/jason-weddington/personal-kb-mcp/commit/a624c00c91b735523f5ffccab059ccaa4d607a8e))

- Configure semantic-release for 0.x versioning
  ([`1bcbdd9`](https://github.com/jason-weddington/personal-kb-mcp/commit/1bcbdd9ffecb2ad3523b2da6e13bf4421867620e))

### Features

- Add knowledge graph with deterministic extraction (Phase 3)
  ([`96608c0`](https://github.com/jason-weddington/personal-kb-mcp/commit/96608c036d18ef8f3d65427ca27584b5f91bd557))

- Add MCP server instructions for proactive KB usage
  ([`d4fc513`](https://github.com/jason-weddington/personal-kb-mcp/commit/d4fc51397c8ded19f79127094bfae993228ebd5f))


## v0.1.0 (2026-02-24)

- Initial Release
