# CHANGELOG

<!-- version list -->

## v0.33.0 (2026-03-05)

### Features

- Query-driven graph explorer with SSE streaming
  ([`9ec4906`](https://github.com/jason-weddington/personal-kb-mcp/commit/9ec490645a4cd28ef16a17260a47fcbf6d93bd09))


## v0.32.0 (2026-03-05)

### Documentation

- Mark audit H8 as done in roadmap
  ([`d4a6f8b`](https://github.com/jason-weddington/personal-kb-mcp/commit/d4a6f8b2783fa104ee66a925ead72cf64382d92a))

### Features

- Kb_explore — interactive graph explorer in browser
  ([`bc96bf0`](https://github.com/jason-weddington/personal-kb-mcp/commit/bc96bf0d6f25ec3244eb4f97f88a1b1b9689b12a))


## v0.31.1 (2026-03-04)

### Bug Fixes

- Handle partial failures in batch store instead of silent half-commit
  ([`f043e38`](https://github.com/jason-weddington/personal-kb-mcp/commit/f043e38a909ce0982a8505e516043c15c66e33d9))


## v0.31.0 (2026-03-04)

### Features

- Add list_projects, list_contributors, list_teams discovery tools
  ([`3ac83de`](https://github.com/jason-weddington/personal-kb-mcp/commit/3ac83deff548f30faffdd07afbeba6b910d50093))


## v0.30.0 (2026-03-04)

### Features

- Add personal_kb_ prefix for KB_INSTANCE_ROLE=personal
  ([`751ce70`](https://github.com/jason-weddington/personal-kb-mcp/commit/751ce702c3c3ddc1dbf771a32b649e996af06e0e))


## v0.29.0 (2026-03-04)

### Features

- Tool name prefixing via KB_INSTANCE_ROLE
  ([`0edbc02`](https://github.com/jason-weddington/personal-kb-mcp/commit/0edbc0211b041ab6bb6ebfb5b1ff65b529126202))


## v0.28.0 (2026-03-04)

### Documentation

- Add AWS team setup guide and migration script usage
  ([`f9fa3f1`](https://github.com/jason-weddington/personal-kb-mcp/commit/f9fa3f1d3928984805dfb50d5d446ad2d4719452))

### Features

- Add KB_INSTANCE_ROLE and update README with uvx guidance
  ([`dac4162`](https://github.com/jason-weddington/personal-kb-mcp/commit/dac41629d8669f1a4bf8a5353a50ab0b240908bf))


## v0.27.0 (2026-03-03)

### Features

- SQLite-to-Postgres migration script and improved tool descriptions
  ([`449a675`](https://github.com/jason-weddington/personal-kb-mcp/commit/449a67597f786c16188a8c427d5659eb07221583))


## v0.26.0 (2026-03-03)

### Features

- Add "check KB first" nudge to server instructions
  ([`92171b3`](https://github.com/jason-weddington/personal-kb-mcp/commit/92171b3956e9497d537f6fd23a122c946db5313a))


## v0.25.4 (2026-03-03)

### Bug Fixes

- Quote-aware placeholder translation in Postgres backend (audit H5)
  ([`ac985b4`](https://github.com/jason-weddington/personal-kb-mcp/commit/ac985b41fd14994caca55b26aa5eb0ccda59bfc3))


## v0.25.3 (2026-03-03)

### Bug Fixes

- Wire update params through kb_store (audit H7)
  ([`6d1550a`](https://github.com/jason-weddington/personal-kb-mcp/commit/6d1550a184334a8148795f4183fd9b053841b7eb))


## v0.25.2 (2026-03-03)

### Bug Fixes

- Replace conditional test assertions with skip-or-assert
  ([`2bb93be`](https://github.com/jason-weddington/personal-kb-mcp/commit/2bb93becb54fe5577d46b48ce89cb032db3585f4))


## v0.25.1 (2026-03-02)

### Bug Fixes

- Harden ingestion, search, and input validation (audit quick wins)
  ([`eb4087a`](https://github.com/jason-weddington/personal-kb-mcp/commit/eb4087a763460739c71c303c872a365e0c4abf66))


## v0.25.0 (2026-03-02)

### Documentation

- Add agentic ingestion, agentic synthesis, and URL ingestion to how_it_works
  ([`10cff95`](https://github.com/jason-weddington/personal-kb-mcp/commit/10cff9548fd939ccdfcc27d5b4132031468622a7))

- Add documentation workflow guidance to CLAUDE.md
  ([`78d661e`](https://github.com/jason-weddington/personal-kb-mcp/commit/78d661eb0434ac7abe12eaccaadca8fb9bc5b7c0))

- Fix detect-secrets detector list and flagging language in how_it_works
  ([`90bfb30`](https://github.com/jason-weddington/personal-kb-mcp/commit/90bfb304c5af0f4ff875827de889ab005650c4c6))

### Features

- Aurora IAM database authentication
  ([`91a4929`](https://github.com/jason-weddington/personal-kb-mcp/commit/91a49292905165c685ae4eba773edab67515bb71))


## v0.24.0 (2026-03-02)

### Features

- URL ingestion support for kb_ingest
  ([`63e08a5`](https://github.com/jason-weddington/personal-kb-mcp/commit/63e08a5ba04a63536b60ec68f72668347698a862))


## v0.23.1 (2026-03-02)

### Bug Fixes

- Pass contributor through deactivate/reactivate audit + update README
  ([`d74dbad`](https://github.com/jason-weddington/personal-kb-mcp/commit/d74dbad16e0c25691b9ea7bf808421963d67fc84))


## v0.23.0 (2026-03-02)

### Chores

- Include optional deps in dev group so all tests run locally
  ([`0b27363`](https://github.com/jason-weddington/personal-kb-mcp/commit/0b27363d1c1c04c4f466034e7ca3e4dc68c93baf))

### Features

- Multi-user Phase 2 & 3 — attribution, filters, audit, sensitivity
  ([`2527fca`](https://github.com/jason-weddington/personal-kb-mcp/commit/2527fcaaf1bd32c5de0c82904e9a26633663c02a))


## v0.22.0 (2026-03-02)

### Features

- Multi-user Phase 1 — attribution, concurrency fixes, secret scanning
  ([`c4c8b0f`](https://github.com/jason-weddington/personal-kb-mcp/commit/c4c8b0f057f9a7c48d332f89ffdf8e30f5a2bd85))


## v0.21.0 (2026-03-02)

### Features

- Agent feedback loop with search telemetry and structured feedback
  ([`63541a9`](https://github.com/jason-weddington/personal-kb-mcp/commit/63541a904b4f6842a27e8eeaeaaac3a7f7784617))


## v0.20.0 (2026-03-01)

### Features

- Agentic synthesis with coverage check for kb_summarize
  ([`806a22a`](https://github.com/jason-weddington/personal-kb-mcp/commit/806a22a645d5108dbfea1bc5f381b6b0e2dc872e))


## v0.19.0 (2026-03-01)

### Documentation

- Expand eval section in CLAUDE.md with agent baseline workflow
  ([`9fae8c6`](https://github.com/jason-weddington/personal-kb-mcp/commit/9fae8c6032a60aa898fcd1aa70fd13ceb221c4e2))

### Features

- Agentic ingestion with chunking and KB-aware dedup
  ([`f2ab5a6`](https://github.com/jason-weddington/personal-kb-mcp/commit/f2ab5a66825de52a5874a20333645a5f0a79c330))


## v0.18.1 (2026-03-01)

### Bug Fixes

- Exclude eval-marked tests from pre-push hook
  ([`ae3a4be`](https://github.com/jason-weddington/personal-kb-mcp/commit/ae3a4bee471b246a8f223998bd9359bf4e974627))

### Documentation

- Document agentic query planning
  ([`4d756e8`](https://github.com/jason-weddington/personal-kb-mcp/commit/4d756e8a0566a749b0e0326efb4191c1a0225d0d))


## v0.18.0 (2026-03-01)

### Features

- Agentic query planning for kb_ask
  ([`643305c`](https://github.com/jason-weddington/personal-kb-mcp/commit/643305c6f399ebf2455c49435a4f907ee7e2c3c5))


## v0.17.0 (2026-02-28)

### Features

- Add relative RRF score threshold to filter low-relevance results
  ([`0e00655`](https://github.com/jason-weddington/personal-kb-mcp/commit/0e0065502895b6b338259571b0260cc817f3add0))

- Switch vector search from L2 to cosine distance
  ([`51fb9ec`](https://github.com/jason-weddington/personal-kb-mcp/commit/51fb9ec80b952143ee90312308ab2c99fa87e899))


## v0.16.0 (2026-02-28)

### Features

- Switch vector search from L2 to cosine distance
  ([`51fb9ec`](https://github.com/jason-weddington/personal-kb-mcp/commit/51fb9ec80b952143ee90312308ab2c99fa87e899))


## v0.15.1 (2026-02-28)

### Bug Fixes

- Auto-rebuild embeddings during postgres migration
  ([`4ae866b`](https://github.com/jason-weddington/personal-kb-mcp/commit/4ae866ba78e3da3bac080fa58c878b7e042a5518))


## v0.15.0 (2026-02-28)

### Documentation

- Add eval baseline workflow to CLAUDE.md
  ([`961ce29`](https://github.com/jason-weddington/personal-kb-mcp/commit/961ce298dda3b93cf3e4e60cf45c76e2fdffc1c0))

- Update roadmap — eval framework and graph quality shipped
  ([`9225b38`](https://github.com/jason-weddington/personal-kb-mcp/commit/9225b38a611a50591688760127d5868c03f85b89))

- Update roadmap — graph ranking rejected, storage portability promoted
  ([`034f879`](https://github.com/jason-weddington/personal-kb-mcp/commit/034f87902c0be2c0f79948e1ab8d43938495d018))

### Features

- Add PostgreSQL backend with asyncpg and pgvector
  ([`a2422ac`](https://github.com/jason-weddington/personal-kb-mcp/commit/a2422ace01dcbae613625c0862d00ae31f8f5b15))

### Refactoring

- Introduce Database protocol and SQLiteBackend abstraction
  ([`2c99ce9`](https://github.com/jason-weddington/personal-kb-mcp/commit/2c99ce9a951019deed869280b8b9d0b915f86e69))


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
