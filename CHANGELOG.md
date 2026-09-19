# CHANGELOG

<!-- version list -->

## v0.67.0 (2026-09-19)

### Bug Fixes

- Apply metadata filters to the vector leg of hybrid search
  ([`e4d36d6`](https://github.com/jason-weddington/personal-kb-mcp/commit/e4d36d6fd61ca3e45c2237be36a576bf4ebce5e6))

- Cast properties to jsonb in delete_llm_edges (Postgres)
  ([`6fe8ab6`](https://github.com/jason-weddington/personal-kb-mcp/commit/6fe8ab6ccf760525b562881f03364d82138ab04b))

- Declare kb-core as a runtime dependency of personal-kb
  ([`57fb447`](https://github.com/jason-weddington/personal-kb-mcp/commit/57fb447a437960d1d09658bb31499fd536be4de4))

- Hook listener request uses the pinned endpoint contract (cwd_project/operating/source_label)
  ([`dc74dfc`](https://github.com/jason-weddington/personal-kb-mcp/commit/dc74dfcb4d75f246b144362f947340414c6811aa))

- Require sqlite-vec >=0.1.9 — 0.1.6 aarch64 wheel ships a 32-bit binary
  ([`cd5ccda`](https://github.com/jason-weddington/personal-kb-mcp/commit/cd5ccdaf2076d79766cbb004ee89ee20bc16f1e5))

- Worker parses the pinned listener response shape (pointer)
  ([`4061296`](https://github.com/jason-weddington/personal-kb-mcp/commit/40612961e7366aaa2fdc817b8380adf187258d2a))

- **embeddings**: Bound retries on post-embed write failure
  ([`d1b8484`](https://github.com/jason-weddington/personal-kb-mcp/commit/d1b8484dce48243b9e242f5dd451a43502924ad3))

- **graph**: Implement neighbors() + get_entries() on SQLiteBackend
  ([`8bbac72`](https://github.com/jason-weddington/personal-kb-mcp/commit/8bbac723f5d45e8e7aec2600dbb32fbcaadec943))

- **graph**: Stop metadata-only updates from deleting LLM-enriched edges
  ([`8917cb0`](https://github.com/jason-weddington/personal-kb-mcp/commit/8917cb0dde5f94b0675e7b60be5f44189400de16))

- **hook**: Cap telemetry POST timeout and bound the orphan sweep
  ([`4703fe1`](https://github.com/jason-weddington/personal-kb-mcp/commit/4703fe11cfc73790a4ac620380de86ca678c6e73))

- **kb-core**: Fail, not skip, postgres tests when KB_REQUIRE_POSTGRES_TESTS=1
  ([`ad4c0a0`](https://github.com/jason-weddington/personal-kb-mcp/commit/ad4c0a0d128b308dee3a7b3b5d31df97f7152cc3))

- **llm**: Place Anthropic cache_control on content blocks, not a top-level kwarg
  ([`c615980`](https://github.com/jason-weddington/personal-kb-mcp/commit/c61598092fd110e282a6b2dd1497972bf9d0d5d2))

- **tests**: Stub boto3 in IAM auth test so it doesn't hit real AWS
  ([`522913f`](https://github.com/jason-weddington/personal-kb-mcp/commit/522913f9641f4628b6e931097de5c49bcd0b661a))

### Chores

- Bump personal-kb-web-service lock pin to the SQLite-default commit
  ([`6f0df0e`](https://github.com/jason-weddington/personal-kb-mcp/commit/6f0df0e0860c311be1dddb5a9e362f7c30218df1))

- Decouple release from deploy; github-gated release discipline
  ([`7f9434b`](https://github.com/jason-weddington/personal-kb-mcp/commit/7f9434b4606623906dbfbe9b7bce8bb362b6bd28))

- Default test command excludes live-API eval tests
  ([`016aeee`](https://github.com/jason-weddington/personal-kb-mcp/commit/016aeee9bdf158bf73d90dba8afd0578b5dea392))

- Delete the deprecated in-repo web explorer
  ([`cf393ce`](https://github.com/jason-weddington/personal-kb-mcp/commit/cf393ce3fac8e80ee6aa07ebaafef2f1bb523153))

- Drop smithy-json fork pin for PyPI 0.2.2
  ([`f662c55`](https://github.com/jason-weddington/personal-kb-mcp/commit/f662c55c1d8ae38c1fb2d6b9017dbb6b4a1dc442))

- Lower coverage floor to the post-extraction baseline
  ([`36d1db4`](https://github.com/jason-weddington/personal-kb-mcp/commit/36d1db4027ef7ba17df30720d4a7593d537fec51))

- Release.sh stamps the standalone hook package to the repo version
  ([`ba9d749`](https://github.com/jason-weddington/personal-kb-mcp/commit/ba9d74960d3ca11692dc8cb0f6cc1c582c859e47))

- Tell build agents how to handle the sqlite-vec sandbox gap
  ([`a73e48d`](https://github.com/jason-weddington/personal-kb-mcp/commit/a73e48d2da4fae4d0ef907133d00e275dae1612d))

- Update roadmap priorities and refresh agent eval baseline
  ([`c2dfc8e`](https://github.com/jason-weddington/personal-kb-mcp/commit/c2dfc8e573a0a220be0a21dfc48f46c1e249eb74))

- Vendor map-building workflows + per-machine setup.sh
  ([`6bda001`](https://github.com/jason-weddington/personal-kb-mcp/commit/6bda001f27b3e6265cb3b0dbb685d5e11f89cb88))

- **kb-core**: Stop mypy depending on whether sqlite-vec is installed
  ([`21a05fd`](https://github.com/jason-weddington/personal-kb-mcp/commit/21a05fd5f9334dfc0a85b19756270006da4ed9bd))

### Documentation

- Add cross-project knowledge surfacing to ROADMAP Next
  ([`a0a609c`](https://github.com/jason-weddington/personal-kb-mcp/commit/a0a609ca61e0121d6afb2642c49281e0c0c599d6))

- Add mental map spec
  ([`756b5c7`](https://github.com/jason-weddington/personal-kb-mcp/commit/756b5c7267a7922ffcd64513eb643ca89530c2d2))

- Correct get_personal_kb_url docstring (no in-process fallback post-14ff626)
  ([`fd75595`](https://github.com/jason-weddington/personal-kb-mcp/commit/fd755957e4b9129cea1eb81b7f285d600457e80a))

- Correct the zero-pointer invariant the workflow taught agents
  ([`5dec8ba`](https://github.com/jason-weddington/personal-kb-mcp/commit/5dec8ba21aaac1a2cf640e04c7ce43fe43f878cd))

- Cover the full mental_map feature in README + how_it_works
  ([`2b8963b`](https://github.com/jason-weddington/personal-kb-mcp/commit/2b8963ba90597b51d193cbffda7cde83e67605ce))

- Drop [postgres] from client install examples + purge dead explorer env vars
  ([`077a4a0`](https://github.com/jason-weddington/personal-kb-mcp/commit/077a4a0df893d146d06666ee720b023481a2f825))

- Fix personal-kb-hook install command (package name)
  ([`1f3ace0`](https://github.com/jason-weddington/personal-kb-mcp/commit/1f3ace02d8f68a98e78b388f8cd961df04188750))

- Headless agents expect working sqlite-vec, self-verify eval
  ([`2e33353`](https://github.com/jason-weddington/personal-kb-mcp/commit/2e33353f1d09bc42fb88bd57a9f6397a38ef01a7))

- Record mental maps under Done in ROADMAP
  ([`aab5e41`](https://github.com/jason-weddington/personal-kb-mcp/commit/aab5e41132183518f1349331948bcac4fbf3be72))

- Rework root how_it_works.md for thin-client split + listener
  ([`df3b547`](https://github.com/jason-weddington/personal-kb-mcp/commit/df3b54735993d067ab576d943a75859ee0587b73))

- Roadmap — dynamic awareness injection (computed graph neighborhood)
  ([`b108f31`](https://github.com/jason-weddington/personal-kb-mcp/commit/b108f31349d57ccb8b4e2a7c82e9a2e2c679b821))

- Settled mental-map design after adversarial debate
  ([`75d6930`](https://github.com/jason-weddington/personal-kb-mcp/commit/75d69305a589bca3a71a46a3ec08bd71761fb6ed))

- **kb-core**: Author engine-internals how_it_works.md
  ([`d1dbf13`](https://github.com/jason-weddington/personal-kb-mcp/commit/d1dbf13952d52471408f9e9c3b1aad2d13436218))

### Features

- Add mental_map entry_type as first-class storable
  ([`11657b0`](https://github.com/jason-weddington/personal-kb-mcp/commit/11657b07189318472daaf2e75862a44893aebda4))

- Advisory fact-free lint for mental_map bodies (§7.3)
  ([`e3bd55a`](https://github.com/jason-weddington/personal-kb-mcp/commit/e3bd55a7502f0c09c809c62458392d42f49b6002))

- Chain-credit whisper telemetry consumed via map pointer lists
  ([`a2a6ece`](https://github.com/jason-weddington/personal-kb-mcp/commit/a2a6ece8e969925c7984ee6e1e091ead0d702421))

- Expose raw per-leg relevance signals on SearchResult
  ([`b88af59`](https://github.com/jason-weddington/personal-kb-mcp/commit/b88af590117354896afaef881c403cb013377d26))

- Kb-core distribution-readiness + clean-install smoke check
  ([`ea1b2ef`](https://github.com/jason-weddington/personal-kb-mcp/commit/ea1b2ef2f6225ee1b6f9efd46dfe901570ea285e))

- Kb_preflight Maps index for mental_map pull surfacing (§7.7)
  ([`400b3e7`](https://github.com/jason-weddington/personal-kb-mcp/commit/400b3e77681a5caa70a36ff0512728a33f8dbbb4))

- KnowledgeBase facade + create_sqlite/create_postgres factories
  ([`24346da`](https://github.com/jason-weddington/personal-kb-mcp/commit/24346da8f0dc55eb6c24c4a6292e43b4e474139e))

- Lift retrieve/synthesis bodies into kb_core.query
  ([`afdd7f5`](https://github.com/jason-weddington/personal-kb-mcp/commit/afdd7f5e3ed6f825ce314b2fa77f4938ee4c1573))

- Listener Gate 0 — cross-project map titles in the hook injection
  ([`f158973`](https://github.com/jason-weddington/personal-kb-mcp/commit/f15897365957dfe4ee3c135b066f8bd3cfc566fc))

- Listener Gate 2b — whisper-next-turn live injection in the hook
  ([`4e1d08b`](https://github.com/jason-weddington/personal-kb-mcp/commit/4e1d08b4aaf02d0e4b9193f3762ef96bff3f2f08))

- Local real-time whisper-decision debug log (hook side)
  ([`71d9bf0`](https://github.com/jason-weddington/personal-kb-mcp/commit/71d9bf0ee679ae42a484ef2ee955c4b332aece11))

- Make silent graph-enrichment failures observable
  ([`cdbbda6`](https://github.com/jason-weddington/personal-kb-mcp/commit/cdbbda6c36c7a8dac1811cfb11f7b7e089339128))

- Maps index per-instance files + startup rebuild + LISTEN/NOTIFY refresh
  ([`020118a`](https://github.com/jason-weddington/personal-kb-mcp/commit/020118aab4bcb2e5808ae6a5fd775468895347cc))

- MCP server auto-spawns a singleton local kb-service daemon
  ([`dc0c5c5`](https://github.com/jason-weddington/personal-kb-mcp/commit/dc0c5c55a121a23e4d56ae4514df8eee8c7fd103))

- Move + de-env ingest pipeline into kb_core
  ([`a0f3424`](https://github.com/jason-weddington/personal-kb-mcp/commit/a0f3424fa2197670128dcc33e330989e2f3a42d1))

- Move + de-env LLM provider impls into kb_core
  ([`edd2ef7`](https://github.com/jason-weddington/personal-kb-mcp/commit/edd2ef79b0e98965e13363d8e4a217708b145546))

- Move + de-env search/embeddings into kb_core
  ([`c56d1e2`](https://github.com/jason-weddington/personal-kb-mcp/commit/c56d1e25021ca1163328faca03ed00a6b103ab68))

- Move formatters/ttl/coverage + graph/agent into kb_core
  ([`60fa381`](https://github.com/jason-weddington/personal-kb-mcp/commit/60fa381bc73b0d021ebc43f462a935ea32a23eaf))

- Move pure-core nucleus into kb_core with re-export shims
  ([`415d3e4`](https://github.com/jason-weddington/personal-kb-mcp/commit/415d3e4273dcbe8b5badb509ccbde7b12b180616))

- On-GET pointer-rot note for mental_map (§7.4)
  ([`cb86550`](https://github.com/jason-weddington/personal-kb-mcp/commit/cb865503163b08d86ea991243b565565e85bac55))

- Per-request attribution kwargs on ingest facade + dry_run on ingest_text
  ([`ba9d5c5`](https://github.com/jason-weddington/personal-kb-mcp/commit/ba9d5c59e9719158ec9d9f2b406324aecfc6db2b))

- Personal-kb-hook push surface + .kb_project convention (§7.7)
  ([`e34da86`](https://github.com/jason-weddington/personal-kb-mcp/commit/e34da86b38207fb6d3b6bf00a2dca31048020b72))

- Personal-kb[local] extra + local-mode README so onboarding actually works
  ([`9112cb7`](https://github.com/jason-weddington/personal-kb-mcp/commit/9112cb7c0b47fe6a7f70bdc68b0a1257a54df335))

- Prompt caching on multi-turn Anthropic paths
  ([`7e77cf3`](https://github.com/jason-weddington/personal-kb-mcp/commit/7e77cf3fbbf23e23cb4f0e51388399300fb6ad6d))

- Rewire the MCP channel onto the KnowledgeBase facade
  ([`e47e1f4`](https://github.com/jason-weddington/personal-kb-mcp/commit/e47e1f4459593a125cbcf3d0845da3b080ed0bfb))

- Rewire the web channel onto KnowledgeBase + drop dead shims
  ([`176fbac`](https://github.com/jason-weddington/personal-kb-mcp/commit/176fbac618701bf66fbb385a9d33b9f30d4de813))

- Scaffold kb-core package + KbConfig + import-purity guard
  ([`0fac760`](https://github.com/jason-weddington/personal-kb-mcp/commit/0fac760d6afcce52802f1dae1d6646fc68cf16a1))

- Setup.sh local/remote mode branch
  ([`ec840bb`](https://github.com/jason-weddington/personal-kb-mcp/commit/ec840bb8cfd5c77521dcdf7cd6ff8f6c7117bd8a))

- Split personal-kb-hook into a standalone zero-dependency package
  ([`0789ff2`](https://github.com/jason-weddington/personal-kb-mcp/commit/0789ff2064b4314f38da3bcbf30b3bbe9b4bffc5))

- Surface ALL of a project's maps (drop the LIMIT 5 cap)
  ([`0a1a3c4`](https://github.com/jason-weddington/personal-kb-mcp/commit/0a1a3c462033f9504f70231502506f266232d011))

- Thin MCP client — Backend seam (HttpBackend + LocalBackend) + 16 tool shims
  ([`3dbf276`](https://github.com/jason-weddington/personal-kb-mcp/commit/3dbf276d84ae327efe2ee45acfa16fc500bb5631))

- Whisper-efficacy telemetry on the hook side (4 touchpoints)
  ([`97737d0`](https://github.com/jason-weddington/personal-kb-mcp/commit/97737d0a193f963febae3acd5674817c42908d80))

- **embeddings**: Keep the embedding model warm via per-request keep_alive
  ([`3b90958`](https://github.com/jason-weddington/personal-kb-mcp/commit/3b90958d82f3209509e41da8c36cbe55b9358179))

- **embeddings**: Self-healing embedding retry queue
  ([`7ba56bf`](https://github.com/jason-weddington/personal-kb-mcp/commit/7ba56bf1bc877a84da574490bfc75b5c6d418f78))

- **eval**: Extend search-eval corpus with LongMemEval 5-ability taxonomy
  ([`cf58a5f`](https://github.com/jason-weddington/personal-kb-mcp/commit/cf58a5f7aa4f317f41f5a1048c74951fed68cf22))

- **hook**: Accept up to two map pointers per KB
  ([`905a33a`](https://github.com/jason-weddington/personal-kb-mcp/commit/905a33ab27e5885a6c1a8f0518d938481fd8123f))

- **hook**: Announce new maps as a one-line delta, not the full roster
  ([`961bd97`](https://github.com/jason-weddington/personal-kb-mcp/commit/961bd97566de80d83fa4af7246c92fb9a9d2032f))

- **hook**: HTTP fetch of maps-index with silent local fallback
  ([`5b7dda7`](https://github.com/jason-weddington/personal-kb-mcp/commit/5b7dda71b37b46f130340d72f8d4d075c561fec0))

- **hook**: Listener (whisper) fan-out across the KB roster (P2, built dark)
  ([`5eff34f`](https://github.com/jason-weddington/personal-kb-mcp/commit/5eff34f3d36f7d0197eedf01f36305b0b1ae0b0c))

- **hook**: Maps-index fan-out across the KB roster (P1, coverage-only)
  ([`92b729b`](https://github.com/jason-weddington/personal-kb-mcp/commit/92b729b170a0ec1da2a03c46132e1a1635e5498b))

- **hook**: Record the hook event name in the whisper-debug header
  ([`d6a43ce`](https://github.com/jason-weddington/personal-kb-mcp/commit/d6a43ce33958ca4ee7630d7f9d34c19a4f4442b5))

- **hook**: Record why the map roster emitted
  ([`4d6efee`](https://github.com/jason-weddington/personal-kb-mcp/commit/4d6efee274a18b66aa504aa64909043b88055ba0))

- **hook**: Roster loader + legacy fallback (P0, no behavior change)
  ([`19bf607`](https://github.com/jason-weddington/personal-kb-mcp/commit/19bf607777e3d29a334ded11d671fbd0dff67254))

### Refactoring

- Delete LocalBackend + maps_index_writer + LISTEN/NOTIFY (thin client everywhere)
  ([`14ff626`](https://github.com/jason-weddington/personal-kb-mcp/commit/14ff6262234eb70f9bc1a60e962fe468a0a981cf))

- Maps index renders orientation (long_title), leads the primer
  ([`e7801c1`](https://github.com/jason-weddington/personal-kb-mcp/commit/e7801c13535be5b28b06e4cfa09912a39eaace4e))

- Remove on-disk JSONL fallback from the maps-index hook
  ([`63c5122`](https://github.com/jason-weddington/personal-kb-mcp/commit/63c5122662520467df9828e9c719b908f8105490))

### Testing

- CI contract + smoke tests for the documented local-mode install path
  ([`d3e7873`](https://github.com/jason-weddington/personal-kb-mcp/commit/d3e7873d32563f582725cb4195efb5f32619c11b))

- Contract-test that local-mode README blocks never set KB_DATABASE_URL
  ([`ed4c4d5`](https://github.com/jason-weddington/personal-kb-mcp/commit/ed4c4d5ddf4442861edb779eccd1faa3a718413e))

- Kb-core real-Postgres integration suite (KB_TEST_DATABASE_URL, skippable)
  ([`c91209b`](https://github.com/jason-weddington/personal-kb-mcp/commit/c91209bb5e3efb989bae60fd2d91879c43c700fc))

- **kb-core**: Guard SQLite/Postgres parity for embedding_retry_queue
  ([`76bf7bd`](https://github.com/jason-weddington/personal-kb-mcp/commit/76bf7bd9704fa7a3dc64c2e5ebc56f808f4956d6))


## v0.66.1 (2026-04-21)

### Bug Fixes

- Change default explorer port from 8765 to 8767
  ([`4cf16a3`](https://github.com/jason-weddington/personal-kb-mcp/commit/4cf16a335015f29f026197267c71146589cc796d))


## v0.66.0 (2026-04-16)

### Features

- Make team field mutable via kb_bulk_update
  ([`e380cf4`](https://github.com/jason-weddington/personal-kb-mcp/commit/e380cf49cd4da0088f6bcab3f1833754fed9c0d5))


## v0.65.0 (2026-04-16)

### Features

- **feedback**: Display attribution badges in list/summarize feedback
  ([`15dc212`](https://github.com/jason-weddington/personal-kb-mcp/commit/15dc212372daf484a04be2ef2979adba6e5285ea))


## v0.64.0 (2026-04-16)

### Bug Fixes

- **explorer**: Unify search and history into single container
  ([`1174914`](https://github.com/jason-weddington/personal-kb-mcp/commit/1174914ce6ff066da16c0a295c934d61180fb690))

- **explorer**: Use shared formatters for chat entry context
  ([`80d0503`](https://github.com/jason-weddington/personal-kb-mcp/commit/80d0503aac171762b8ca4816a9d715f837fa55f0))

### Features

- **feedback**: Add team attribution to agent_feedback storage
  ([`caffd82`](https://github.com/jason-weddington/personal-kb-mcp/commit/caffd8233e31352e860d195b1806ca724613bc96))


## v0.63.1 (2026-04-10)

### Bug Fixes

- **explorer**: Use shared formatters for chat entry context
  ([`80d0503`](https://github.com/jason-weddington/personal-kb-mcp/commit/80d0503aac171762b8ca4816a9d715f837fa55f0))


## v0.63.0 (2026-04-10)

### Features

- Add dev-setup script and CONTRIBUTING guide
  ([`ae0c858`](https://github.com/jason-weddington/personal-kb-mcp/commit/ae0c8582365f2a7c95f7649d83ff559807c451ed))


## v0.62.0 (2026-04-09)

### Bug Fixes

- **store**: Wrap bulk_update writes in transaction
  ([`de425b0`](https://github.com/jason-weddington/personal-kb-mcp/commit/de425b0187e3588057260b9000bb9da2656362ca))

### Chores

- Remove .kiro/ from repo and add to .gitignore
  ([`0b17f0b`](https://github.com/jason-weddington/personal-kb-mcp/commit/0b17f0b3dcc9f84c673f67bdba9f8b2bb5125645))

### Documentation

- Add scaling analysis for corpus-level ingestion
  ([`90db556`](https://github.com/jason-weddington/personal-kb-mcp/commit/90db5562407d61d9599b4c323581e74448a5c8c0))

- **store**: Add transaction convention comment to KnowledgeStore
  ([`74d9074`](https://github.com/jason-weddington/personal-kb-mcp/commit/74d9074c5d8aa7b2e79d40b1cbf1f9f09b5d26d1))

### Features

- Batch ingestion pipeline and browser PDF filename fix
  ([`8c798f0`](https://github.com/jason-weddington/personal-kb-mcp/commit/8c798f0f869d8834a50cdcc8a11b48bd20520228))

- Persistent chat history in web explorer
  ([`27eab63`](https://github.com/jason-weddington/personal-kb-mcp/commit/27eab63ea5d70d39c45352fc91d98a23be06b727))

- Show updated_by attribution in kb_get full output
  ([`87de497`](https://github.com/jason-weddington/personal-kb-mcp/commit/87de4971aa4892f1cbf2a0d246ae4cf535a40eeb))

- **tools**: Add kb_bulk_update for batch metadata changes
  ([`c75319a`](https://github.com/jason-weddington/personal-kb-mcp/commit/c75319a6ed49fdc13d1a5d65fdcd2f61d3c97d80))

### Performance Improvements

- **store**: Eliminate N+1 queries in bulk_update
  ([`cec4332`](https://github.com/jason-weddington/personal-kb-mcp/commit/cec4332f2f651fc6b902a13951c0a1201e35a02b))


## v0.61.0 (2026-03-25)

### Features

- Support PDF upload in web explorer
  ([`1fd17b6`](https://github.com/jason-weddington/personal-kb-mcp/commit/1fd17b607985c001cc306737c73da518e6c8dbcf))


## v0.60.1 (2026-03-23)

### Bug Fixes

- Restore [safety] extra as empty alias for backwards compatibility
  ([`b06e463`](https://github.com/jason-weddington/personal-kb-mcp/commit/b06e463b722b422d502c49ae9e2d05f1fd13def5))


## v0.60.0 (2026-03-22)

### Features

- Add PDF ingestion support via PyMuPDF
  ([`d56a703`](https://github.com/jason-weddington/personal-kb-mcp/commit/d56a7030872379868f22f253c670d356774c2664))


## v0.59.0 (2026-03-20)

### Features

- Add transaction() context manager for atomic multi-step DB operations
  ([`b78a2ca`](https://github.com/jason-weddington/personal-kb-mcp/commit/b78a2ca57d48ceba459eda0c3e5c43fbfd13ef14))


## v0.58.3 (2026-03-13)

### Bug Fixes

- Validate supersedes hints match kb-XXXXX format
  ([`d6e1da8`](https://github.com/jason-weddington/personal-kb-mcp/commit/d6e1da8a194907c3413b1f743269733b71a513c4))

### Documentation

- Clarify kb_ingest_url modes for internal vs public sites
  ([`17e5c67`](https://github.com/jason-weddington/personal-kb-mcp/commit/17e5c67f0525498069dfd20f6c1b5e05cb0dfea5))


## v0.58.2 (2026-03-12)

### Bug Fixes

- Expand ingestion deny list with common secret patterns
  ([`ebfbefa`](https://github.com/jason-weddington/personal-kb-mcp/commit/ebfbefab1fe44fbc97665b28a5355209e3076dd7))

### Documentation

- Fix 6 factual errors and add missing features to how_it_works
  ([`7ac3a69`](https://github.com/jason-weddington/personal-kb-mcp/commit/7ac3a69f6922cea43da0905d8ca944de06057ace))


## v0.58.1 (2026-03-12)

### Bug Fixes

- Add config validation and deduplicate ingester pipeline
  ([`77220b3`](https://github.com/jason-weddington/personal-kb-mcp/commit/77220b398240f00ea35ec6d3e19bf166d2e2fa18))

### Chores

- Add vulture dead code detection to pre-commit
  ([`5464e94`](https://github.com/jason-weddington/personal-kb-mcp/commit/5464e94ae4868595b861194de549ece14e1f4010))

### Refactoring

- Delete dead VersionStore, narrow query_llm type, add agent dedup
  ([`0f3255d`](https://github.com/jason-weddington/personal-kb-mcp/commit/0f3255d03bef3d5ca59f35eb37ec7dcdcacc5e00))

- Extract shared LLM JSON parser from 7 files
  ([`4b71957`](https://github.com/jason-weddington/personal-kb-mcp/commit/4b7195748e36f63a965c49fbb2e01970b68a7e0c))


## v0.58.0 (2026-03-11)

### Documentation

- Add graph explorer research reports from initial design phase
  ([`cf01d9c`](https://github.com/jason-weddington/personal-kb-mcp/commit/cf01d9c478b7c3373c16efb0167e3ce0954876ed))

### Features

- Add content param to kb_ingest_url for pre-fetched content
  ([`3135525`](https://github.com/jason-weddington/personal-kb-mcp/commit/31355250bffadcfa6ab67a6c2a343d1f86a572d9))


## v0.57.2 (2026-03-09)

### Bug Fixes

- Set StaticCredentialsResolver for profile-based Bedrock auth
  ([`5fe37d0`](https://github.com/jason-weddington/personal-kb-mcp/commit/5fe37d0f908fdbac05a2102382d6f33e11039fe1))


## v0.57.1 (2026-03-09)

### Bug Fixes

- Profile credentials take priority over bearer token and env vars
  ([`ee264af`](https://github.com/jason-weddington/personal-kb-mcp/commit/ee264afe30f464a7abdfc928eae4287d1514c8e3))


## v0.57.0 (2026-03-09)

### Features

- Add AWS profile-based credentials for Bedrock
  ([`bbe8b63`](https://github.com/jason-weddington/personal-kb-mcp/commit/bbe8b63715bffa87c71597ec87d1caed738b584e))


## v0.56.0 (2026-03-09)

### Features

- Add kb_preflight tool with 2-hop graph expansion
  ([`c5d363b`](https://github.com/jason-weddington/personal-kb-mcp/commit/c5d363b1d1efcdbcc75b183adb7916aa83cb9ca3))


## v0.55.5 (2026-03-09)

### Bug Fixes

- Log preflight CWD at WARNING level for debugging
  ([`955f437`](https://github.com/jason-weddington/personal-kb-mcp/commit/955f4371ef33d9f81561c863cdd9204dcfa3d696))

### Chores

- Rework setup script to output MCP config, explorer optional
  ([`39b67b8`](https://github.com/jason-weddington/personal-kb-mcp/commit/39b67b8efe229c5b87a42351c80c536a53552ca3))


## v0.55.4 (2026-03-08)

### Bug Fixes

- Ignore Enter during IME composition in explorer inputs
  ([`7fc3f21`](https://github.com/jason-weddington/personal-kb-mcp/commit/7fc3f218bfefbe91b6190bb6871a84a3c15f69e6))


## v0.55.3 (2026-03-08)

### Bug Fixes

- Send Firefox user-agent on URL ingestion to avoid 403s
  ([`ad2cd31`](https://github.com/jason-weddington/personal-kb-mcp/commit/ad2cd31fbb257e92393f43025e915b54df8f5ecc))


## v0.55.2 (2026-03-08)

### Bug Fixes

- Standalone explorer missing ingestion deps, read prompt in piped script
  ([`b4e150f`](https://github.com/jason-weddington/personal-kb-mcp/commit/b4e150fa8f6aee8a256f4089ee8b27b6d0ff0f47))


## v0.55.1 (2026-03-08)

### Bug Fixes

- Check for Xcode CLT on macOS before proceeding with setup
  ([`8e158c3`](https://github.com/jason-weddington/personal-kb-mcp/commit/8e158c3d8723827d50bd1da8806c4a55b63054ef))

### Chores

- Add setup script for guided install on macOS and Linux
  ([`f83899a`](https://github.com/jason-weddington/personal-kb-mcp/commit/f83899a6911ca3a31fbe2e311c18b693a6b5309f))

- Remove outdated install and setup scripts
  ([`e97f7a4`](https://github.com/jason-weddington/personal-kb-mcp/commit/e97f7a4a7416e7834b80e88fdb1540b625483c03))


## v0.55.0 (2026-03-07)

### Features

- Team-scoped preflight context injection
  ([`1c4bea2`](https://github.com/jason-weddington/personal-kb-mcp/commit/1c4bea234b2a31e74857daa29e7f3ed8267f22d0))


## v0.54.0 (2026-03-07)

### Documentation

- Add preflight context injection to roadmap
  ([`7ef219c`](https://github.com/jason-weddington/personal-kb-mcp/commit/7ef219c2b5191f0edc25c919fe63cf4014011948))

- Document kb_ingest_url, entry TTL, Sonnet synthesis, explorer auto-start and write tools
  ([`1d8c251`](https://github.com/jason-weddington/personal-kb-mcp/commit/1d8c251f4abe9417ab41bc3a515f0a86ee10ae30))

### Features

- CWD-based preflight context injection
  ([`a50f598`](https://github.com/jason-weddington/personal-kb-mcp/commit/a50f598339f2a15b51404b8ff24192922d945472))


## v0.53.0 (2026-03-07)

### Chores

- Move trafilatura from optional web extra to core dependency
  ([`120d021`](https://github.com/jason-weddington/personal-kb-mcp/commit/120d0215f98ecb28a1ad2ee6b84ffd7b61074f0b))

### Documentation

- Add agent steering guide to README
  ([`3211f27`](https://github.com/jason-weddington/personal-kb-mcp/commit/3211f271fab14a0754bae20a63ecc2fc508f5db2))

### Features

- Auto-start explorer web server on MCP server startup
  ([`46b8730`](https://github.com/jason-weddington/personal-kb-mcp/commit/46b87306a6ee38b80fd181112226b7b78ceccc30))


## v0.52.4 (2026-03-07)

### Bug Fixes

- Reload graph data after ingestion and extend modal dismiss delay
  ([`43efe32`](https://github.com/jason-weddington/personal-kb-mcp/commit/43efe3237a5a9cecf388beab78cd8ac2dfa2c177))


## v0.52.3 (2026-03-07)

### Bug Fixes

- Replace project dropdown with editable combo box in ingest modal
  ([`711ffea`](https://github.com/jason-weddington/personal-kb-mcp/commit/711ffea3029fdbe0aa03a376d7791b0b3840e0f6))


## v0.52.2 (2026-03-07)

### Bug Fixes

- Show query status in search box placeholder instead of status line
  ([`71b7561`](https://github.com/jason-weddington/personal-kb-mcp/commit/71b7561be28b1e48cab093a40a65cb740ba68a02))


## v0.52.1 (2026-03-07)

### Bug Fixes

- Rename +URL toolbar button to +URL(s)
  ([`1a78f60`](https://github.com/jason-weddington/personal-kb-mcp/commit/1a78f60988f1421c9b1a339494df4d7f51b20ec4))


## v0.52.0 (2026-03-07)

### Features

- Add file upload, multi-URL, and progress streaming to explorer ingest
  ([`a0d61fc`](https://github.com/jason-weddington/personal-kb-mcp/commit/a0d61fcf5183f5d71c46e819e225ea5cca15df3b))


## v0.51.2 (2026-03-07)

### Bug Fixes

- Improve kb_list_projects description to prevent project_ref duplication
  ([`028fbf2`](https://github.com/jason-weddington/personal-kb-mcp/commit/028fbf2919f271afc22cd7c6e795488c50101fdb))


## v0.51.1 (2026-03-07)

### Bug Fixes

- Pass limit as int in filter-only search SQL params
  ([`eed73bc`](https://github.com/jason-weddington/personal-kb-mcp/commit/eed73bc0da1a9c2e3b4b862d7a03bc76e3766b1f))


## v0.51.0 (2026-03-07)

### Features

- Add explorer ingest URL button, project dropdown, and filter-only search
  ([`0ddf48c`](https://github.com/jason-weddington/personal-kb-mcp/commit/0ddf48ce7566f203a8157f72fd5cac828cb1dcb4))


## v0.50.0 (2026-03-07)

### Features

- Split kb_ingest into file and URL tools with HTML extraction
  ([`ee1c950`](https://github.com/jason-weddington/personal-kb-mcp/commit/ee1c9508edc6c478fe8214cadcadc99df4f19de4))


## v0.49.0 (2026-03-07)

### Features

- Add get_entry read tool to explorer chat
  ([`9b35ed3`](https://github.com/jason-weddington/personal-kb-mcp/commit/9b35ed39775d3f9362c5cb233ed3bd0dd75af448))


## v0.48.0 (2026-03-07)

### Features

- Explorer chat write-back — update_entry and ingest_url tools
  ([`fbbefe3`](https://github.com/jason-weddington/personal-kb-mcp/commit/fbbefe3c4e61235794e3996c2382d4c4c8e468b8))


## v0.47.0 (2026-03-07)

### Features

- Explorer thinking visuals — dim/glow/pulse during agent search
  ([`7567c23`](https://github.com/jason-weddington/personal-kb-mcp/commit/7567c23a3a01ef57899c0a7a750062fe8e54d9c2))


## v0.46.1 (2026-03-07)

### Bug Fixes

- Filter orphan nodes from graph explorer visualization
  ([`474f802`](https://github.com/jason-weddington/personal-kb-mcp/commit/474f8025e4a03770cbf8306a4404991324894cb9))


## v0.46.0 (2026-03-06)

### Features

- Explorer UX polish — chat header, copy, maximize, textareas, zoom cap
  ([`8f9c259`](https://github.com/jason-weddington/personal-kb-mcp/commit/8f9c25942d193812b5e7d37cc86f8d0db099cd71))


## v0.45.0 (2026-03-06)

### Features

- Bedrock retry/timeout, classifier fix, explore→chat bridge
  ([`45ca9eb`](https://github.com/jason-weddington/personal-kb-mcp/commit/45ca9eba4841b70177dc7cdf6500e727d6431e7b))


## v0.44.0 (2026-03-06)

### Features

- Explore port kill, Sonnet synthesis, metadata-only updates
  ([`8803359`](https://github.com/jason-weddington/personal-kb-mcp/commit/88033593f58cc6214d8b1363c147415e62f9f04a))


## v0.43.0 (2026-03-06)

### Features

- Entry TTL / expiry
  ([`308b146`](https://github.com/jason-weddington/personal-kb-mcp/commit/308b14628948db374f94deae311cc5981cb6a74b))


## v0.42.0 (2026-03-06)

### Features

- Zoom-aware label visibility in graph explorer
  ([`3211f07`](https://github.com/jason-weddington/personal-kb-mcp/commit/3211f07c47348ca39a836b6b45c1c63c7ac54ee2))


## v0.41.0 (2026-03-06)

### Features

- Per-chunk secret scanning and ingestion hardening
  ([`7beaa4f`](https://github.com/jason-weddington/personal-kb-mcp/commit/7beaa4fadf478c456ec64ca1c2195d1de4f63328))


## v0.40.0 (2026-03-05)

### Features

- Match info panel close button and animation to chat panel
  ([`69436ff`](https://github.com/jason-weddington/personal-kb-mcp/commit/69436ff2a2949ede01b6dc46928f786708617c4d))


## v0.39.0 (2026-03-05)

### Documentation

- Update explorer docs with multi-turn chat and animation details
  ([`aaad08f`](https://github.com/jason-weddington/personal-kb-mcp/commit/aaad08fdbfdfe48bdc160d5c1bfdb916596d28f2))

### Features

- Make fastapi and uvicorn core dependencies
  ([`2ffb18c`](https://github.com/jason-weddington/personal-kb-mcp/commit/2ffb18c9d50eca2500939c96739bc462e245f0f2))


## v0.38.0 (2026-03-05)

### Features

- Chat panel slide transition and visual polish
  ([`0ce8248`](https://github.com/jason-weddington/personal-kb-mcp/commit/0ce8248c43823780b9708054c38d5b4f863f985d))


## v0.37.0 (2026-03-05)

### Features

- Multi-turn chat in graph explorer
  ([`939a038`](https://github.com/jason-weddington/personal-kb-mcp/commit/939a0384420d1c5d39cd0ecc0e766e7646d1f2e1))


## v0.36.0 (2026-03-05)

### Features

- Info panel improvements — bold labels, confidence %, entry accordion, explore results
  ([`3ced88c`](https://github.com/jason-weddington/personal-kb-mcp/commit/3ced88ce094484a2c2d81fb3d78e518601402f09))


## v0.35.1 (2026-03-05)

### Bug Fixes

- Replace dizzy node-to-node jumps with smooth widening view
  ([`1b11c63`](https://github.com/jason-weddington/personal-kb-mcp/commit/1b11c63b43cdf7b6e5f025be108dee18b4696745))


## v0.35.0 (2026-03-05)

### Features

- Staggered node-by-node traversal animation in explorer
  ([`4415e66`](https://github.com/jason-weddington/personal-kb-mcp/commit/4415e6623fd248c4db10c8862499532b2bd7eab8))


## v0.34.5 (2026-03-05)

### Bug Fixes

- Use correct CDN path for marked.js UMD build
  ([`15f0964`](https://github.com/jason-weddington/personal-kb-mcp/commit/15f0964e3b78e0a6e1f385533d6ccdecd844daba))


## v0.34.4 (2026-03-05)

### Bug Fixes

- Prevent silent failures in response panel rendering
  ([`ee2b1b7`](https://github.com/jason-weddington/personal-kb-mcp/commit/ee2b1b7b091d6685d465eb99b1f58e640cfb06d0))


## v0.34.3 (2026-03-05)

### Bug Fixes

- Exclude deactivated entries from graph visualization
  ([`d6bff53`](https://github.com/jason-weddington/personal-kb-mcp/commit/d6bff531be8f6e86fda68192768d28ec5e03274f))


## v0.34.2 (2026-03-05)

### Bug Fixes

- Include exception detail in SSE error events
  ([`05b15f0`](https://github.com/jason-weddington/personal-kb-mcp/commit/05b15f077dca61244574d9f527539cba12793c5a))


## v0.34.1 (2026-03-05)

### Bug Fixes

- Handle query task errors in SSE stream gracefully
  ([`a083758`](https://github.com/jason-weddington/personal-kb-mcp/commit/a08375838c8a616b82b07a8583f67793b2d5bdc8))


## v0.34.0 (2026-03-05)

### Documentation

- Add graph explorer to README and how_it_works.md
  ([`0f3e205`](https://github.com/jason-weddington/personal-kb-mcp/commit/0f3e205f06495a99ea7b141ddd41c2799a65d7f9))

### Features

- Render markdown in explorer response panel
  ([`a95e778`](https://github.com/jason-weddington/personal-kb-mcp/commit/a95e778516478bdbc789cff13ecd39253f234a11))


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
