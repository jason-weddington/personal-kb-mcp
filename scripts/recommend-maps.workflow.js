export const meta = {
  name: 'recommend-maps',
  description: 'Study a repo, find its component seams, and recommend the mental_map orientation nodes it needs',
  phases: [
    { title: 'Discover seams', detail: 'multi-lens parallel survey of the repo' },
    { title: 'Canonicalize seams', detail: 'dedup candidate seams into one canonical list' },
    { title: 'Recommend maps', detail: 'one grounded map recommendation per seam' },
    { title: 'Synthesize', detail: 'final ordered map list + system map + doc-debt' },
  ],
}

const ARGS = typeof args === 'string' ? JSON.parse(args) : (args || {})
const REPO = ARGS.repoPath || '.'
const PROJECT = ARGS.projectRef || ''

const MAP_DESIGN = [
  'WHAT A MENTAL MAP IS (what you are recommending):',
  'A `mental_map` is a fact-free ORIENTATION node for ONE component/subsystem — the "directory tier" of the KB. Its job: tell an agent WHAT a subsystem is, WHERE its code lives, and WHICH KB entries explain it, so the agent orients without re-reading everything.',
  'HARD RULES for a well-formed map (so the recommendation is actually buildable):',
  '- Fact-free: orientation prose + POINTERS only. NO retrievable values (ports, thresholds, config values, code signatures) — those belong in the factual_reference entries the map POINTS TO.',
  '- Must have >=1 outbound pointer to an EXISTING KB entry (a real `kb-XXXXX` id). Zero-pointer maps are rejected on CREATE by both the HTTP service and the MCP client, and (as of 2026-09-19) on UPDATE too for the hosted service — only the local/no-auth kb-core path is still unguarded. Do not rely on the store as your safety net anyway: the update guarantee is new, older maps were authored without it, and a map you propose should stand up without it. So every recommended map MUST name the real kb-XXXXX entries it would point to, and if you cannot name a real, substantive (chunky) detail entry to point at, do NOT propose the map — put the missing fact in documentation_gaps instead so a future pass can author it.',
  '- Scoped to a single project_ref; surfaced via kb_preflight + the push hook.',
  '- A map MAY also point to other maps (system->component nesting) and to ordinary detail entries (leaves).',
  'A GOOD recommendation names: the seam, a one-line orientation, the code paths it covers, the EXISTING kb-XXXXX entries it can point to NOW, and the GAPS (subsystem facts with no KB entry yet that should be authored to give the map substance).',
].join('\n')

const CONTEXT = [
  `REPO (read the actual code here): ${REPO}`,
  `KB project_ref for this repo: "${PROJECT}". Use kb_search(project_ref="${PROJECT}") to find pointer-target entries; also kb_search by keyword for related entries that may live under other project_refs.`,
  'GROUND EVERYTHING. Read code at REPO; find pointer targets via real kb_search calls. NEVER invent a kb-XXXXX id — cite only ids that appeared in a real kb_search result.',
  MAP_DESIGN,
].join('\n\n')

const SEAM_CAND = {
  type: 'object', additionalProperties: false,
  required: ['lens', 'seams'],
  properties: {
    lens: { type: 'string' },
    seams: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['name', 'responsibility', 'key_paths', 'why_seam'],
      properties: {
        name: { type: 'string' },
        responsibility: { type: 'string' },
        key_paths: { type: 'array', items: { type: 'string' } },
        why_seam: { type: 'string', description: 'why this is a distinct subsystem boundary' },
      },
    } },
  },
}

const CANONICAL = {
  type: 'object', additionalProperties: false,
  required: ['seams', 'system_map_warranted', 'notes'],
  properties: {
    system_map_warranted: { type: 'boolean' },
    notes: { type: 'string' },
    seams: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['name', 'responsibility', 'key_paths', 'tier'],
      properties: {
        name: { type: 'string' },
        responsibility: { type: 'string' },
        key_paths: { type: 'array', items: { type: 'string' } },
        tier: { type: 'string', enum: ['system', 'component'] },
      },
    } },
  },
}

const MAP_REC = {
  type: 'object', additionalProperties: false,
  required: ['seam', 'build_now', 'short_title', 'long_title', 'scope_paths', 'pointer_entries', 'gaps', 'value', 'tier'],
  properties: {
    seam: { type: 'string' },
    build_now: { type: 'boolean', description: 'true ONLY if >=1 real pointer entry was found' },
    short_title: { type: 'string' },
    long_title: { type: 'string', description: 'one-line fact-free orientation (the directory entry an agent reads)' },
    scope_paths: { type: 'array', items: { type: 'string' } },
    pointer_entries: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['id', 'why'],
      properties: { id: { type: 'string', description: 'real kb-XXXXX from a kb_search result' }, why: { type: 'string' } },
    } },
    gaps: { type: 'array', items: { type: 'string' }, description: 'subsystem facts with NO KB entry yet that should be authored' },
    value: { type: 'string' },
    tier: { type: 'string', enum: ['system', 'component'] },
  },
}

const FINAL = {
  type: 'object', additionalProperties: false,
  required: ['summary', 'system_map', 'component_maps', 'documentation_gaps'],
  properties: {
    summary: { type: 'string' },
    system_map: {
      type: 'object', additionalProperties: false,
      required: ['recommended', 'short_title', 'long_title', 'points_to'],
      properties: {
        recommended: { type: 'boolean' },
        short_title: { type: 'string' },
        long_title: { type: 'string' },
        points_to: { type: 'array', items: { type: 'string' }, description: 'the component maps / key entries it orients' },
      },
    },
    component_maps: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['short_title', 'long_title', 'scope_paths', 'pointer_entries', 'gaps', 'priority', 'build_now'],
      properties: {
        short_title: { type: 'string' },
        long_title: { type: 'string' },
        scope_paths: { type: 'array', items: { type: 'string' } },
        pointer_entries: { type: 'array', items: { type: 'string' }, description: 'real kb-XXXXX ids' },
        gaps: { type: 'array', items: { type: 'string' } },
        priority: { type: 'string', enum: ['high', 'medium', 'low'] },
        build_now: { type: 'boolean' },
      },
    } },
    documentation_gaps: { type: 'array', items: { type: 'string' }, description: 'detail entries that should be authored so thin-pointer maps become buildable' },
  },
}

// ---- Phase 1: multi-lens seam discovery (parallel, read-only) ----
phase('Discover seams')
const LENSES = [
  { key: 'structure', title: 'Package structure', brief: 'Walk the src/ package tree. For each package/module, name its single responsibility and where one subsystem ends and the next begins. Report cohesive subsystems with a clear boundary.' },
  { key: 'dataflow', title: 'Runtime data flows', brief: 'Trace the primary flows end-to-end: the store/write path, the search/query path, the ingestion pipeline, MCP tool dispatch, decay/confidence computation, graph edge building. Report the components each flow crosses as seams.' },
  { key: 'surface', title: 'Capability surface', brief: 'Inventory the MCP tool surface (tools/), the web/explorer, and CLI entrypoints. For each user-facing capability, name the subsystems it leans on. Report capability-aligned seams.' },
  { key: 'crosscut', title: 'Cross-cutting infra', brief: 'Find cross-cutting concerns / infrastructure spanning subsystems: config, db backends (sqlite/postgres), LLM providers (anthropic/bedrock/ollama), graph, confidence/decay, safety/secrets, the eval framework, the mental_map feature, the hook package. These often deserve their own maps.' },
]
const candidates = await parallel(LENSES.map((L) => () =>
  agent(
    `${CONTEXT}\n\nLENS: ${L.title}\n${L.brief}\n\nSurvey ${REPO} through THIS lens ONLY and return candidate component seams. Be concrete — name real paths under ${REPO}.`,
    { label: `seam:${L.key}`, phase: 'Discover seams', schema: SEAM_CAND, agentType: 'Explore' },
  )
))

// ---- Phase 2: canonicalize (barrier — need all lenses to dedup) ----
phase('Canonicalize seams')
const allCand = candidates.filter(Boolean).flatMap((c) => (c.seams || []).map((s) => ({ ...s, lens: c.lens })))
const canon = await agent(
  `${CONTEXT}\n\nCandidate seams from ${LENSES.length} lenses (overlapping/duplicated):\n${JSON.stringify(allCand, null, 2)}\n\nCluster these into a CANONICAL, de-duplicated list of component seams — each canonical seam = one candidate mental_map. Merge duplicates; split anything that's really two subsystems; drop non-seams. Assign tier (system = whole-repo umbrella; component = a subsystem). Aim for ~8-14 component seams. Decide whether a single system-level map is warranted. Verify boundaries against ${REPO} where unsure.`,
  { label: 'canonicalize', phase: 'Canonicalize seams', schema: CANONICAL, agentType: 'Explore' },
)

// ---- Phase 3: one grounded map recommendation per seam (parallel) ----
phase('Recommend maps')
const recs = await parallel((canon.seams || []).map((s) => () =>
  agent(
    `${CONTEXT}\n\nSEAM: ${s.name}\nResponsibility: ${s.responsibility}\nKey paths: ${(s.key_paths || []).join(', ')}\n\nProduce a RECOMMENDED mental_map for this seam:\n1. Confirm the boundary by reading the key files at ${REPO}.\n2. kb_search the KB (project_ref="${PROJECT}" and by keyword) for entries this map would POINT TO. Cite only real kb-XXXXX ids that appeared in results, each with a one-line why.\n3. Draft: short_title; long_title (a fact-free one-line orientation); scope_paths; pointer_entries (existing kb-XXXXX); gaps (subsystem facts with NO KB entry yet that should be authored); value (why this map helps an agent orient); tier.\nSet build_now=true ONLY if you found >=1 real pointer entry; otherwise false (and gaps must explain what to author first).`,
    { label: `map:${(s.name || '').slice(0, 40)}`, phase: 'Recommend maps', schema: MAP_REC, agentType: 'Explore' },
  )
))

// ---- Phase 4: synthesize the final recommended map set ----
phase('Synthesize')
const final = await agent(
  `${CONTEXT}\n\nIndividual map recommendations:\n${JSON.stringify(recs.filter(Boolean), null, 2)}\n\nsystem_map_warranted (from canonicalize): ${canon.system_map_warranted}\n\nSynthesize the FINAL recommended map set:\n- Resolve overlap; each map a distinct, coherent orientation node.\n- If a single SYSTEM map (umbrella pointing at the component maps + key entries) is warranted, draft it; else set recommended=false.\n- Order component maps by priority (most orientation-per-token first).\n- For each, list its real pointer kb-XXXXX ids and set build_now (has real pointers) vs not.\n- documentation_gaps = the detail entries that should be authored so thin-pointer maps become buildable.\nKeep every map fact-free with concrete pointer ids.`,
  { label: 'synthesize', phase: 'Synthesize', schema: FINAL },
)

return { final, recommendations: recs.filter(Boolean), seams: canon.seams, system_map_warranted: canon.system_map_warranted }
