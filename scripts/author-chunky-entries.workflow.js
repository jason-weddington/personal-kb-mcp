export const meta = {
  name: 'author-chunky-entries',
  description: 'Author grounded factual_reference detail entries for under-documented subsystems, so mental_maps have real targets to point at',
  phases: [
    { title: 'Draft', detail: 'read each subsystem, write a dense grounded factual_reference + slim map skeleton' },
    { title: 'Verify', detail: 'ground each draft against the code; flag phantom signatures / wrong values' },
    { title: 'Finalize', detail: 'fold corrections into the final entry + map' },
  ],
}

const ARGS = typeof args === 'string' ? JSON.parse(args) : (args || {})
const REPO = ARGS.repoPath
const PROJECT = ARGS.projectRef
const SUBSYSTEMS = ARGS.subsystems || []

const CONTEXT = [
  `REPO (read the actual code here): ${REPO}`,
  `KB project_ref: "${PROJECT}".`,
  'You are authoring a CHUNKY detail entry — the OPPOSITE of a mental_map. It must be DENSE and SPECIFIC: exact table/column names, enum members and their values, numeric bounds, env-var names, function/route signatures, default-resolution order. These retrievable values are the whole point — a map will point HERE for them. Ground every claim by reading the cited files. Never invent a signature or value; if you cannot confirm something, omit it or mark it open.',
].join('\n\n')

const DRAFT = {
  type: 'object', additionalProperties: false,
  required: ['subsystem', 'entry', 'map', 'grounding_notes'],
  properties: {
    subsystem: { type: 'string' },
    entry: {
      type: 'object', additionalProperties: false,
      required: ['short_title', 'long_title', 'knowledge_details', 'tags'],
      properties: {
        short_title: { type: 'string' },
        long_title: { type: 'string' },
        knowledge_details: { type: 'string', description: 'dense, retrievable-value-rich factual_reference body grounded in the code' },
        tags: { type: 'array', items: { type: 'string' } },
      },
    },
    map: {
      type: 'object', additionalProperties: false,
      required: ['short_title', 'long_title', 'orientation', 'lives_in', 'points_to_existing', 'not_yet_documented'],
      properties: {
        short_title: { type: 'string' },
        long_title: { type: 'string' },
        orientation: { type: 'string', description: '2-3 sentences fact-free WHAT the subsystem is' },
        lives_in: { type: 'string', description: 'one coarse package/dir location phrase, directory-level not .py files' },
        points_to_existing: { type: 'array', items: { type: 'string' }, description: 'real kb-XXXXX ids (from kb_search) the map can ALSO point at, beyond the new entry — may be empty' },
        not_yet_documented: { type: 'string', description: 'short prose naming what still lacks a chunky entry' },
      },
    },
    grounding_notes: { type: 'string' },
  },
}

const VERDICT = {
  type: 'object', additionalProperties: false,
  required: ['accurate', 'corrections', 'density_ok'],
  properties: {
    accurate: { type: 'boolean', description: 'true if every value/signature in the entry was confirmed in the code' },
    density_ok: { type: 'boolean', description: 'true if the entry is genuinely chunky (retrievable values present), not map-like prose' },
    corrections: { type: 'array', items: { type: 'string' }, description: 'specific fixes: wrong value, phantom signature, missing key detail' },
  },
}

const FINAL = {
  type: 'object', additionalProperties: false,
  required: ['subsystem', 'entry', 'map'],
  properties: {
    subsystem: { type: 'string' },
    entry: {
      type: 'object', additionalProperties: false,
      required: ['short_title', 'long_title', 'knowledge_details', 'tags'],
      properties: {
        short_title: { type: 'string' },
        long_title: { type: 'string' },
        knowledge_details: { type: 'string' },
        tags: { type: 'array', items: { type: 'string' } },
      },
    },
    map: {
      type: 'object', additionalProperties: false,
      required: ['short_title', 'long_title', 'orientation', 'lives_in', 'points_to_existing', 'not_yet_documented'],
      properties: {
        short_title: { type: 'string' },
        long_title: { type: 'string' },
        orientation: { type: 'string' },
        lives_in: { type: 'string' },
        points_to_existing: { type: 'array', items: { type: 'string' } },
        not_yet_documented: { type: 'string' },
      },
    },
  },
}

const finals = await pipeline(
  SUBSYSTEMS,
  (s) => agent(
    `${CONTEXT}\n\nSUBSYSTEM: ${s.name}\nResponsibility: ${s.responsibility}\nScope files (read these): ${(s.scope_paths || []).join(', ')}\nDetail to capture (the gaps a map needs filled): ${(s.gaps || []).join(' | ')}\n\nRead the scope files at ${REPO}. Write ONE dense factual_reference entry capturing the listed detail with exact values. Also draft the slim mental_map skeleton that will POINT to this entry (fact-free orientation + coarse lives_in + any existing kb-XXXXX it can also point at via kb_search(project_ref="${PROJECT}") + a short not_yet_documented line).`,
    { label: `draft:${s.key}`, phase: 'Draft', schema: DRAFT, agentType: 'Explore' },
  ),
  (draft, s) => agent(
    `${CONTEXT}\n\nVERIFY this drafted detail entry for subsystem "${draft.subsystem}" by reading the code at ${REPO} (${(s.scope_paths || []).join(', ')}).\n\nDRAFT ENTRY:\n${JSON.stringify(draft.entry, null, 2)}\n\nConfirm EVERY value, column name, enum member, bound, env-var, and signature against the actual code. Flag anything phantom, wrong, or stale. Confirm it is genuinely dense (chunky), not map-like. Return verdict + specific corrections.`,
    { label: `verify:${s.key}`, phase: 'Verify', schema: VERDICT, agentType: 'Explore' },
  ).then((v) => ({ draft, verdict: v, subsystem: s })),
  (checked) => {
    const { draft, verdict, subsystem } = checked
    if (verdict.accurate && verdict.density_ok && (verdict.corrections || []).length === 0) {
      return { subsystem: draft.subsystem, entry: draft.entry, map: draft.map }
    }
    return agent(
      `${CONTEXT}\n\nFINALIZE the detail entry + map for "${draft.subsystem}". Re-read the code at ${REPO} (${(subsystem.scope_paths || []).join(', ')}) for any contested point and fold in these verifier corrections:\n${JSON.stringify(verdict.corrections, null, 2)}\n\nDRAFT:\n${JSON.stringify(draft, null, 2)}\n\nReturn the corrected final entry + map. Keep the entry dense and the map fact-free.`,
      { label: `final:${subsystem.key}`, phase: 'Finalize', schema: FINAL, agentType: 'Explore' },
    )
  },
)

return { finals: finals.filter(Boolean) }
