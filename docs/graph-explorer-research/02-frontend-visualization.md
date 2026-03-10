# Graph Visualization & Frontend Architecture

**Researcher**: frontend-researcher
**Date**: 2026-03-05

---

## Library Comparison

### force-graph (vanilla) + 3d-force-graph — RECOMMENDED

- **Renderer**: 2D Canvas (force-graph) / 3D WebGL via Three.js (3d-force-graph)
- **Performance**: Canvas variant smooth at 5k nodes/30fps; WebGL variant at 7k nodes/30fps
- **Built-in traversal animation**: YES — `linkDirectionalParticles`, `linkDirectionalParticleSpeed`, `linkDirectionalParticleColor`, `emitParticle()` (fire single particles on demand). This is the killer feature — particles visually travel from source to target along edges, exactly the "query traversal" effect we want.
- **Custom rendering**: `nodeCanvasObject`, `linkCanvasObject`, `onRenderFramePre/Post` callbacks — full control over node/link appearance per frame
- **Animation control**: `pauseAnimation()`, `resumeAnimation()`, `zoomToFit()`, `centerAt()` with transition durations
- **Framework agnostic**: Vanilla JS, zero framework dependency. React wrapper exists (`react-force-graph`) but the vanilla version is leaner.
- **Author**: Vasco Asturiano — extremely well-maintained, consistent API across 2D/3D/VR/AR variants

### Sigma.js v3 + Graphology

- **Renderer**: WebGL (fastest raw rendering)
- **Performance**: Best for large static graphs (10k+ nodes)
- **Animation API**: Limited — no built-in traversal animation. Would need custom WebGL shader programs for glow/particle effects.
- **Weakness**: Custom animation requires writing WebGL programs. No built-in particle system.

### Cytoscape.js

- **Renderer**: Canvas
- **Performance**: Good up to ~3k nodes, slows beyond that
- **Animation API**: Rich — `ele.animation()`, `cy.animate()`, built-in BFS/DFS traversal with animated demo. `eles.bfs()` returns ordered nodes for step-by-step highlighting.
- **Weakness**: Canvas-only (lower ceiling). Heavier bundle (~500KB). Animation is step-based, not particle-flow-based.

### Others

- **Reagraph**: WebGL, React-only, younger project, less flexibility
- **vis.js**: Slowest of the bunch — not recommended
- **D3-force (raw)**: Ultimate flexibility but DIY everything. force-graph is D3-force with batteries included.

### Summary Matrix

| Library | Renderer | 30fps Ceiling | Particle Animation | Custom Rendering | Bundle |
|---------|----------|--------------|-------------------|-----------------|--------|
| force-graph | Canvas | ~5k nodes | YES (built-in) | YES (callbacks) | ~50KB |
| 3d-force-graph | WebGL/Three.js | ~7k nodes | YES (built-in) | YES (Three.js) | ~200KB |
| Sigma.js v3 | WebGL | ~10k+ nodes | No (custom shader) | Yes (GLSL) | ~80KB |
| Cytoscape.js | Canvas | ~3k nodes | No (step animation) | Limited | ~500KB |
| Reagraph | WebGL | ~5k nodes | No | Limited | ~150KB |

**Verdict**: `force-graph` (2D) is the clear winner. Built-in directional particles + `emitParticle()` give us the "Hollywood traversal" effect with zero custom shader work. For 100-500 entries (500-2000 total nodes), well within the 5k comfort zone.

---

## Traversal Animation Architecture

The key insight: **force-graph's `emitParticle(link)` method fires a single particle along a specific link on demand**. This maps perfectly to backend traversal events.

### Data flow

```
Backend query engine
  → emits traversal events via SSE
  → Frontend receives event
  → Highlights node (change color/size via nodeCanvasObject)
  → Calls emitParticle(link) on the traversed edge
  → Particle visually flows along the edge
```

### Node state machine (per frame)

- `idle` → default color/size
- `visiting` → glow effect (larger radius + additive color via `nodeCanvasObject`)
- `result` → permanent highlight (different color)
- `dimmed` → everything NOT in the result set fades after query completes

### Implementation sketch

```javascript
function onTraversalEvent(event) {
  if (event.type === 'visit_node') {
    nodeStates.set(event.id, 'visiting');
    graph.centerAt(nodePositions[event.id].x, nodePositions[event.id].y, 300);
  }
  if (event.type === 'traverse_edge') {
    const link = links.find(l => l.source.id === event.source && l.target.id === event.target);
    graph.emitParticle(link); // Built-in particle animation!
  }
}

// Custom node rendering with glow:
graph.nodeCanvasObject((node, ctx, globalScale) => {
  const state = nodeStates.get(node.id) || 'idle';
  if (state === 'visiting') {
    ctx.shadowColor = nodeColors[node.type];
    ctx.shadowBlur = 15;
    ctx.beginPath();
    ctx.arc(node.x, node.y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = nodeColors[node.type];
    ctx.fill();
    ctx.shadowBlur = 0;
  }
});
```

---

## SPA Framework Choice

### Recommendation: Svelte 5 (with vanilla force-graph)

**Why Svelte over React:**
- 2.5x smaller bundles (~1.6KB vs ~42KB for React)
- 39% faster on benchmarks (compiles to direct DOM manipulation)
- Surgical DOM updates — perfect for split-panel (chat updates frequently, graph canvas doesn't need framework re-renders)
- No React wrapper overhead — force-graph is vanilla JS. In Svelte, just mount in `onMount()` and call methods directly.

**Why NOT React:**
- react-force-graph adds abstraction over a vanilla canvas library — unnecessary indirection
- React's re-render model fights against imperative canvas updates
- Larger bundle for no benefit

**Why NOT vanilla JS:**
- Chat panel needs reactive state management
- Split-panel layout with resizing needs component structure
- Svelte gives reactivity for the chrome while staying out of the way for canvas

### Architecture

```
src/
  App.svelte               — split-panel layout
  lib/
    ChatPanel.svelte       — LLM chat with streaming
    GraphPanel.svelte      — force-graph mount + traversal animation
    graph/
      GraphController.ts   — imperative graph API
      nodeStyles.ts        — color/size/glow per node type
      eventHandler.ts      — SSE event → graph animation mapping
    chat/
      streamHandler.ts     — SSE streaming for chat responses
```

---

## Streaming: SSE (Server-Sent Events)

**Why SSE over WebSocket:**
- Flow is server→client during queries (client sends one request, server streams back)
- ChatGPT, Claude, and virtually all LLM chat UIs use SSE
- Simpler server implementation, auto-reconnects, works through proxies
- No bidirectional need

### Event stream format (single SSE connection per query)

```
event: chat_token
data: {"token": "The"}

event: graph_visit
data: {"node": "kb-00042", "score": 0.85}

event: graph_traverse
data: {"source": "kb-00042", "target": "tag:python", "edge_type": "tagged"}

event: chat_token
data: {"token": " relevant"}

event: graph_result
data: {"entries": ["kb-00042", "kb-00103"]}

event: done
data: {}
```

Chat tokens and graph events are **interleaved** — the graph lights up AS the answer streams. This is the "wow factor."

---

## Visual Design

### Node type color coding

| Node Type | Color | Notes |
|-----------|-------|-------|
| entry (kb-XXXXX) | White/bright | Circle (default) |
| tag:X | Cyan/teal | Small circle |
| project:X | Orange | Diamond or larger circle |
| person:X | Yellow | Circle with ring |
| tool:X | Green | Hexagon |
| concept:X | Purple | Circle |
| technology:X | Blue | Circle |

### Edge styling

- `tagged` → thin, gray, dashed
- `belongs_to` → medium, orange
- `uses` / `depends_on` → medium, directional arrow
- `supersedes` → thick, red, directional (shows evolution)
- `extracted_from` → thin, green (note → entries)
- LLM-enriched → slight glow to distinguish from deterministic

### Layout & clustering

- Force-directed with collision detection to prevent overlap
- Increase charge between different node types → natural clustering
- Entry nodes heavier (higher mass) → anchor; entity nodes orbit
- `forceCluster` pulls same-project entries together

### Interaction

- Hover: tooltip with title/tags/score
- Click: entry details in side panel
- Double-click: expand neighbors from API
- Zoom: scroll wheel, pinch
- Pan: drag background
- Minimap: small overview in corner

---

## Performance Analysis

For personal-kb's scale:

| Metric | Expected Range | Library Comfort Zone |
|--------|---------------|---------------------|
| Nodes | 500-2000 | force-graph: 5000 @ 30fps |
| Edges | 1000-5000 | force-graph: 5000 @ 30fps |
| Particles/frame | 5-20 | force-graph: 100s supported |

Well within safe margins. Even at 2000 nodes, force-graph on Canvas runs at 60fps.

Benchmarks (PMC study 2025):
- SVG: usable to ~2k nodes
- Canvas (force-graph): usable to ~5k nodes at 30fps
- WebGL (Sigma/3d-force-graph): usable to ~7-10k nodes at 30fps
- Beyond 10k: need GPU-accelerated layout + LOD rendering

---

## "Wow Factor" Features by Effort Tier

### Tier 1 — Built into force-graph, trivial to enable
- **Directional particles on edges**: `linkDirectionalParticles(3)`
- **Particle burst on traversal**: `emitParticle(link)`
- **Zoom-to-fit on query**: `zoomToFit(400, 50, node => resultSet.has(node.id))`
- **Center on active node**: `centerAt(x, y, 300)`

### Tier 2 — Custom nodeCanvasObject, moderate effort
- **Glow effect**: Canvas `shadowBlur` + `shadowColor`
- **Pulse animation**: oscillating radius/opacity via requestAnimationFrame
- **Score-based sizing**: nodes sized proportional to relevance
- **Fade-in on discovery**: new nodes animate from transparent to opaque
- **Edge color transitions**: edges change color as particles pass through

### Tier 3 — Advanced, significant effort
- **Bloom/HDR glow**: post-processing (offscreen canvas, gaussian blur, composite)
- **Trail effects**: particles leave fading trails (custom `linkDirectionalParticleCanvasObject`)
- **Ripple on node visit**: expanding concentric circles
- **Constellation mode**: dim everything except query path
- **3D mode toggle**: switch between 2D and 3D for dramatic effect

### "Hollywood Hacker" Recipe

1. Dark background (`#0a0a0f`)
2. Nodes as bright dots with type-coded colors
3. Edges as thin semi-transparent lines
4. On query: all nodes dim to 20% opacity
5. Visited nodes light up with glow (`shadowBlur=15`), camera pans smoothly
6. Particles burst along traversed edges (`emitParticle`)
7. Result nodes pulse with oscillating glow
8. Non-result nodes fade back to 40%
9. Status overlay: "Searching... 3 nodes visited, 2 graph hops"

---

## Prior Art

- **GraphRAG Workbench** (Microsoft): ChatPanel + GraphVisualizer split layout using 3d-force-graph. Closest to our vision.
- **Neo4j Bloom**: GPU-accelerated, 100k+ nodes. "Slicer" animation for temporal data. Commercial.
- **Obsidian graph view**: Force-directed (d3-force), 3D plugin uses `3d-force-graph`.
- **Whyis Knowledge Explorer**: Progressive exploration — start from one node, expand on demand.
