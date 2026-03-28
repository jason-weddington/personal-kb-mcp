// GRAPH_DATA and NODE_COLORS are injected by the HTML template
const HIGH_CONN_THRESHOLD = 5;
const LABEL_ZOOM_MIN = 3.0;
const LABEL_ZOOM_MAX = 5.0;
const LABEL_ZOOM_SOLO_MIN = 1.5;
const LABEL_ZOOM_SOLO_MAX = 3.0;
const LABEL_TRUNCATE_SCALE = 2.0;
const LABEL_TRUNCATE_LEN = 20;
const isServed = location.protocol !== 'file:';

// Auto-resize textarea to fit content (up to CSS max-height)
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = el.scrollHeight + 'px';
  // Switch to scroll if we hit the cap
  el.style.overflow = el.scrollHeight > el.offsetHeight ? 'auto' : 'hidden';
}

// Icon SVGs
const copySvg = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
  + 'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
  + 'stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" '
  + 'rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 '
  + '2v1"/></svg>';
const expandSvg = '<svg width="14" height="14" viewBox="0 0 24 24" '
  + 'fill="none" stroke="currentColor" stroke-width="2" '
  + 'stroke-linecap="round" stroke-linejoin="round">'
  + '<polyline points="15 3 21 3 21 9"/>'
  + '<polyline points="9 21 3 21 3 15"/>'
  + '<line x1="21" y1="3" x2="14" y2="10"/>'
  + '<line x1="3" y1="21" x2="10" y2="14"/></svg>';
const shrinkSvg = '<svg width="14" height="14" viewBox="0 0 24 24" '
  + 'fill="none" stroke="currentColor" stroke-width="2" '
  + 'stroke-linecap="round" stroke-linejoin="round">'
  + '<polyline points="4 14 10 14 10 20"/>'
  + '<polyline points="20 10 14 10 14 4"/>'
  + '<line x1="14" y1="10" x2="21" y2="3"/>'
  + '<line x1="3" y1="21" x2="10" y2="14"/></svg>';

// Initialize maximize button icon
document.getElementById('chat-maximize-btn').innerHTML = expandSvg;

// Copy message text to clipboard
function copyMsgText(btn) {
  const msg = btn.closest('.chat-msg');
  const clone = msg.cloneNode(true);
  const cb = clone.querySelector('.copy-btn');
  if (cb) cb.remove();
  navigator.clipboard.writeText(clone.innerText).then(() => {
    btn.innerHTML = '\u2713';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.innerHTML = copySvg;
      btn.classList.remove('copied');
    }, 1500);
  });
}

// Maximize / restore chat panel
let chatMaximized = false;
function toggleMaximize() {
  chatMaximized = !chatMaximized;
  chatPanel.classList.toggle('maximized', chatMaximized);
  document.getElementById('chat-maximize-btn').innerHTML =
    chatMaximized ? shrinkSvg : expandSvg;
}

// Query-driven traversal state
let queryMode = null;       // 'explore' | 'summarize' | null
const visitedNodes = new Set();
const resultNodes = new Set();
const nodeStates = new Map();  // node_id -> 'visited' | 'result'
const pulseNodes = new Map();  // node_id -> start_time (ms)
const PULSE_DURATION = 600;

function triggerPulse(nodeId) {
  pulseNodes.set(nodeId, performance.now());
}

function getPulseRadius(nodeId) {
  const start = pulseNodes.get(nodeId);
  if (!start) return 0;
  const elapsed = performance.now() - start;
  if (elapsed > PULSE_DURATION) {
    pulseNodes.delete(nodeId);
    return 0;
  }
  const t = elapsed / PULSE_DURATION;
  return (1 - t) * 6;
}

function isActiveLink(link) {
  const s = link.source.id || link.source;
  const t = link.target.id || link.target;
  const sA = visitedNodes.has(s) || resultNodes.has(s);
  const tA = visitedNodes.has(t) || resultNodes.has(t);
  return { either: sA || tA, both: sA && tA };
}

// Build legend (clickable for solo mode)
const legend = document.getElementById('legend');
const legendItems = new Map();
Object.entries(NODE_COLORS).forEach(([type, color]) => {
  const item = document.createElement('span');
  item.className = 'legend-item';
  item.innerHTML = `<span class="legend-dot" style="background:${color}"></span>${type}`;
  item.addEventListener('click', () => toggleSolo(type));
  legend.appendChild(item);
  legendItems.set(type, item);
});

// Stats bar
document.getElementById('stats-bar').textContent =
  `${GRAPH_DATA.stats.node_count} nodes \u00b7 ${GRAPH_DATA.stats.edge_count} edges`;

// Build adjacency index for info panel
const adjacency = new Map();
GRAPH_DATA.edges.forEach(e => {
  if (!adjacency.has(e.source)) adjacency.set(e.source, []);
  if (!adjacency.has(e.target)) adjacency.set(e.target, []);
  adjacency.get(e.source).push({ node: e.target, type: e.type, dir: 'out' });
  adjacency.get(e.target).push({ node: e.source, type: e.type, dir: 'in' });
});

const connected = n => adjacency.has(n.id);
let hoveredNode = null;
let highlightedNode = null;
const navHistory = [];
let soloType = null;
let focusedNode = null;

function nodeVisible(node) {
  if (focusedNode) {
    if (node.id === focusedNode) return true;
    const nb = adjacency.get(focusedNode) || [];
    return nb.some(n => n.node === node.id);
  }
  if (!soloType) return true;
  return node.type === soloType;
}

function linkVisible(link) {
  if (focusedNode) {
    const s = link.source.id || link.source;
    const t = link.target.id || link.target;
    return s === focusedNode || t === focusedNode;
  }
  if (!soloType) return true;
  return false;
}

function selectNode(node) {
  const ctr = graph.centerAt();
  navHistory.push({
    node, zoom: graph.zoom(), cx: ctr.x, cy: ctr.y,
    focused: focusedNode
  });
  focusedNode = node.id;
  highlightedNode = node;
  graph.nodeVisibility(nodeVisible)
    .linkVisibility(linkVisible);
  graph.centerAt(node.x, node.y, 800);
  graph.zoom(4, 800);
  setTimeout(() => showInfoPanel(node), 850);
}

function computeLabelOpacity(node, globalScale) {
  if (node === hoveredNode) return 1.0;
  if (node === highlightedNode) return 1.0;
  const nState = nodeStates.get(node.id);
  if (nState === 'result' || nState === 'visited') return 1.0;
  if (queryMode !== null) return 0;
  if (focusedNode) {
    if (node.id === focusedNode) return 1.0;
    const nb = adjacency.get(focusedNode) || [];
    if (nb.some(n => n.node === node.id)) return 1.0;
  }
  const useSolo = soloType !== null;
  const zMin = useSolo ? LABEL_ZOOM_SOLO_MIN : LABEL_ZOOM_MIN;
  const zMax = useSolo ? LABEL_ZOOM_SOLO_MAX : LABEL_ZOOM_MAX;
  const t = Math.min(Math.max((globalScale - zMin) / (zMax - zMin), 0.0), 1.0);
  return t;
}

function truncateLabel(text, globalScale) {
  if (globalScale < LABEL_TRUNCATE_SCALE && text.length > LABEL_TRUNCATE_LEN) {
    return text.slice(0, LABEL_TRUNCATE_LEN - 1) + '\u2026';
  }
  return text;
}

const MAX_AUTO_ZOOM = 6;

const graph = ForceGraph()(document.getElementById('graph'))
  .graphData({ nodes: GRAPH_DATA.nodes, links: GRAPH_DATA.edges.map(e => ({
    source: e.source, target: e.target, type: e.type
  }))})
  .backgroundColor('#0a0a0f')
  .maxZoom(12)
  .nodeId('id')
  .nodeVal('val')
  .nodeColor(n => NODE_COLORS[n.type] || '#666')
  .nodeCanvasObject((node, ctx, globalScale) => {
    const isHL = node === highlightedNode;
    const nState = nodeStates.get(node.id);
    const isResult = nState === 'result';
    const isVisited = nState === 'visited';
    const isActive = isResult || isVisited || isHL;
    // 1-hop neighbors of active nodes get their original color back
    let isNeighbor = false;
    if (queryMode !== null && !isActive) {
      const nb = adjacency.get(node.id);
      if (nb) isNeighbor = nb.some(n =>
        visitedNodes.has(n.node) || resultNodes.has(n.node));
    }
    const isDimmed = queryMode !== null && !isActive && !isNeighbor;
    const pulseR = getPulseRadius(node.id);
    const extraR = isResult ? 3 : (isHL ? 6 : 0);
    const r = Math.sqrt(node.val || 1) * 2.5 + extraR + pulseR;
    let color = NODE_COLORS[node.type] || '#666';

    // Traversal state overrides
    if (isResult) color = '#00ff88';
    else if (isVisited) color = '#ffaa00';

    if (isActive) {
      ctx.shadowColor = color;
      ctx.shadowBlur = isResult ? 30 : (isVisited ? 20 : 25);
    }
    if (isDimmed) ctx.globalAlpha = 0.15;

    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = isDimmed ? '#555' : color;
    ctx.fill();
    ctx.shadowBlur = 0;

    if (isDimmed) { ctx.globalAlpha = 1.0; return; }

    // Zoom-aware label rendering
    const labelOpacity = computeLabelOpacity(node, globalScale);
    if (labelOpacity >= 0.01) {
      const fontSize = Math.max(10 / globalScale, 2);
      const label = truncateLabel(node.label || node.id, globalScale);
      ctx.font = `${fontSize}px system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const textW = ctx.measureText(label).width;
      const padX = 3 / globalScale;
      const padY = 1.5 / globalScale;
      const pillY = node.y + r + 2;
      ctx.globalAlpha = labelOpacity * 0.6;
      ctx.fillStyle = '#0a0a0f';
      ctx.beginPath();
      ctx.roundRect(node.x - textW / 2 - padX, pillY - padY,
                    textW + padX * 2, fontSize + padY * 2, 3 / globalScale);
      ctx.fill();
      ctx.globalAlpha = labelOpacity;
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, node.x, pillY);
      ctx.globalAlpha = 1.0;
    }
  })
  .nodeCanvasObjectMode(() => 'replace')
  .linkColor(link => {
    if (queryMode === null) return 'rgba(255,255,255,0.08)';
    const a = isActiveLink(link);
    if (a.both) return 'rgba(0,255,136,0.5)';
    if (a.either) return 'rgba(0,255,136,0.25)';
    return 'rgba(255,255,255,0.03)';
  })
  .linkWidth(link => {
    if (queryMode === null) return 0.5;
    return isActiveLink(link).either ? 1.5 : 0.3;
  })
  .linkDirectionalParticles(2)
  .linkDirectionalParticleWidth(link => {
    if (queryMode === null) return 1.5;
    return isActiveLink(link).either ? 2.5 : 0;
  })
  .linkDirectionalParticleColor(link => {
    if (queryMode === null) return 'rgba(255,255,255,0.3)';
    return isActiveLink(link).either
      ? 'rgba(0,255,136,0.6)' : 'rgba(0,0,0,0)';
  })
  .linkDirectionalParticleSpeed(link => {
    if (queryMode === null) return 0.003;
    return isActiveLink(link).either ? 0.006 : 0.003;
  })
  .nodeVisibility(nodeVisible)
  .linkVisibility(linkVisible)
  .onNodeHover(node => { hoveredNode = node; })
  .onNodeClick(node => {
    if (!node) return;
    selectNode(node);
  })
  .onBackgroundClick(() => {
    focusedNode = null;
    highlightedNode = null;
    graph.nodeVisibility(nodeVisible)
      .linkVisibility(linkVisible);
    document.getElementById('info-panel')
      .classList.remove('visible');
    setTimeout(() => {
      graph.zoomToFit(400, 40, connected);
      setTimeout(clampZoom, 450);
    }, 100);
  });

// Edge labels on hover
graph.linkLabel(link => `${link.type}`);

// Zoom to fit early — skip if user already started exploring
setTimeout(() => {
  if (!focusedNode && navHistory.length === 0) {
    graph.zoomToFit(400, 40, connected);
    setTimeout(clampZoom, 450);
  }
}, 1500);

// Solo mode — click legend to filter by node type
function toggleSolo(type) {
  if (soloType === type) {
    soloType = null;
  } else {
    soloType = type;
  }
  focusedNode = null;
  highlightedNode = null;
  legendItems.forEach((el, t) => {
    el.classList.toggle(
      'inactive', soloType != null && t !== soloType
    );
  });
  graph.nodeVisibility(nodeVisible)
    .linkVisibility(linkVisible);
  setTimeout(() => {
    graph.zoomToFit(400, 40, connected);
    setTimeout(clampZoom, 450);
  }, 100);
}

// Info panel
document.getElementById('close-btn').addEventListener('click', () => {
  document.getElementById('info-panel').classList.remove('visible');
});

function showInfoPanel(node) {
  const panel = document.getElementById('info-panel');
  const content = document.getElementById('info-content');
  const color = NODE_COLORS[node.type] || '#666';
  const props = node.properties || {};

  let html = `<span class="type-badge" style="background:${color};color:#000">${node.type}</span>`;
  html += `<h2>${escapeHtml(node.label || node.id)}</h2>`;
  html += `<div class="meta">${escapeHtml(node.id)}</div>`;

  if (props.long_title) html += `<div class="meta">${escapeHtml(props.long_title)}</div>`;
  if (props.entry_type)
    html += `<div class="meta"><b>Type:</b> ${escapeHtml(props.entry_type)}</div>`;
  if (props.tags) html += `<div class="meta"><b>Tags:</b> ${escapeHtml(props.tags)}</div>`;
  if (props.project_ref)
    html += `<div class="meta"><b>Project:</b> ${escapeHtml(props.project_ref)}</div>`;
  if (props.contributor)
    html += `<div class="meta"><b>By:</b> ${escapeHtml(props.contributor)}</div>`;
  if (props.confidence_level != null)
    html += `<div class="meta"><b>Confidence:</b> `
      + Math.round(props.confidence_level * 100) + `%</div>`;
  if (props.summary) html += `<div class="meta">${escapeHtml(props.summary)}</div>`;

  // Full entry accordion (fetched on demand from web server)
  if (node.type === 'entry' && isServed) {
    html += '<div class="entry-details">'
      + '<button class="details-toggle" onclick="toggleDetails(this, \x27'
      + escapeAttr(node.id) + '\x27)">'
      + '<span class="arrow">\u25b6</span> Full Entry\u2026</button>'
      + '<div class="details-body"></div></div>';
  }

  const conns = adjacency.get(node.id) || [];
  if (conns.length > 0) {
    html += `<div class="connections"><h3>Connections (${conns.length})</h3>`;
    conns.slice(0, 30).forEach(c => {
      const arrow = c.dir === 'out' ? '\u2192' : '\u2190';
      html += `<div class="conn-item" onclick="focusNode('${escapeAttr(c.node)}')">`
            + `${arrow} <strong>${escapeHtml(c.type)}</strong> ${escapeHtml(c.node)}</div>`;
    });
    if (conns.length > 30) html += `<div class="meta">... and ${conns.length - 30} more</div>`;
    html += '</div>';
  }

  content.innerHTML = html;
  panel.classList.add('visible');
}

function focusNode(nodeId) {
  const node = GRAPH_DATA.nodes.find(n => n.id === nodeId);
  if (node) selectNode(node);
}

async function toggleDetails(btn, entryId) {
  const arrow = btn.querySelector('.arrow');
  const body = btn.nextElementSibling;
  if (body.classList.contains('visible')) {
    body.classList.remove('visible');
    arrow.classList.remove('open');
    return;
  }
  // Fetch if not already loaded
  if (!body.dataset.loaded) {
    arrow.textContent = '\u23f3';
    try {
      const resp = await fetch('/api/entry/' + encodeURIComponent(entryId));
      if (!resp.ok) throw new Error(resp.statusText);
      const data = await resp.json();
      body.innerHTML = renderMarkdown(data.knowledge_details || '(no content)');
      body.dataset.loaded = '1';
    } catch (e) {
      body.innerHTML = '<em>Failed to load entry</em>';
    }
    arrow.textContent = '\u25b6';
  }
  body.classList.add('visible');
  arrow.classList.add('open');
}

// Left arrow = backtrack through navigation history
const searchInput = document.getElementById('search-input');
document.addEventListener('keydown', e => {
  if (document.activeElement === searchInput) return;
  if (e.key === 'Escape') {
    if (queryMode !== null) { closeConversation(); return; }
    focusedNode = null;
    highlightedNode = null;
    navHistory.length = 0;
    graph.nodeVisibility(nodeVisible)
      .linkVisibility(linkVisible);
    document.getElementById('info-panel')
      .classList.remove('visible');
    graph.zoomToFit(400, 40, connected);
    setTimeout(clampZoom, 450);
    return;
  }
  if (e.key !== 'ArrowLeft') return;
  if (navHistory.length <= 1) return;
  navHistory.pop();
  const prev = navHistory[navHistory.length - 1];
  focusedNode = prev.focused ?? null;
  highlightedNode = null;
  graph.nodeVisibility(nodeVisible)
    .linkVisibility(linkVisible);
  graph.centerAt(prev.cx, prev.cy, 600);
  graph.zoom(prev.zoom, 600);
  setTimeout(() => showInfoPanel(prev.node), 650);
});

// Search
const searchResultsEl = document.getElementById('search-results');
let searchIdx = -1;

// Update placeholder based on serve mode
if (isServed) {
  searchInput.placeholder = 'Search nodes or ask a question...';
}

searchInput.addEventListener('input', () => {
  autoResize(searchInput);
  const q = searchInput.value.trim().toLowerCase();
  searchIdx = -1;
  if (!q) {
    searchResultsEl.classList.remove('visible');
    searchResultsEl.innerHTML = '';
    return;
  }
  const matches = GRAPH_DATA.nodes
    .filter(n => {
      const hay = [
        n.label, n.id,
        n.properties?.long_title,
        n.properties?.tags,
        n.properties?.entry_type,
        n.properties?.project_ref,
      ].filter(Boolean).join(' ').toLowerCase();
      return hay.includes(q);
    })
    .sort((a, b) => {
      const ae = (a.label || '').toLowerCase() === q ? 1 : 0;
      const be = (b.label || '').toLowerCase() === q ? 1 : 0;
      if (ae !== be) return be - ae;
      return (b.val || 0) - (a.val || 0);
    })
    .slice(0, 12);
  if (!matches.length) {
    searchResultsEl.classList.remove('visible');
    searchResultsEl.innerHTML = '';
    return;
  }
  searchResultsEl.innerHTML = matches.map(n => {
    const c = NODE_COLORS[n.type] || '#666';
    const idStr = n.id !== n.label ? n.id : '';
    return '<div class="search-result" data-id="'
      + escapeHtml(n.id) + '">'
      + '<span class="sr-dot" style="background:'
      + c + '"></span>'
      + '<span class="sr-label">'
      + escapeHtml(n.label) + '</span>'
      + '<span class="sr-id">'
      + escapeHtml(idStr) + '</span></div>';
  }).join('');
  searchResultsEl.classList.add('visible');
});

searchResultsEl.addEventListener('click', e => {
  const item = e.target.closest('.search-result');
  if (!item) return;
  flyToNode(item.dataset.id);
  clearSearch();
});

searchInput.addEventListener('keydown', e => {
  const items = searchResultsEl.querySelectorAll(
    '.search-result'
  );
  if (e.key === 'Escape') {
    clearSearch();
    searchInput.blur();
  } else if (e.key === 'ArrowDown') {
    e.preventDefault();
    searchIdx = Math.min(searchIdx + 1, items.length - 1);
    updateSearchHL(items);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    searchIdx = Math.max(searchIdx - 1, 0);
    updateSearchHL(items);
  } else if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    // If user selected an autocomplete item, fly to it
    if (searchIdx >= 0 && items[searchIdx]) {
      flyToNode(items[searchIdx].dataset.id);
      clearSearch();
    } else if (items.length > 0 && !isServed) {
      // file:// mode: always fly to first autocomplete result
      flyToNode(items[0].dataset.id);
      clearSearch();
    } else if (isServed && searchInput.value.trim()) {
      // Web mode: free-form query
      const q = searchInput.value.trim();
      clearSearch();
      startQuery(q);
    } else if (items.length > 0) {
      flyToNode(items[0].dataset.id);
      clearSearch();
    }
  }
});

function updateSearchHL(items) {
  items.forEach((el, i) => {
    el.classList.toggle('active', i === searchIdx);
  });
  if (items[searchIdx]) {
    items[searchIdx].scrollIntoView({ block: 'nearest' });
  }
}

function clearSearch() {
  searchInput.value = '';
  searchInput.style.height = 'auto';
  searchResultsEl.classList.remove('visible');
  searchResultsEl.innerHTML = '';
  searchIdx = -1;
}

function flyToNode(nodeId) {
  const node = GRAPH_DATA.nodes.find(n => n.id === nodeId);
  if (!node) return;
  if (soloType) {
    soloType = null;
    legendItems.forEach(el => {
      el.classList.remove('inactive');
    });
  }
  selectNode(node);
}

function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function escapeAttr(s) {
  return String(s).replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

// --- Query-driven exploration (web server mode only) ---
const statusLine = document.getElementById('status-line');
const responsePanel = document.getElementById('response-panel');
const responseContent = document.getElementById('response-content');
const defaultPlaceholder = isServed
  ? 'Search nodes or ask a question...' : 'Search nodes...';

function setStatus(msg) {
  if (!msg) {
    statusLine.classList.add('hidden');
    statusLine.textContent = '';
    searchInput.placeholder = defaultPlaceholder;
  } else {
    statusLine.textContent = msg;
    statusLine.classList.remove('hidden');
    searchInput.placeholder = msg;
  }
}

function resetTraversalState() {
  queryMode = null;
  visitedNodes.clear();
  resultNodes.clear();
  nodeStates.clear();
  pulseNodes.clear();
  animQueue.length = 0;
  animRunning = false;
  hideResponsePanel();
  // Reset chat panel immediately (no animation) on new query
  chatPanel.classList.remove('visible');
  chatSessionId = null;
  chatMessages.innerHTML = '';
  document.getElementById('search-bar').classList.remove('hidden-for-chat');
  setStatus('');
}

function markVisited(nodeId) {
  visitedNodes.add(nodeId);
  if (!nodeStates.has(nodeId) || nodeStates.get(nodeId) !== 'result') {
    nodeStates.set(nodeId, 'visited');
  }
  triggerPulse(nodeId);
}

function markResult(nodeId) {
  resultNodes.add(nodeId);
  nodeStates.set(nodeId, 'result');
  triggerPulse(nodeId);
}

// Animation queue — stagger node reveals for a traversal feel
const animQueue = [];
let animRunning = false;
const ANIM_DELAY = 250;  // ms between node reveals

function queueAnimation(fn) {
  animQueue.push(fn);
  if (!animRunning) drainAnimQueue();
}

async function drainAnimQueue() {
  animRunning = true;
  while (animQueue.length > 0) {
    const fn = animQueue.shift();
    fn();
    await new Promise(r => setTimeout(r, ANIM_DELAY));
  }
  animRunning = false;
}

function flushAnimQueue() {
  // Run remaining animations instantly (for stream_end)
  while (animQueue.length > 0) animQueue.shift()();
  animRunning = false;
}

function emitTraversalParticles(entryIds) {
  const links = graph.graphData().links;
  const idSet = new Set(entryIds);
  links.forEach(link => {
    const s = link.source.id || link.source;
    const t = link.target.id || link.target;
    if (idSet.has(s) || idSet.has(t)) {
      graph.emitParticle(link);
    }
  });
}

function clampZoom() {
  // After zoomToFit, cap zoom to avoid over-zooming on small clusters
  if (graph.zoom() > MAX_AUTO_ZOOM) graph.zoom(MAX_AUTO_ZOOM, 300);
}

function zoomToVisited() {
  // Smoothly widen view to fit all visited + result nodes so far
  const allActive = new Set([...visitedNodes, ...resultNodes]);
  if (allActive.size === 0) return;
  graph.zoomToFit(500, 60, n => allActive.has(n.id));
  setTimeout(clampZoom, 550);
}

function revealNodesStaggered(nodeIds, state) {
  // Queue each node reveal as a separate animation step.
  // Instead of flying to each node, just mark + particles,
  // then widen the view to fit all revealed nodes so far.
  const markFn = state === 'result' ? markResult : markVisited;
  nodeIds.forEach((id, i) => {
    queueAnimation(() => {
      markFn(id);
      emitTraversalParticles([id]);
      zoomToVisited();
    });
  });
}

function zoomToResults() {
  if (resultNodes.size === 0) return;
  // Queue the zoom after any pending animations
  queueAnimation(() => {
    graph.zoomToFit(600, 40, n => resultNodes.has(n.id));
    setTimeout(clampZoom, 650);
  });
}

function renderMarkdown(text) {
  // Use marked if loaded, fall back to escaped plain text
  if (typeof marked !== 'undefined' && marked.parse) {
    try { return marked.parse(text); }
    catch (e) { /* fall through */ }
  }
  return '<p>' + escapeHtml(text).replace(/\n/g, '<br>') + '</p>';
}

function showResponsePanel(answer) {
  // Render markdown first, then replace [kb-XXXXX] citations in output.
  // Brackets survive markdown rendering (no matching link definition).
  const html = renderMarkdown(answer).replace(
    /\[(kb-\d{5})\]/g,
    function(_, id) {
      return '<span class="citation" onclick="flyToNode(\x27' + id + '\x27)">[' + id + ']</span>';
    }
  );
  responseContent.innerHTML = html;
  responsePanel.classList.add('visible');
}

function hideResponsePanel(keepSearchHidden) {
  responsePanel.classList.remove('visible');
  responseContent.innerHTML = '';
  if (!keepSearchHidden) {
    document.getElementById('search-bar').classList.remove('hidden-for-chat');
  }
}

function handleSSEEvent(eventType, data) {
  if (eventType === 'status') {
    setStatus(data.message || '');
  } else if (eventType === 'classified') {
    queryMode = data.mode;
    setStatus(data.mode === 'summarize'
      ? 'Preparing answer...' : 'Exploring knowledge graph...');
  } else if (eventType === 'tool_call') {
    if (data.tool === 'graph_neighbors' && data.args?.node_id) {
      queueAnimation(() => {
        markVisited(data.args.node_id);
        emitTraversalParticles([data.args.node_id]);
        zoomToVisited();
      });
    }
  } else if (eventType === 'tool_result') {
    // Stagger through each returned entry one by one
    if (data.entry_ids) {
      revealNodesStaggered(data.entry_ids, 'visited');
    }
  } else if (eventType === 'fast_path' || eventType === 'agent_done') {
    // Reveal final results one by one in green
    if (data.entry_ids) {
      revealNodesStaggered(data.entry_ids, 'result');
    }
  } else if (eventType === 'entries') {
    // Final entries from route handler — mark results and zoom
    if (data.entries && data.entries.length > 0) {
      const entryIds = data.entries.map(e => e.id);
      revealNodesStaggered(entryIds, 'result');
      // Hide search bar — response panel takes its place
      document.getElementById('search-bar').classList.add('hidden-for-chat');
      // Store for chat seeding
      lastExploreQuestion = lastQueryQuestion;
      lastExploreEntryIds = entryIds;
      // Show result list in response panel with summarize button
      let md = '**Found ' + data.entries.length + ' entries:**\n\n';
      data.entries.forEach(e => {
        md += '- **[' + e.id + ']** ' + e.short_title;
        if (e.context) md += ' — _' + e.context + '_';
        md += '\n';
      });
      queueAnimation(() => {
        showResponsePanel(md);
        // Add summarize button and follow-up input
        const btnBar = document.createElement('div');
        btnBar.innerHTML = '<button class="summarize-btn" '
          + 'onclick="summarizeExploreResults()">Summarize these</button>';
        responseContent.appendChild(btnBar);
        // Add follow-up input bar
        const followUp = document.createElement('div');
        followUp.style.cssText = 'display:flex;gap:8px;margin-top:10px;';
        followUp.innerHTML = '<textarea id="explore-followup" rows="1" '
          + 'placeholder="Ask a follow-up..." autocomplete="off" '
          + 'style="flex:1;padding:8px 10px;background:rgba(30,30,40,0.9);'
          + 'border:1px solid #444;border-radius:6px;color:#e0e0e0;'
          + 'font-size:13px;outline:none;font-family:inherit;resize:none;'
          + 'overflow:hidden;line-height:1.4;max-height:120px;"></textarea>'
          + '<button class="summarize-btn" '
          + 'onclick="startExploreChat()">Send</button>';
        responseContent.appendChild(followUp);
        const inp = document.getElementById('explore-followup');
        if (inp) {
          inp.addEventListener('input', () => autoResize(inp));
          inp.addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
              e.preventDefault(); startExploreChat();
            }
          });
        }
      });
    } else {
      queueAnimation(() => {
        setStatus('No results found');
        setTimeout(() => setStatus(''), 3000);
      });
    }
    zoomToResults();
  } else if (eventType === 'synthesis_result') {
    if (data.answer) {
      queueAnimation(() => {
        openChatPanel(
          data.question || '', data.answer,
          data.entry_ids || [], 'summarize'
        );
      });
    }
    zoomToResults();
  } else if (eventType === 'error') {
    setStatus('Error: ' + (data.message || 'query failed'));
    setTimeout(() => setStatus(''), 15000);
  } else if (eventType === 'stream_end') {
    // Flush remaining animations and clear status
    queueAnimation(() => setStatus(''));
  }
}

// Track last explore results for chat seeding
let lastExploreQuestion = '';
let lastExploreEntryIds = [];
let lastQueryQuestion = '';

function summarizeExploreResults() {
  // Seed chat with the explore results and ask for a summary
  const syntheticAnswer = 'Found ' + lastExploreEntryIds.length
    + ' entries: ' + lastExploreEntryIds.join(', ')
    + '. Ask me anything about these results.';
  openChatPanel(
    lastExploreQuestion, syntheticAnswer, lastExploreEntryIds,
    'explore'
  );
  // Auto-send a summarize request
  chatInput.value = 'Summarize these entries for me.';
  sendChatMessage();
}

function startExploreChat() {
  const inp = document.getElementById('explore-followup');
  const msg = inp ? inp.value.trim() : '';
  if (!msg) return;
  const syntheticAnswer = 'Found ' + lastExploreEntryIds.length
    + ' entries: ' + lastExploreEntryIds.join(', ') + '.';
  openChatPanel(
    lastExploreQuestion, syntheticAnswer, lastExploreEntryIds,
    'explore'
  );
  chatInput.value = msg;
  sendChatMessage();
}

async function startQuery(question) {
  resetTraversalState();
  queryMode = 'pending';
  lastQueryQuestion = question;
  setStatus('Classifying query...');

  // Reset focus/solo mode
  focusedNode = null;
  highlightedNode = null;
  soloType = null;
  legendItems.forEach(el => el.classList.remove('inactive'));
  graph.nodeVisibility(nodeVisible).linkVisibility(linkVisible);

  try {
    const resp = await fetch('/api/query/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question}),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});

      // Parse SSE events from buffer
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      let currentEvent = null;
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ') && currentEvent) {
          let data;
          try { data = JSON.parse(line.slice(6)); }
          catch (e) { currentEvent = null; continue; }
          try { handleSSEEvent(currentEvent, data); }
          catch (e) { console.error('SSE handler error:', e); }
          currentEvent = null;
        }
      }
    }
  } catch (err) {
    setStatus('Query failed: ' + err.message);
    setTimeout(() => setStatus(''), 3000);
  }
}

// --- Chat panel ---
const chatPanel = document.getElementById('chat-panel');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
let chatSessionId = null;
let chatSeedQuestion = '';
let chatSeedAnswer = '';
let chatSeedEntryIds = [];
let chatCurrentMode = '';
let chatBusy = false;

function addChatCitation(html) {
  return html.replace(
    /\[(kb-\d{5})\]/g,
    function(_, id) {
      return '<span class="citation" onclick="flyToNode(\x27'
        + id + '\x27)">[' + id + ']</span>';
    }
  );
}

function appendChatMsg(role, contentHtml) {
  const div = document.createElement('div');
  div.className = 'chat-msg ' + role;
  div.innerHTML = addChatCitation(contentHtml);
  if (role === 'assistant') {
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.innerHTML = copySvg;
    btn.onclick = function() { copyMsgText(this); };
    div.appendChild(btn);
  }
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

function showTypingIndicator() {
  const div = document.createElement('div');
  div.className = 'chat-typing';
  div.id = 'chat-typing';
  div.textContent = 'Thinking';
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById('chat-typing');
  if (el) el.remove();
}

function openChatPanel(question, answer, entryIds, mode) {
  // Generate a session ID upfront so it can be persisted immediately
  chatSessionId = crypto.randomUUID();
  chatSeedQuestion = question;
  chatSeedAnswer = answer;
  chatSeedEntryIds = entryIds || [];
  chatCurrentMode = mode || queryMode || 'summarize';
  chatMessages.innerHTML = '';

  // Persist the chat immediately (fire-and-forget)
  if (isServed && question) {
    fetch('/api/chat/create', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        chat_id: chatSessionId,
        question: question,
        answer: answer || '',
        mode: chatCurrentMode,
      }),
    }).catch(e => console.error('Failed to persist chat:', e));
  }

  // Populate header
  const badge = document.getElementById('chat-mode-badge');
  const queryEl = document.getElementById('chat-query');
  const chatMode = mode || queryMode || 'summarize';
  badge.textContent = chatMode;
  badge.className = 'chat-mode-badge ' + chatMode;
  queryEl.textContent = question || '';
  queryEl.title = question || '';

  // Seed with original Q + A
  if (question) {
    appendChatMsg('user', '<p>' + escapeHtml(question) + '</p>');
  }
  appendChatMsg('assistant', renderMarkdown(answer));

  hideResponsePanel(true);  // keep search hidden — chat panel manages it
  // Hide search bar, then reveal chat panel
  const searchBar = document.getElementById('search-bar');
  searchBar.classList.add('hidden-for-chat');
  // Small delay so search bar fades first
  setTimeout(() => {
    chatPanel.classList.add('visible');
    chatInput.focus();
  }, 100);
}

function closeConversation() {
  // Clear traversal visuals
  queryMode = null;
  visitedNodes.clear();
  resultNodes.clear();
  nodeStates.clear();
  pulseNodes.clear();
  animQueue.length = 0;
  animRunning = false;
  setStatus('');
  // Hide panels
  responsePanel.classList.remove('visible');
  responseContent.innerHTML = '';
  chatMaximized = false;
  chatPanel.classList.remove('maximized', 'visible');
  document.getElementById('chat-maximize-btn').innerHTML = expandSvg;
  chatSessionId = null;
  chatCurrentMode = '';
  chatMessages.innerHTML = '';
  document.getElementById('info-panel').classList.remove('visible');
  // Show search bar
  document.getElementById('search-bar').classList.remove('hidden-for-chat');
  loadChatHistory();
  // Reset graph to fresh state
  focusedNode = null;
  highlightedNode = null;
  soloType = null;
  navHistory.length = 0;
  legendItems.forEach(el => el.classList.remove('inactive'));
  graph.nodeVisibility(nodeVisible).linkVisibility(linkVisible);
  graph.zoomToFit(400, 40, connected);
  setTimeout(clampZoom, 450);
}

function setChatBusy(busy) {
  chatBusy = busy;
  chatInput.disabled = busy;
  chatSend.disabled = busy;
}

async function sendChatMessage() {
  const msg = chatInput.value.trim();
  if (!msg || chatBusy) return;
  chatInput.value = '';
  chatInput.style.height = 'auto';
  setChatBusy(true);

  appendChatMsg('user', '<p>' + escapeHtml(msg) + '</p>');
  showTypingIndicator();

  try {
    const payload = {message: msg, mode: chatCurrentMode};
    if (chatSessionId) {
      payload.session_id = chatSessionId;
    } else {
      payload.seed_question = chatSeedQuestion;
      payload.seed_answer = chatSeedAnswer;
      payload.seed_entry_ids = chatSeedEntryIds;
    }

    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let assistantDiv = null;

    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      let currentEvent = null;
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ') && currentEvent) {
          let data;
          try { data = JSON.parse(line.slice(6)); }
          catch (e) { currentEvent = null; continue; }

          if (currentEvent === 'chat_session') {
            chatSessionId = data.session_id;
          } else if (currentEvent === 'chat_response') {
            removeTypingIndicator();
            assistantDiv = appendChatMsg(
              'assistant', renderMarkdown(data.answer || '')
            );
            // Highlight any new entries on graph
            if (data.new_entries) {
              data.new_entries.forEach(id => markResult(id));
            }
          } else if (currentEvent === 'chat_done') {
            if (data.new_entries && data.new_entries.length > 0) {
              revealNodesStaggered(data.new_entries, 'result');
            }
          } else if (currentEvent === 'error') {
            removeTypingIndicator();
            appendChatMsg(
              'assistant',
              '<p style="color:#e57373">Error: '
                + escapeHtml(data.message || 'unknown') + '</p>'
            );
          }
          currentEvent = null;
        }
      }
    }
  } catch (err) {
    removeTypingIndicator();
    appendChatMsg(
      'assistant',
      '<p style="color:#e57373">Error: '
        + escapeHtml(err.message) + '</p>'
    );
  } finally {
    setChatBusy(false);
    chatInput.focus();
  }
}

chatInput.addEventListener('input', () => autoResize(chatInput));
chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    sendChatMessage();
  }
});

// --- Chat history ---
function formatRelativeTime(isoStr) {
  // Append Z only if no timezone info present (no Z, +, or - after the date)
  let s = isoStr;
  if (!s.endsWith('Z') && !/[+-]\d{2}:\d{2}$/.test(s)) s += 'Z';
  const d = new Date(s);
  const now = new Date();
  const diff = Math.floor((now - d) / 1000);
  if (diff < 60) return 'just now';
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
  if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
  if (diff < 172800) return 'yesterday';
  const mo = d.toLocaleString('en', {month: 'short'});
  return mo + ' ' + d.getDate();
}

async function loadChatHistory() {
  if (!isServed) return;
  const list = document.getElementById('chat-history-list');
  try {
    const resp = await fetch('/api/chat/history');
    const chats = await resp.json();
    list.innerHTML = '';
    if (chats.length === 0) {
      list.classList.remove('visible');
      return;
    }
    list.innerHTML = '<div class="ch-header">Recent chats</div>';
    chats.forEach(chat => {
      const item = document.createElement('div');
      item.className = 'chat-history-item';
      const t = escapeHtml(chat.title);
      const ts = formatRelativeTime(chat.updated_at);
      item.innerHTML =
        '<span class="ch-title">' + t + '</span>'
        + '<span class="ch-time">' + ts + '</span>';
      const delBtn = document.createElement('button');
      delBtn.className = 'ch-delete';
      delBtn.title = 'Delete';
      delBtn.innerHTML = '&#x1f5d1;';
      delBtn.onclick = (e) => {
        e.stopPropagation(); deleteChat(chat.id);
      };
      item.appendChild(delBtn);
      item.onclick = () => resumeChat(
        chat.id, chat.title, chat.mode
      );
      list.appendChild(item);
    });
    list.classList.add('visible');
  } catch (e) {
    console.error('Failed to load chat history:', e);
    list.classList.remove('visible');
  }
}

async function resumeChat(chatId, title, mode) {
  try {
    const resp = await fetch('/api/chat/' + chatId + '/messages');
    if (!resp.ok) { console.error('Load failed'); return; }
    const messages = await resp.json();

    chatSessionId = chatId;
    chatSeedQuestion = title;
    chatSeedAnswer = '';
    chatSeedEntryIds = [];
    chatCurrentMode = mode || 'chat';
    chatMessages.innerHTML = '';

    const badge = document.getElementById('chat-mode-badge');
    badge.textContent = chatCurrentMode;
    badge.className = 'chat-mode-badge ' + chatCurrentMode;
    document.getElementById('chat-query').textContent = title;
    document.getElementById('chat-query').title = title;

    messages.forEach(m => {
      if (m.role === 'user') {
        appendChatMsg('user',
          '<p>' + escapeHtml(m.content) + '</p>');
      } else {
        appendChatMsg('assistant', renderMarkdown(m.content));
      }
    });

    hideResponsePanel(true);
    const searchBar = document.getElementById('search-bar');
    searchBar.classList.add('hidden-for-chat');
    setTimeout(() => {
      chatPanel.classList.add('visible');
      chatInput.focus();
    }, 100);
  } catch (e) {
    console.error('Failed to resume chat:', e);
  }
}

async function deleteChat(chatId) {
  try {
    await fetch('/api/chat/' + chatId, {method: 'DELETE'});
    if (chatSessionId === chatId) closeConversation();
    else loadChatHistory();
  } catch (e) { console.error('Delete failed:', e); }
}

// Load history on page load
loadChatHistory();

// --- Ingest modal ---
const ingestModal = document.getElementById('ingest-modal');
const ingestProjectInput = document.getElementById('ingest-project-input');
const ingestStatus = document.getElementById('ingest-status');
const ingestSubmitBtn = document.getElementById('ingest-submit-btn');
const ingestProgress = document.getElementById('ingest-progress');
const ingestProgressFill = document.getElementById('ingest-progress-fill');
const ingestProgressText = document.getElementById('ingest-progress-text');
const ingestUrlRows = document.getElementById('ingest-url-rows');
const ingestFileInput = document.getElementById('ingest-file-input');
const ingestFileList = document.getElementById('ingest-file-list');
let activeIngestTab = 'url';

async function populateProjectDropdown() {
  const dl = document.getElementById('ingest-project-list');
  let projects = [];
  try {
    const resp = await fetch('/api/projects');
    projects = await resp.json();
  } catch (e) {
    projects = GRAPH_DATA.nodes
      .filter(n => n.type === 'project')
      .map(n => n.id.replace(/^project:/, ''))
      .sort();
  }
  dl.innerHTML = '';
  projects.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p;
    dl.appendChild(opt);
  });
}

function switchIngestTab(tab) {
  activeIngestTab = tab;
  document.querySelectorAll('.ingest-tab').forEach(
    t => t.classList.toggle('active', t.dataset.tab === tab));
  document.getElementById('ingest-tab-url').classList.toggle('active', tab === 'url');
  document.getElementById('ingest-tab-file').classList.toggle('active', tab === 'file');
}

function addUrlRow() {
  const row = document.createElement('div');
  row.className = 'ingest-url-row';
  row.innerHTML = '<input type="url" placeholder="https://..."'
    + ' autocomplete="off" />'
    + '<button class="ingest-url-remove"'
    + ' onclick="this.parentElement.remove()">&times;</button>';
  ingestUrlRows.appendChild(row);
  row.querySelector('input').focus();
}

function updateFileList() {
  const files = ingestFileInput.files;
  ingestFileList.innerHTML = '';
  for (const f of files) {
    const d = document.createElement('div');
    d.textContent = f.name + ' (' + (f.size / 1024).toFixed(1) + ' KB)';
    ingestFileList.appendChild(d);
  }
}

function openIngestModal(tab) {
  if (!isServed) return;
  // Reset URL rows to single empty row
  ingestUrlRows.innerHTML = '<div class="ingest-url-row">'
    + '<input type="url" placeholder="https://..."'
    + ' autocomplete="off" /></div>';
  ingestFileInput.value = '';
  ingestFileList.innerHTML = '';
  ingestProjectInput.value = '';
  populateProjectDropdown();
  ingestStatus.textContent = '';
  ingestStatus.className = '';
  ingestProgress.classList.remove('visible');
  ingestProgressFill.style.width = '0%';
  ingestProgressText.textContent = '';
  ingestSubmitBtn.disabled = false;
  switchIngestTab(tab || 'url');
  ingestModal.classList.add('visible');
  if (activeIngestTab === 'url') {
    ingestUrlRows.querySelector('input').focus();
  }
}

function closeIngestModal() {
  ingestModal.classList.remove('visible');
}

ingestModal.addEventListener('click', e => {
  if (e.target === ingestModal) closeIngestModal();
});

ingestModal.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeIngestModal();
  if (e.key === 'Enter' && !e.isComposing && e.target.tagName === 'INPUT') {
    e.preventDefault(); submitIngest();
  }
});

async function collectIngestItems() {
  const items = [];
  if (activeIngestTab === 'url') {
    const inputs = ingestUrlRows.querySelectorAll('input');
    for (const inp of inputs) {
      const v = inp.value.trim();
      if (v) items.push({type: 'url', value: v});
    }
  } else {
    const files = ingestFileInput.files;
    for (const f of files) {
      if (f.name.toLowerCase().endsWith('.pdf')) {
        const buf = await f.arrayBuffer();
        const bytes = new Uint8Array(buf);
        let bin = '';
        for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
        items.push({type: 'file', name: f.name, content: btoa(bin), encoding: 'base64'});
      } else {
        const text = await f.text();
        items.push({type: 'file', name: f.name, content: text});
      }
    }
  }
  return items;
}

async function submitIngest() {
  const items = await collectIngestItems();
  if (items.length === 0) {
    ingestStatus.textContent = activeIngestTab === 'url'
      ? 'Enter at least one URL' : 'Select at least one file';
    ingestStatus.className = 'error';
    return;
  }

  ingestSubmitBtn.disabled = true;
  ingestStatus.textContent = '';
  ingestStatus.className = '';
  ingestProgress.classList.add('visible');
  ingestProgressFill.style.width = '0%';
  ingestProgressText.textContent = 'Starting...';

  const payload = {items};
  const project = ingestProjectInput.value.trim();
  if (project) payload.project_ref = project;

  try {
    const resp = await fetch('/api/ingest/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let batchTotal = items.length;
    let batchIndex = 0;
    let chunksDone = 0;
    let chunksTotal = 1;
    let allEntryIds = [];

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});

      const lines = buffer.split('\n');
      buffer = lines.pop();

      let eventType = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ') && eventType) {
          const data = JSON.parse(line.slice(6));
          if (eventType === 'batch_progress') {
            batchIndex = data.batch_index;
            batchTotal = data.batch_total;
            chunksDone = 0;
            chunksTotal = 1;
            const label = batchTotal > 1 ? (batchIndex + 1) + '/' + batchTotal + ': ' : '';
            ingestProgressText.textContent = label + 'Starting ' + data.source + '...';
          } else if (eventType === 'ingest_summarizing') {
            const label = batchTotal > 1 ? (batchIndex + 1) + '/' + batchTotal + ': ' : '';
            ingestProgressText.textContent = label + 'Summarizing...';
          } else if (eventType === 'ingest_start') {
            chunksTotal = data.total_chunks || 1;
            chunksDone = 0;
          } else if (eventType === 'ingest_chunk_start') {
            chunksTotal = data.total_chunks || 1;
            const ci = data.chunk_index || 0;
            const label = batchTotal > 1 ? (batchIndex + 1) + '/' + batchTotal + ': ' : '';
            ingestProgressText.textContent = label
              + 'Extracting chunk ' + (ci+1) + '/' + chunksTotal + '...';
          } else if (eventType === 'ingest_chunk_done') {
            chunksDone = (data.chunk_index || 0) + 1;
            const itemFrac = chunksDone / chunksTotal;
            const overallPct = ((batchIndex + itemFrac) / batchTotal * 100).toFixed(0);
            ingestProgressFill.style.width = overallPct + '%';
          } else if (eventType === 'item_done') {
            const ids = data.entry_ids || [];
            allEntryIds.push(...ids);
            ingestProgressFill.style.width = ((batchIndex + 1) / batchTotal * 100).toFixed(0) + '%';
          } else if (eventType === 'ingest_error') {
            const label = batchTotal > 1 ? (batchIndex + 1) + '/' + batchTotal + ': ' : '';
            ingestProgressText.textContent = label + (data.error || 'Error');
          } else if (eventType === 'batch_done') {
            allEntryIds = data.entry_ids || allEntryIds;
            const n = data.total_entries || 0;
            ingestProgressFill.style.width = '100%';
            ingestStatus.textContent = n === 0
              ? 'No entries created'
              : 'Created ' + n + ' entries: ' + allEntryIds.join(', ');
            ingestStatus.className = n > 0 ? 'success' : '';

            // Reload graph data then highlight new entries
            if (n > 0) {
              fetch('/api/graph').then(r => r.json()).then(gd => {
                const links = gd.edges.map(e => ({
                  source: e.source, target: e.target, type: e.type
                }));
                graph.graphData({ nodes: gd.nodes, links: links });
                setTimeout(() => {
                  allEntryIds.forEach(id => markResult(id));
                  emitTraversalParticles(allEntryIds);
                  setTimeout(() => {
                    graph.zoomToFit(500, 60, nd => allEntryIds.includes(nd.id));
                    setTimeout(clampZoom, 550);
                  }, 300);
                }, 200);
              });
            }
            setTimeout(closeIngestModal, 4000);
          } else if (eventType === 'error') {
            ingestStatus.textContent = data.message || 'Error';
            ingestStatus.className = 'error';
            ingestSubmitBtn.disabled = false;
          }
          eventType = '';
        }
      }
    }
  } catch (err) {
    ingestStatus.textContent = 'Error: ' + err.message;
    ingestStatus.className = 'error';
    ingestSubmitBtn.disabled = false;
  }
}
