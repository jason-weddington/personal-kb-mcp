"""Render a self-contained HTML file with the knowledge graph visualization."""

import json
from typing import Any

# Node colors by type — matches the research doc visual design
_NODE_COLORS = {
    "entry": "#e0e0e0",
    "tag": "#00bcd4",
    "project": "#ff9800",
    "person": "#ffc107",
    "tool": "#4caf50",
    "concept": "#9c27b0",
    "technology": "#2196f3",
    "note": "#78909c",
}

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Knowledge Graph Explorer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0a0a0f; color: #e0e0e0;
    font-family: system-ui, sans-serif; overflow: hidden;
  }
  #graph { width: 100vw; height: 100vh; }
  #info-panel {
    display: none; position: fixed; top: 20px; right: 20px;
    width: 340px; max-height: calc(100vh - 40px); overflow-y: auto;
    background: rgba(20, 20, 30, 0.95); border: 1px solid #333;
    border-radius: 8px; padding: 16px; z-index: 10;
    font-size: 13px; line-height: 1.5;
  }
  #info-panel.visible { display: block; }
  #info-panel h2 { font-size: 15px; margin-bottom: 8px; word-break: break-word; }
  #info-panel .type-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600; margin-bottom: 8px;
  }
  #info-panel .meta { color: #999; font-size: 12px; margin-bottom: 4px; }
  #info-panel .connections { margin-top: 12px; }
  #info-panel .connections h3 { font-size: 13px; margin-bottom: 6px; color: #aaa; }
  #info-panel .conn-item {
    padding: 3px 0; font-size: 12px; color: #bbb;
    cursor: pointer; border-bottom: 1px solid #222;
  }
  #info-panel .conn-item:hover { color: #fff; }
  #close-btn {
    position: absolute; top: 8px; right: 12px;
    background: none; border: none; color: #666; cursor: pointer;
    font-size: 18px; line-height: 1;
  }
  #close-btn:hover { color: #fff; }
  #stats-bar {
    position: fixed; bottom: 12px; left: 12px;
    font-size: 12px; color: #555; z-index: 10;
  }
  #legend {
    position: fixed; bottom: 12px; right: 12px;
    font-size: 16px; color: #888; z-index: 10;
    display: flex; gap: 16px; flex-wrap: wrap;
    background: rgba(10, 10, 15, 0.5);
    padding: 8px 12px; border-radius: 6px;
  }
  .legend-item {
    display: flex; align-items: center; gap: 6px;
    cursor: pointer; transition: opacity 0.2s;
  }
  .legend-item.inactive { opacity: 0.3; }
  .legend-dot { width: 12px; height: 12px; border-radius: 50%; }
  #search-bar {
    position: fixed; top: 12px; left: 12px;
    z-index: 10; display: flex; flex-direction: column;
  }
  #search-input {
    width: 280px; padding: 8px 12px;
    background: rgba(20, 20, 30, 0.9);
    border: 1px solid #333; border-radius: 6px;
    color: #e0e0e0; font-size: 14px; outline: none;
  }
  #search-input:focus { border-color: #555; }
  #search-input::placeholder { color: #555; }
  #search-results {
    display: none; width: 280px; max-height: 300px;
    overflow-y: auto; margin-top: 4px;
    background: rgba(20, 20, 30, 0.95);
    border: 1px solid #333; border-radius: 6px;
  }
  #search-results.visible { display: block; }
  #status-line {
    position: fixed; top: 52px; left: 12px;
    font-size: 12px; color: #888; z-index: 10;
    transition: opacity 0.3s;
  }
  #status-line.hidden { opacity: 0; }
  #response-panel {
    display: none; position: fixed; top: 80px; left: 12px;
    width: 420px; max-height: calc(100vh - 120px); overflow-y: auto;
    background: rgba(20, 20, 30, 0.95); border: 1px solid #333;
    border-radius: 8px; padding: 16px; z-index: 10;
    font-size: 13px; line-height: 1.6; color: #ddd;
  }
  #response-panel.visible { display: block; }
  #response-panel .close-response {
    position: absolute; top: 8px; right: 12px;
    background: none; border: none; color: #666;
    cursor: pointer; font-size: 18px; line-height: 1;
  }
  #response-panel .close-response:hover { color: #fff; }
  .citation {
    color: #00bcd4; cursor: pointer; text-decoration: underline;
    text-underline-offset: 2px;
  }
  .citation:hover { color: #4dd0e1; }
  #response-content h1, #response-content h2, #response-content h3 {
    font-size: 14px; font-weight: 600; margin: 12px 0 6px;
    color: #eee;
  }
  #response-content h1 { font-size: 16px; }
  #response-content p { margin: 6px 0; }
  #response-content ul, #response-content ol {
    margin: 6px 0 6px 20px;
  }
  #response-content li { margin: 3px 0; }
  #response-content code {
    background: rgba(255,255,255,0.08); padding: 1px 4px;
    border-radius: 3px; font-size: 12px;
  }
  #response-content pre {
    background: rgba(255,255,255,0.06); padding: 10px;
    border-radius: 4px; overflow-x: auto; margin: 8px 0;
  }
  #response-content pre code { background: none; padding: 0; }
  #response-content blockquote {
    border-left: 3px solid #444; padding-left: 10px;
    color: #aaa; margin: 8px 0;
  }
  #response-content strong { color: #fff; }
  #response-content a { color: #00bcd4; }
  .search-result {
    padding: 6px 12px; cursor: pointer;
    font-size: 13px; color: #bbb;
    border-bottom: 1px solid #222;
    display: flex; align-items: center; gap: 8px;
  }
  .search-result:hover,
  .search-result.active {
    background: rgba(255,255,255,0.05); color: #fff;
  }
  .sr-dot {
    width: 8px; height: 8px;
    border-radius: 50%; flex-shrink: 0;
  }
  .sr-label {
    flex: 1; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
  }
  .sr-id { font-size: 11px; color: #555; }
</style>
</head>
<body>
<div id="search-bar">
  <input id="search-input" type="text"
    placeholder="Search nodes..." autocomplete="off">
  <div id="search-results"></div>
</div>
<div id="status-line" class="hidden"></div>
<div id="response-panel">
  <button class="close-response" onclick="hideResponsePanel()">&times;</button>
  <div id="response-content"></div>
</div>
<div id="graph"></div>
<div id="info-panel">
  <button id="close-btn">&times;</button>
  <div id="info-content"></div>
</div>
<div id="stats-bar"></div>
<div id="legend"></div>
<script src="https://unpkg.com/force-graph"></script>
<script src="https://unpkg.com/marked/marked.min.js"></script>
<script>
const GRAPH_DATA = __GRAPH_DATA__;
const NODE_COLORS = __NODE_COLORS__;
const HIGH_CONN_THRESHOLD = 5;
const isServed = location.protocol !== 'file:';

// Query-driven traversal state
let queryMode = null;       // 'explore' | 'summarize' | null
const visitedNodes = new Set();
const resultNodes = new Set();
const nodeStates = new Map();  // node_id -> 'visited' | 'result'

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
  `${GRAPH_DATA.stats.node_count} nodes \\u00b7 ${GRAPH_DATA.stats.edge_count} edges`;

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

const graph = ForceGraph()(document.getElementById('graph'))
  .graphData({ nodes: GRAPH_DATA.nodes, links: GRAPH_DATA.edges.map(e => ({
    source: e.source, target: e.target, type: e.type
  }))})
  .backgroundColor('#0a0a0f')
  .nodeId('id')
  .nodeVal('val')
  .nodeColor(n => NODE_COLORS[n.type] || '#666')
  .nodeCanvasObject((node, ctx, globalScale) => {
    const isHL = node === highlightedNode;
    const nState = nodeStates.get(node.id);
    const isResult = nState === 'result';
    const isVisited = nState === 'visited';
    const extraR = isResult ? 3 : (isHL ? 6 : 0);
    const r = Math.sqrt(node.val || 1) * 2.5 + extraR;
    let color = NODE_COLORS[node.type] || '#666';

    // Traversal state overrides
    if (isResult) color = '#00ff88';
    else if (isVisited) color = '#ffaa00';

    if (isHL || isResult || isVisited) {
      ctx.shadowColor = color;
      ctx.shadowBlur = isResult ? 30 : (isVisited ? 20 : 25);
    }
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Show label on hover, highlight, traversal, or high-connectivity nodes
    const showLabel = isHL || isResult || isVisited
      || node === hoveredNode || (node.val || 0) >= HIGH_CONN_THRESHOLD;
    if (showLabel) {
      const fontSize = Math.max(10 / globalScale, 2);
      ctx.font = `${fontSize}px system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.fillText(node.label || node.id, node.x, node.y + r + 2);
    }
  })
  .nodeCanvasObjectMode(() => 'replace')
  .linkColor(() => 'rgba(255,255,255,0.08)')
  .linkWidth(0.5)
  .linkDirectionalParticles(2)
  .linkDirectionalParticleWidth(1.5)
  .linkDirectionalParticleColor(() => 'rgba(255,255,255,0.3)')
  .linkDirectionalParticleSpeed(0.003)
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
    setTimeout(() => graph.zoomToFit(400, 40, connected), 100);
  });

// Edge labels on hover
graph.linkLabel(link => `${link.type}`);

// Zoom to fit early — skip if user already started exploring
setTimeout(() => {
  if (!focusedNode && navHistory.length === 0) {
    graph.zoomToFit(400, 40, connected);
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
  setTimeout(() => graph.zoomToFit(400, 40, connected), 100);
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
  if (props.entry_type) html += `<div class="meta">Type: ${escapeHtml(props.entry_type)}</div>`;
  if (props.tags) html += `<div class="meta">Tags: ${escapeHtml(props.tags)}</div>`;
  if (props.project_ref)
    html += `<div class="meta">Project: ${escapeHtml(props.project_ref)}</div>`;
  if (props.contributor)
    html += `<div class="meta">By: ${escapeHtml(props.contributor)}</div>`;
  if (props.confidence_level != null)
    html += `<div class="meta">Confidence: ${props.confidence_level}</div>`;
  if (props.summary) html += `<div class="meta">${escapeHtml(props.summary)}</div>`;

  const conns = adjacency.get(node.id) || [];
  if (conns.length > 0) {
    html += `<div class="connections"><h3>Connections (${conns.length})</h3>`;
    conns.slice(0, 30).forEach(c => {
      const arrow = c.dir === 'out' ? '\\u2192' : '\\u2190';
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

// Left arrow = backtrack through navigation history
const searchInput = document.getElementById('search-input');
document.addEventListener('keydown', e => {
  if (document.activeElement === searchInput) return;
  if (e.key === 'Escape') {
    focusedNode = null;
    highlightedNode = null;
    navHistory.length = 0;
    graph.nodeVisibility(nodeVisible)
      .linkVisibility(linkVisible);
    document.getElementById('info-panel')
      .classList.remove('visible');
    graph.zoomToFit(400, 40, connected);
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
  } else if (e.key === 'Enter') {
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
  return String(s).replace(/'/g, "\\\\'").replace(/"/g, '&quot;');
}

// --- Query-driven exploration (web server mode only) ---
const statusLine = document.getElementById('status-line');
const responsePanel = document.getElementById('response-panel');
const responseContent = document.getElementById('response-content');

function setStatus(msg) {
  if (!msg) {
    statusLine.classList.add('hidden');
    statusLine.textContent = '';
  } else {
    statusLine.textContent = msg;
    statusLine.classList.remove('hidden');
  }
}

function resetTraversalState() {
  queryMode = null;
  visitedNodes.clear();
  resultNodes.clear();
  nodeStates.clear();
  hideResponsePanel();
  setStatus('');
}

function markVisited(nodeId) {
  visitedNodes.add(nodeId);
  if (!nodeStates.has(nodeId) || nodeStates.get(nodeId) !== 'result') {
    nodeStates.set(nodeId, 'visited');
  }
}

function markResult(nodeId) {
  resultNodes.add(nodeId);
  nodeStates.set(nodeId, 'result');
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

function zoomToResults() {
  if (resultNodes.size === 0) return;
  graph.zoomToFit(600, 40, n => resultNodes.has(n.id));
}

function showResponsePanel(answer) {
  // Render markdown, then convert [kb-XXXXX] citations to clickable spans
  const html = marked.parse(answer).replace(
    /\\[(kb-\\d{5})\\]/g,
    function(_, id) {
      return '<span class="citation" onclick="flyToNode(\\x27' + id + '\\x27)">[' + id + ']</span>';
    }
  );
  responseContent.innerHTML = html;
  responsePanel.classList.add('visible');
}

function hideResponsePanel() {
  responsePanel.classList.remove('visible');
  responseContent.innerHTML = '';
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
      markVisited(data.args.node_id);
      const node = GRAPH_DATA.nodes.find(
        n => n.id === data.args.node_id
      );
      if (node) {
        graph.centerAt(node.x, node.y, 600);
        graph.zoom(3, 600);
      }
    }
  } else if (eventType === 'tool_result') {
    if (data.entry_ids) {
      data.entry_ids.forEach(id => markVisited(id));
      emitTraversalParticles(data.entry_ids);
    }
  } else if (eventType === 'fast_path' || eventType === 'agent_done') {
    if (data.entry_ids) {
      data.entry_ids.forEach(id => markResult(id));
      emitTraversalParticles(data.entry_ids);
    }
  } else if (eventType === 'entries') {
    if (data.entries) {
      data.entries.forEach(e => markResult(e.id));
    }
    zoomToResults();
  } else if (eventType === 'synthesis_result') {
    if (data.answer) {
      showResponsePanel(data.answer);
    }
    // Also zoom to any result nodes
    zoomToResults();
  } else if (eventType === 'stream_end') {
    setStatus('');
  }
}

async function startQuery(question) {
  resetTraversalState();
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
      const lines = buffer.split('\\n');
      buffer = lines.pop() || '';

      let currentEvent = null;
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ') && currentEvent) {
          try {
            const data = JSON.parse(line.slice(6));
            handleSSEEvent(currentEvent, data);
          } catch (e) { /* skip bad JSON */ }
          currentEvent = null;
        }
      }
    }
  } catch (err) {
    setStatus('Query failed: ' + err.message);
    setTimeout(() => setStatus(''), 3000);
  }
}
</script>
</body>
</html>
"""


def render_explorer_html(graph_data: dict[str, Any]) -> str:
    """Render a self-contained HTML file with the knowledge graph visualization.

    Args:
        graph_data: Output of extract_graph_data().

    Returns:
        Complete HTML string ready to write to a file.
    """
    data_json = json.dumps(graph_data, separators=(",", ":"))
    colors_json = json.dumps(_NODE_COLORS, separators=(",", ":"))
    return _TEMPLATE.replace("__GRAPH_DATA__", data_json).replace("__NODE_COLORS__", colors_json)
