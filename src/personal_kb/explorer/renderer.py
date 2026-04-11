"""Render the explorer HTML page, referencing static CSS and JS assets."""

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
<link rel="stylesheet" href="/static/explorer.css">
</head>
<body>
<div id="search-bar">
  <div id="search-container">
    <textarea id="search-input" rows="1"
      placeholder="Search nodes or ask a question..." autocomplete="off"></textarea>
    <div id="search-toolbar">
      <button class="toolbar-btn" onclick="openIngestModal('url')">+URL(s)</button>
      <button class="toolbar-btn" onclick="openIngestModal('file')">+File(s)</button>
    </div>
    <div id="search-results"></div>
    <div id="chat-history-list"></div>
  </div>
</div>
<div id="ingest-modal">
  <div id="ingest-modal-content">
    <h3>Ingest</h3>
    <div class="ingest-tabs">
      <button class="ingest-tab active" data-tab="url"
        onclick="switchIngestTab('url')">URLs</button>
      <button class="ingest-tab" data-tab="file" onclick="switchIngestTab('file')">Files</button>
    </div>
    <div id="ingest-tab-url" class="ingest-tab-body active">
      <div id="ingest-url-rows">
        <div class="ingest-url-row">
          <input type="url" placeholder="https://example.com/article" autocomplete="off" />
        </div>
      </div>
      <button class="ingest-url-add" onclick="addUrlRow()">+ Add URL</button>
    </div>
    <div id="ingest-tab-file" class="ingest-tab-body">
      <label class="ingest-file-pick"
        for="ingest-file-input">Choose files...</label>
      <input id="ingest-file-input" type="file"
        multiple onchange="updateFileList()" />
      <div id="ingest-file-list"></div>
    </div>
    <input id="ingest-project-input" list="ingest-project-list"
      placeholder="Project (optional)" autocomplete="off" />
    <datalist id="ingest-project-list"></datalist>
    <div id="ingest-progress">
      <div id="ingest-progress-bar"><div id="ingest-progress-fill"></div></div>
      <div id="ingest-progress-text"></div>
    </div>
    <div id="ingest-status"></div>
    <div id="ingest-modal-btns">
      <button class="ingest-cancel" onclick="closeIngestModal()">Cancel</button>
      <button class="ingest-submit" id="ingest-submit-btn" onclick="submitIngest()">Ingest</button>
    </div>
  </div>
</div>
<div id="status-line" class="hidden"></div>
<div id="response-panel">
  <button class="close-response" onclick="closeConversation()">&times;</button>
  <div id="response-content"></div>
</div>
<div id="chat-panel">
  <button id="chat-maximize-btn" class="chat-maximize" onclick="toggleMaximize()"></button>
  <button class="chat-close" onclick="closeConversation()">&times;</button>
  <div id="chat-header">
    <span id="chat-mode-badge" class="chat-mode-badge"></span>
    <span id="chat-query" class="chat-query"></span>
  </div>
  <div id="chat-messages"></div>
  <div id="chat-input-bar">
    <textarea id="chat-input" rows="1"
      placeholder="Ask a follow-up..." autocomplete="off"></textarea>
    <button id="chat-send" onclick="sendChatMessage()">Send</button>
  </div>
</div>
<div id="graph"></div>
<div id="info-panel">
  <button id="close-btn">&times;</button>
  <div id="info-content"></div>
</div>
<div id="stats-bar"></div>
<div id="legend"></div>
<script src="https://unpkg.com/force-graph"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/lib/marked.umd.js"></script>
<script>
const GRAPH_DATA = __GRAPH_DATA__;
const NODE_COLORS = __NODE_COLORS__;
</script>
<script src="/static/explorer.js"></script>
</body>
</html>
"""


def render_explorer_html(graph_data: dict[str, Any]) -> str:
    """Render the explorer HTML page.

    Injects graph data and node colors as inline JS constants,
    then loads the main explorer.js from static files.

    Args:
        graph_data: Output of extract_graph_data().

    Returns:
        Complete HTML string.
    """
    data_json = json.dumps(graph_data, separators=(",", ":"))
    colors_json = json.dumps(_NODE_COLORS, separators=(",", ":"))
    return _TEMPLATE.replace("__GRAPH_DATA__", data_json).replace("__NODE_COLORS__", colors_json)
