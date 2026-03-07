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
    position: fixed; top: 20px; right: 20px;
    width: 340px; max-height: calc(100vh - 40px); overflow-y: auto;
    background: rgba(20, 20, 30, 0.95); border: 1px solid #333;
    border-radius: 8px; padding: 16px; z-index: 10;
    font-size: 13px; line-height: 1.5;
    opacity: 0; pointer-events: none;
    transform: scaleY(0.05) scaleX(1);
    transform-origin: top right;
    transition: opacity 0.3s ease, transform 0.3s ease;
  }
  #info-panel.visible {
    display: flex; flex-direction: column;
    opacity: 1; pointer-events: auto;
    transform: scaleY(1) scaleX(1);
  }
  #info-panel h2 { font-size: 15px; margin-bottom: 8px; word-break: break-word; }
  #info-panel .type-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600; margin-bottom: 8px;
  }
  #info-panel .meta { color: #999; font-size: 12px; margin-bottom: 4px; }
  #info-panel .meta b { color: #fff; font-weight: 600; }
  #info-panel .entry-details {
    margin-top: 12px; border-top: 1px solid #333; padding-top: 8px;
  }
  #info-panel .details-toggle {
    background: none; border: none; color: #00bcd4; cursor: pointer;
    font-size: 12px; padding: 0; display: flex; align-items: center; gap: 4px;
  }
  #info-panel .details-toggle:hover { color: #4dd0e1; }
  #info-panel .details-toggle .arrow {
    display: inline-block; transition: transform 0.2s;
    font-size: 10px;
  }
  #info-panel .details-toggle .arrow.open { transform: rotate(90deg); }
  #info-panel .details-body {
    display: none; margin-top: 8px; font-size: 12px;
    line-height: 1.6; color: #ccc;
  }
  #info-panel .details-body.visible { display: block; }
  #info-panel .details-body h1, #info-panel .details-body h2,
  #info-panel .details-body h3 {
    font-size: 13px; font-weight: 600; margin: 10px 0 4px; color: #eee;
  }
  #info-panel .details-body p { margin: 4px 0; }
  #info-panel .details-body ul, #info-panel .details-body ol {
    margin: 4px 0 4px 16px;
  }
  #info-panel .details-body li { margin: 2px 0; }
  #info-panel .details-body code {
    background: rgba(255,255,255,0.08); padding: 1px 3px;
    border-radius: 3px; font-size: 11px;
  }
  #info-panel .details-body pre {
    background: rgba(255,255,255,0.06); padding: 8px;
    border-radius: 4px; overflow-x: auto; margin: 6px 0;
  }
  #info-panel .details-body pre code { background: none; padding: 0; }
  #info-panel .details-body strong { color: #fff; }
  #info-panel .connections { margin-top: 12px; }
  #info-panel .connections h3 { font-size: 13px; margin-bottom: 6px; color: #aaa; }
  #info-panel .conn-item {
    padding: 3px 0; font-size: 12px; color: #bbb;
    cursor: pointer; border-bottom: 1px solid #222;
  }
  #info-panel .conn-item:hover { color: #fff; }
  #close-btn {
    position: absolute; top: 10px; right: 10px;
    width: 28px; height: 28px; border-radius: 50%;
    background: rgba(80, 80, 90, 0.8); border: none;
    color: #fff; cursor: pointer; font-size: 15px;
    font-weight: 700; line-height: 28px; text-align: center;
    z-index: 1; transition: background 0.15s;
    display: flex; align-items: center; justify-content: center;
  }
  #close-btn:hover { background: rgba(120, 120, 130, 0.9); }
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
    transition: opacity 0.25s ease, transform 0.25s ease;
  }
  #search-bar.hidden-for-chat {
    opacity: 0; pointer-events: none; transform: translateY(4px);
  }
  #search-container {
    width: 420px; background: rgba(20, 20, 30, 0.9);
    border: 1px solid #333; border-radius: 6px;
    overflow: hidden;
  }
  #search-container:focus-within { border-color: #555; }
  #search-input {
    width: 100%; padding: 8px 12px; box-sizing: border-box;
    background: transparent; border: none;
    color: #e0e0e0; font-size: 14px; outline: none;
    font-family: inherit; resize: none; overflow: hidden;
    line-height: 1.4; max-height: 120px;
  }
  #search-input::placeholder { color: #555; }
  #search-toolbar {
    display: flex; gap: 4px; padding: 4px 8px 6px;
    border-top: 1px solid #2a2a3a;
  }
  .toolbar-btn {
    padding: 2px 8px; font-size: 11px;
    background: transparent; border: 1px solid #3a3a4a;
    border-radius: 3px; color: #777; cursor: pointer;
    font-family: inherit; transition: background 0.15s, color 0.15s, border-color 0.15s;
  }
  .toolbar-btn:hover { background: rgba(50, 50, 70, 0.6); color: #bbb; border-color: #555; }
  .toolbar-btn:disabled { opacity: 0.4; cursor: default; }
  #ingest-modal {
    display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.6); z-index: 100;
    align-items: center; justify-content: center;
  }
  #ingest-modal.visible { display: flex; }
  #ingest-modal-content {
    background: rgba(25, 25, 35, 0.98); border: 1px solid #444;
    border-radius: 8px; padding: 20px; width: 480px;
    color: #ddd; font-size: 13px;
  }
  #ingest-modal-content h3 {
    margin: 0 0 12px 0; font-size: 15px; color: #e0e0e0;
  }
  .ingest-tabs {
    display: flex; gap: 0; margin-bottom: 12px;
    border-bottom: 1px solid #3a3a4a;
  }
  .ingest-tab {
    padding: 6px 16px; font-size: 12px; cursor: pointer;
    background: transparent; border: none; border-bottom: 2px solid transparent;
    color: #888; font-family: inherit; transition: color 0.15s, border-color 0.15s;
  }
  .ingest-tab:hover { color: #bbb; }
  .ingest-tab.active { color: #a8d8f0; border-bottom-color: #1a5276; }
  .ingest-tab-body { display: none; }
  .ingest-tab-body.active { display: block; }
  .ingest-url-row {
    display: flex; gap: 6px; margin-bottom: 6px; align-items: center;
  }
  .ingest-url-row input {
    flex: 1; padding: 7px 10px; box-sizing: border-box;
    background: rgba(15, 15, 25, 0.9); border: 1px solid #555;
    border-radius: 4px; color: #e0e0e0; font-size: 13px;
    font-family: inherit; outline: none;
  }
  .ingest-url-row input:focus { border-color: #777; }
  .ingest-url-row input::placeholder { color: #555; }
  .ingest-url-remove {
    background: transparent; border: none; color: #666; cursor: pointer;
    font-size: 16px; padding: 0 4px; line-height: 1;
  }
  .ingest-url-remove:hover { color: #e57373; }
  .ingest-url-add {
    background: transparent; border: 1px dashed #3a3a4a; border-radius: 4px;
    color: #666; cursor: pointer; padding: 4px 12px; font-size: 12px;
    font-family: inherit; margin-bottom: 4px;
  }
  .ingest-url-add:hover { border-color: #555; color: #999; }
  #ingest-file-input { display: none; }
  .ingest-file-pick {
    display: inline-block; padding: 8px 16px; cursor: pointer;
    background: rgba(15, 15, 25, 0.9); border: 1px dashed #555;
    border-radius: 4px; color: #888; font-size: 12px;
    font-family: inherit; text-align: center; width: 100%;
    box-sizing: border-box;
  }
  .ingest-file-pick:hover { border-color: #777; color: #bbb; }
  #ingest-file-list {
    margin-top: 8px; font-size: 12px; color: #aaa;
    max-height: 100px; overflow-y: auto;
  }
  #ingest-file-list div { padding: 2px 0; }
  #ingest-project-input {
    width: 100%; padding: 6px 10px; box-sizing: border-box; margin-top: 8px;
    background: rgba(15, 15, 25, 0.9); border: 1px solid #555;
    border-radius: 4px; color: #e0e0e0; font-size: 12px;
    font-family: inherit; outline: none;
  }
  #ingest-project-input:focus { border-color: #777; }
  #ingest-project-input::placeholder { color: #555; }
  #ingest-progress {
    display: none; margin-top: 10px;
  }
  #ingest-progress.visible { display: block; }
  #ingest-progress-bar {
    width: 100%; height: 4px; background: #2a2a3a; border-radius: 2px;
    overflow: hidden;
  }
  #ingest-progress-fill {
    height: 100%; width: 0%; background: #1a5276;
    transition: width 0.3s ease;
  }
  #ingest-progress-text {
    margin-top: 4px; font-size: 11px; color: #888;
  }
  #ingest-modal-btns {
    display: flex; gap: 8px; margin-top: 14px; justify-content: flex-end;
  }
  #ingest-modal-btns button {
    padding: 6px 16px; border-radius: 4px; font-size: 12px;
    cursor: pointer; font-family: inherit; border: 1px solid #555;
  }
  .ingest-cancel { background: transparent; color: #aaa; }
  .ingest-cancel:hover { color: #ddd; }
  .ingest-submit { background: #1a5276; color: #a8d8f0; border-color: #1a5276 !important; }
  .ingest-submit:hover { background: #21689a; }
  .ingest-submit:disabled { opacity: 0.4; cursor: default; }
  #ingest-status {
    margin-top: 10px; font-size: 12px; color: #888;
    min-height: 16px;
  }
  #ingest-status.error { color: #e57373; }
  #ingest-status.success { color: #81c784; }
  #search-results {
    display: none; width: 100%; max-height: 300px;
    overflow-y: auto; margin-top: 4px;
    background: rgba(20, 20, 30, 0.95);
    border: 1px solid #333; border-radius: 6px;
    box-sizing: border-box;
  }
  #search-results.visible { display: block; }
  #status-line {
    position: fixed; top: 80px; left: 12px;
    font-size: 12px; color: #888; z-index: 10;
    transition: opacity 0.3s;
    display: none;
  }
  #status-line.hidden { opacity: 0; }
  #response-panel {
    display: none; position: fixed; top: 12px; left: 12px;
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
  #response-panel .summarize-btn {
    display: inline-block; margin-top: 12px; padding: 6px 14px;
    background: #1a5276; border: none; border-radius: 6px;
    color: #e0e8f0; cursor: pointer; font-size: 13px;
  }
  #response-panel .summarize-btn:hover { background: #1f6a9a; }
  #response-panel { overflow-wrap: break-word; word-break: break-word; }
  #response-content table, .chat-msg table {
    width: 100%; border-collapse: collapse; margin: 8px 0;
    font-size: 12px; table-layout: fixed; word-break: break-word;
  }
  #response-content th, #response-content td,
  .chat-msg th, .chat-msg td {
    border: 1px solid #444; padding: 4px 8px; text-align: left;
    overflow-wrap: break-word;
  }
  #response-content th, .chat-msg th {
    background: rgba(255,255,255,0.06); font-weight: 600; color: #eee;
  }
  /* Chat panel */
  #chat-panel {
    position: fixed; top: 12px; left: 12px;
    width: 420px; max-height: calc(100vh - 24px);
    background: rgba(20, 20, 30, 0.95); border: 1px solid #333;
    border-radius: 8px; z-index: 11;
    font-size: 13px; line-height: 1.6; color: #ddd;
    flex-direction: column; display: flex;
    /* Collapsed state: looks like search bar */
    opacity: 0; pointer-events: none;
    transform: scaleY(0.05) scaleX(1);
    transform-origin: top left;
    transition: opacity 0.3s ease, transform 0.3s ease,
      width 0.3s ease, left 0.3s ease;
  }
  #chat-panel.visible {
    opacity: 1; pointer-events: auto;
    transform: scaleY(1) scaleX(1);
  }
  #chat-panel.maximized {
    width: min(60vw, 900px);
    left: calc((100vw - min(60vw, 900px)) / 2);
  }
  #chat-header {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 76px 10px 14px;
    border-bottom: 1px solid #333;
    flex-shrink: 0;
  }
  #chat-header .chat-mode-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    flex-shrink: 0;
  }
  #chat-header .chat-mode-badge.summarize {
    background: #1a5276; color: #a8d8f0;
  }
  #chat-header .chat-mode-badge.explore {
    background: #4a3520; color: #f0c878;
  }
  #chat-header .chat-query {
    font-size: 12px; color: #999;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    flex: 1; min-width: 0;
  }
  #chat-panel .chat-close {
    position: absolute; top: 8px; right: 10px;
    width: 28px; height: 28px; border-radius: 50%;
    background: rgba(80, 80, 90, 0.8); border: none;
    color: #fff; cursor: pointer; font-size: 15px;
    font-weight: 700; line-height: 28px; text-align: center;
    z-index: 1; transition: background 0.15s;
    display: flex; align-items: center; justify-content: center;
  }
  #chat-panel .chat-close:hover { background: rgba(120, 120, 130, 0.9); }
  #chat-panel .chat-maximize {
    position: absolute; top: 8px; right: 42px;
    width: 28px; height: 28px; border-radius: 50%;
    background: rgba(80, 80, 90, 0.8); border: none;
    color: #fff; cursor: pointer;
    z-index: 1; transition: background 0.15s;
    display: flex; align-items: center; justify-content: center;
  }
  #chat-panel .chat-maximize:hover { background: rgba(120, 120, 130, 0.9); }
  #chat-messages {
    flex: 1; overflow-y: auto; padding: 16px;
    display: flex; flex-direction: column; gap: 10px;
    min-height: 0;
  }
  .chat-msg {
    max-width: 85%; padding: 8px 12px;
    border-radius: 12px; font-size: 13px;
    line-height: 1.5; overflow-wrap: break-word; word-break: break-word;
  }
  .chat-msg.user {
    align-self: flex-end; background: #1a5276;
    color: #e0e8f0; border-bottom-right-radius: 4px;
  }
  .chat-msg.assistant {
    align-self: flex-start; background: #2a2a3a;
    color: #ddd; border-bottom-left-radius: 4px;
    position: relative;
  }
  .chat-msg .copy-btn {
    position: absolute; top: 6px; right: 6px;
    padding: 3px 5px; background: rgba(60, 60, 70, 0.8); border: none;
    border-radius: 4px; color: #888; cursor: pointer;
    opacity: 0; transition: opacity 0.15s;
    display: flex; align-items: center; justify-content: center;
  }
  .chat-msg:hover .copy-btn { opacity: 1; }
  .chat-msg .copy-btn:hover { color: #fff; background: rgba(100, 100, 110, 0.9); }
  .chat-msg .copy-btn.copied { opacity: 1; color: #00ff88; }
  .chat-msg h1, .chat-msg h2, .chat-msg h3 {
    font-size: 14px; font-weight: 600;
    margin: 8px 0 4px; color: #eee;
  }
  .chat-msg h1 { font-size: 15px; }
  .chat-msg p { margin: 4px 0; }
  .chat-msg ul, .chat-msg ol { margin: 4px 0 4px 16px; }
  .chat-msg li { margin: 2px 0; }
  .chat-msg code {
    background: rgba(255,255,255,0.08); padding: 1px 4px;
    border-radius: 3px; font-size: 12px;
  }
  .chat-msg pre {
    background: rgba(255,255,255,0.06); padding: 8px;
    border-radius: 4px; overflow-x: auto; margin: 6px 0;
  }
  .chat-msg pre code { background: none; padding: 0; }
  .chat-msg strong { color: #fff; }
  .chat-msg a { color: #00bcd4; }
  .chat-typing {
    align-self: flex-start; color: #888;
    font-size: 12px; padding: 4px 12px;
  }
  .chat-typing::after {
    content: ''; animation: dots 1.4s steps(4, end) infinite;
  }
  @keyframes dots {
    0% { content: ''; }
    25% { content: '.'; }
    50% { content: '..'; }
    75% { content: '...'; }
  }
  #chat-input-bar {
    display: flex; gap: 8px; padding: 10px 12px;
    border-top: 1px solid #333;
  }
  #chat-input {
    flex: 1; padding: 8px 10px;
    background: rgba(30, 30, 40, 0.9);
    border: 1px solid #444; border-radius: 6px;
    color: #e0e0e0; font-size: 13px; outline: none;
    font-family: inherit; resize: none; overflow: hidden;
    line-height: 1.4; max-height: 120px;
  }
  #chat-input:focus { border-color: #666; }
  #chat-input::placeholder { color: #555; }
  #chat-send {
    padding: 8px 14px; background: #1a5276;
    border: none; border-radius: 6px;
    color: #e0e8f0; cursor: pointer; font-size: 13px;
  }
  #chat-send:hover { background: #1f6a9a; }
  #chat-send:disabled { opacity: 0.5; cursor: default; }
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
  <div id="search-container">
    <textarea id="search-input" rows="1"
      placeholder="Search nodes or ask a question..." autocomplete="off"></textarea>
    <div id="search-toolbar">
      <button class="toolbar-btn" onclick="openIngestModal('url')">+URL(s)</button>
      <button class="toolbar-btn" onclick="openIngestModal('file')">+File(s)</button>
    </div>
  </div>
  <div id="search-results"></div>
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
      <label class="ingest-file-pick" for="ingest-file-input">Choose .txt or .md files...</label>
      <input id="ingest-file-input" type="file" multiple
        accept=".txt,.md" onchange="updateFileList()" />
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
    btn.innerHTML = '\\u2713';
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
      + '<button class="details-toggle" onclick="toggleDetails(this, \\x27'
      + escapeAttr(node.id) + '\\x27)">'
      + '<span class="arrow">\\u25b6</span> Full Entry\\u2026</button>'
      + '<div class="details-body"></div></div>';
  }

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
    arrow.textContent = '\\u23f3';
    try {
      const resp = await fetch('/api/entry/' + encodeURIComponent(entryId));
      if (!resp.ok) throw new Error(resp.statusText);
      const data = await resp.json();
      body.innerHTML = renderMarkdown(data.knowledge_details || '(no content)');
      body.dataset.loaded = '1';
    } catch (e) {
      body.innerHTML = '<em>Failed to load entry</em>';
    }
    arrow.textContent = '\\u25b6';
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
  } else if (e.key === 'Enter' && !e.shiftKey) {
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
  return String(s).replace(/'/g, "\\\\'").replace(/"/g, '&quot;');
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
  return '<p>' + escapeHtml(text).replace(/\\n/g, '<br>') + '</p>';
}

function showResponsePanel(answer) {
  // Render markdown first, then replace [kb-XXXXX] citations in output.
  // Brackets survive markdown rendering (no matching link definition).
  const html = renderMarkdown(answer).replace(
    /\\[(kb-\\d{5})\\]/g,
    function(_, id) {
      return '<span class="citation" onclick="flyToNode(\\x27' + id + '\\x27)">[' + id + ']</span>';
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
      let md = '**Found ' + data.entries.length + ' entries:**\\n\\n';
      data.entries.forEach(e => {
        md += '- **[' + e.id + ']** ' + e.short_title;
        if (e.context) md += ' — _' + e.context + '_';
        md += '\\n';
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
            if (e.key === 'Enter' && !e.shiftKey) {
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
      const lines = buffer.split('\\n');
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
let chatBusy = false;

function addChatCitation(html) {
  return html.replace(
    /\\[(kb-\\d{5})\\]/g,
    function(_, id) {
      return '<span class="citation" onclick="flyToNode(\\x27'
        + id + '\\x27)">[' + id + ']</span>';
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
  chatSessionId = null;
  chatSeedQuestion = question;
  chatSeedAnswer = answer;
  chatSeedEntryIds = entryIds || [];
  chatMessages.innerHTML = '';

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
  chatMessages.innerHTML = '';
  document.getElementById('info-panel').classList.remove('visible');
  // Show search bar
  document.getElementById('search-bar').classList.remove('hidden-for-chat');
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
    const payload = {message: msg};
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

      const lines = buffer.split('\\n');
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
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
});

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
  if (e.key === 'Enter' && e.target.tagName === 'INPUT') { e.preventDefault(); submitIngest(); }
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
      const text = await f.text();
      items.push({type: 'file', name: f.name, content: text});
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

      const lines = buffer.split('\\n');
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

            // Highlight new entries on graph
            allEntryIds.forEach(id => markResult(id));
            if (allEntryIds.length > 0) {
              emitTraversalParticles(allEntryIds);
              setTimeout(() => {
                graph.zoomToFit(500, 60, n => allEntryIds.includes(n.id));
                setTimeout(clampZoom, 550);
              }, 300);
            }
            setTimeout(closeIngestModal, 2000);
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
