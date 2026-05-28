#!/usr/bin/env python3
"""
PS-style eyebrow patch blending UI.

Run:
  python tools/brow_blend_ui.py --host 0.0.0.0 --port 8799
"""

import argparse
import base64
import io
import os
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageOps


DEFAULT_OUTPUT_DIR = Path("/media/jinsha/娱乐1/眉毛/换眉输出")
DEFAULT_PATCH_DIR = Path("/media/jinsha/娱乐1/眉毛/眉毛贴片库")


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PS式换眉贴合工具</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101319;
      --panel: #1b2029;
      --panel2: #222833;
      --line: #35404d;
      --text: #eef3f8;
      --muted: #9da8b6;
      --accent: #3d8ee8;
      --ok: #2e8a4e;
      --danger: #8d3940;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 52px;
      padding: 0 16px;
      display: flex;
      align-items: center;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      background: #151922;
    }
    h1 {
      margin: 0;
      font-size: 17px;
      font-weight: 650;
    }
    main {
      display: grid;
      grid-template-columns: minmax(440px, 0.72fr) minmax(520px, 1fr) 360px;
      min-height: calc(100vh - 52px);
    }
    section {
      min-width: 0;
      padding: 14px;
      overflow: auto;
      border-right: 1px solid var(--line);
    }
    aside {
      padding: 14px;
      background: var(--panel);
      overflow: auto;
    }
    .title {
      height: 30px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }
    .board {
      position: relative;
      width: fit-content;
      max-width: 100%;
      margin: 0 auto 12px;
      background: #2a303c;
      border: 1px solid var(--line);
    }
    canvas {
      display: block;
      touch-action: none;
    }
    #sourcePaintCanvas, #targetHitCanvas {
      position: absolute;
      inset: 0;
      cursor: crosshair;
    }
    #targetHitCanvas {
      cursor: grab;
    }
    #targetHitCanvas.dragging {
      cursor: grabbing;
    }
    #resultImg {
      display: none;
      max-width: 100%;
      margin: 0 auto;
      border: 1px solid var(--line);
      background-color: #ddd;
      background-image:
        linear-gradient(45deg, #bbb 25%, transparent 25%),
        linear-gradient(-45deg, #bbb 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #bbb 75%),
        linear-gradient(-45deg, transparent 75%, #bbb 75%);
      background-size: 22px 22px;
      background-position: 0 0, 0 11px, 11px -11px, -11px 0;
    }
    .card {
      padding: 12px;
      margin-bottom: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel2);
    }
    .card h2 {
      margin: 0 0 10px;
      font-size: 14px;
      font-weight: 650;
    }
    label {
      display: block;
      margin: 10px 0 5px;
      color: var(--muted);
      font-size: 12px;
    }
    input[type="text"], input[type="number"], select {
      width: 100%;
      padding: 8px 9px;
      border: 1px solid var(--line);
      border-radius: 7px;
      color: var(--text);
      background: #0f131a;
      font-size: 13px;
    }
    input[type="file"], input[type="range"] { width: 100%; }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .row3 {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
    }
    .valueInput {
      margin-top: 6px;
    }
    button {
      border: 1px solid var(--line);
      background: #2a3341;
      color: var(--text);
      border-radius: 7px;
      padding: 8px 10px;
      font-weight: 650;
      cursor: pointer;
    }
    button.primary {
      background: #245b9e;
      border-color: #3479c7;
    }
    button.ok {
      background: #206a3b;
      border-color: #32985a;
    }
    button.danger {
      background: #6d3036;
      border-color: #974850;
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .checkLine {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .checkLine input {
      width: auto;
    }
    .tools {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      padding: 4px 8px;
      min-height: 28px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: #0f131a;
      font-size: 12px;
    }
    .value {
      float: right;
      color: var(--text);
      font-variant-numeric: tabular-nums;
    }
    .status {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      word-break: break-all;
    }
    .status strong { color: var(--text); }
    .swatch {
      width: 28px;
      height: 28px;
      border-radius: 7px;
      border: 1px solid var(--line);
      background: #777;
      flex: 0 0 auto;
    }
    .swatchRow {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 1200px) {
      main { grid-template-columns: 1fr; }
      section, aside { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <h1>PS式换眉贴合工具</h1>
    <span class="pill" id="sourceInfo">来源未载入</span>
    <span class="pill" id="targetInfo">目标未载入</span>
  </header>

  <main>
    <section>
      <div class="title">
        <strong>1. 来源眉毛</strong>
        <span>涂眉毛和周边一点皮肤，像视频里套索取脸一样</span>
      </div>
      <div class="board" id="sourceBoard">
        <canvas id="sourceCanvas"></canvas>
        <canvas id="sourcePaintCanvas"></canvas>
      </div>
      <div class="card">
        <label>来源图片路径</label>
        <input id="sourcePath" type="text" value="/media/jinsha/娱乐1/眉毛/IMG_4157.JPG" />
        <div class="row" style="margin-top:8px;">
          <button class="primary" id="loadSourcePath">打开来源</button>
          <button id="fitSource">适合窗口</button>
        </div>
        <label>浏览器选择来源</label>
        <input id="sourceFile" type="file" accept="image/*" />
      </div>
      <div class="card">
        <h2>来源选区</h2>
        <div class="tools">
          <button class="ok" id="brushBtn">画笔</button>
          <button id="eraserBtn">橡皮</button>
          <button id="undoBtn">撤销</button>
          <button class="danger" id="clearMaskBtn">清空</button>
        </div>
        <label>画笔大小 <span class="value" id="brushValue">44</span></label>
        <input id="brushSize" type="range" min="4" max="220" value="44" />
        <label>来源缩放 <span class="value" id="sourceZoomValue">100%</span></label>
        <input id="sourceZoom" type="range" min="20" max="220" value="100" />
      </div>
    </section>

    <section>
      <div class="title">
        <strong>2. 目标脸</strong>
        <span>贴片会直接显示在这里，按住眉毛拖动位置</span>
      </div>
      <div class="board" id="targetBoard">
        <canvas id="targetCanvas"></canvas>
        <canvas id="targetHitCanvas"></canvas>
      </div>
      <div class="title" id="resultTitle" style="display:none;">
        <strong>3. 合成预览</strong>
        <span>仅预览，不保存</span>
      </div>
      <img id="resultImg" alt="" />
      <div class="card">
        <label>目标图片路径</label>
        <input id="targetPath" type="text" value="/media/jinsha/娱乐1/眉毛/IMG_4157.JPG" />
        <div class="row" style="margin-top:8px;">
          <button class="primary" id="loadTargetPath">打开目标</button>
          <button id="fitTarget">适合窗口</button>
        </div>
        <label>浏览器选择目标</label>
        <input id="targetFile" type="file" accept="image/*" />
      </div>
    </section>

    <aside>
      <div class="card">
        <h2>贴合</h2>
        <div class="tools" style="margin-bottom:10px;">
          <button class="ok" id="moveTargetBtn">拖动贴片</button>
          <button id="eyedropperBtn">吸取肤色</button>
        </div>
        <div class="tools" style="margin-bottom:10px;">
          <button class="ok" id="primaryPatchBtn">主眉</button>
          <button id="secondaryPatchBtn">副眉</button>
          <button id="copyPatchBtn">复制一份</button>
          <button class="danger" id="removeSecondBtn">删除副眉</button>
        </div>
        <label class="checkLine"><input id="flipX" type="checkbox" /> 水平翻转当前眉</label>
        <label>中心 X <span class="value" id="xValue">0</span></label>
        <input id="posX" type="range" min="0" max="100" value="50" />
        <input id="posXNum" class="valueInput" type="number" min="0" step="1" value="0" />
        <label>中心 Y <span class="value" id="yValue">0</span></label>
        <input id="posY" type="range" min="0" max="100" value="35" />
        <input id="posYNum" class="valueInput" type="number" min="0" step="1" value="0" />
        <label>缩放 <span class="value" id="scaleValue">100%</span></label>
        <input id="scale" type="range" min="10" max="300" value="100" />
        <input id="scaleNum" class="valueInput" type="number" min="10" max="300" step="1" value="100" />
        <label>旋转 <span class="value" id="rotationValue">0°</span></label>
        <input id="rotation" type="range" min="-45" max="45" value="0" />
        <input id="rotationNum" class="valueInput" type="number" min="-45" max="45" step="0.5" value="0" />
        <div class="row3" style="margin-top:8px;">
          <button id="leftBtn">左</button>
          <button id="upBtn">上</button>
          <button id="rightBtn">右</button>
          <button id="smallBtn">缩小</button>
          <button id="downBtn">下</button>
          <button id="bigBtn">放大</button>
        </div>
      </div>

      <div class="card">
        <h2>融合</h2>
        <label>融合模式</label>
        <select id="blendMode">
          <option value="normal_clone">PS式融合</option>
          <option value="alpha">羽化贴片</option>
          <option value="mixed_clone">PS式保边融合</option>
        </select>
        <label>羽化 <span class="value" id="featherValue">18</span></label>
        <input id="feather" type="range" min="0" max="100" value="18" />
        <label>边缘收缩 <span class="value" id="contractValue">0</span></label>
        <input id="contract" type="range" min="0" max="60" value="0" />
        <label>贴片透明度 <span class="value" id="opacityValue">100%</span></label>
        <input id="opacity" type="range" min="0" max="100" value="100" />
        <label>肤色匹配 <span class="value" id="colorValue">35%</span></label>
        <input id="colorMatch" type="range" min="0" max="100" value="35" />
        <label>底色融合 <span class="value" id="skinTintValue">25%</span></label>
        <input id="skinTint" type="range" min="0" max="100" value="25" />
        <div class="swatchRow">
          <span id="skinSwatch" class="swatch"></span>
          <span id="skinText">未吸取目标肤色</span>
        </div>
        <label>眉毛保护 <span class="value" id="protectValue">90%</span></label>
        <input id="protectDark" type="range" min="0" max="100" value="90" />
        <label>输出留边 <span class="value" id="padValue">0</span></label>
        <input id="pad" type="range" min="0" max="100" value="0" />
        <button class="primary" id="previewBtn" style="width:100%; margin-top:12px;">生成精细融合</button>
      </div>

      <div class="card">
        <h2>保存</h2>
        <label>输出目录</label>
        <input id="outputDir" type="text" value="/media/jinsha/娱乐1/眉毛/换眉输出" />
        <label>文件名</label>
        <input id="outputName" type="text" placeholder="留空自动命名" />
        <button class="ok" id="saveBtn" style="width:100%; margin-top:10px;">保存当前合成</button>
        <div class="status" id="status" style="margin-top:10px;">等待载入图片。</div>
      </div>

      <div class="card">
        <h2>贴片库</h2>
        <label>贴片模式</label>
        <select id="patchMode">
          <option value="hair_only">只保留眉毛线条</option>
          <option value="soft_patch">带羽化皮肤过渡</option>
        </select>
        <label>贴片目录</label>
        <input id="patchOutputDir" type="text" value="/media/jinsha/娱乐1/眉毛/眉毛贴片库" />
        <label>贴片文件名</label>
        <input id="patchOutputName" type="text" placeholder="留空自动命名" />
        <button class="primary" id="savePatchBtn" style="width:100%; margin-top:10px;">保存透明眉毛贴片</button>
        <button class="ok" id="saveTunedPatchBtn" style="width:100%; margin-top:10px;">保存当前调好贴片</button>
      </div>
    </aside>
  </main>

  <script>
    const sourceCanvas = document.getElementById('sourceCanvas');
    const sourcePaintCanvas = document.getElementById('sourcePaintCanvas');
    const targetCanvas = document.getElementById('targetCanvas');
    const targetHitCanvas = document.getElementById('targetHitCanvas');
    const sourceCtx = sourceCanvas.getContext('2d');
    const sourcePaintCtx = sourcePaintCanvas.getContext('2d');
    const targetCtx = targetCanvas.getContext('2d');
    const targetHitCtx = targetHitCanvas.getContext('2d');
    const maskCanvas = document.createElement('canvas');
    const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
    const targetOriginalCanvas = document.createElement('canvas');
    const targetOriginalCtx = targetOriginalCanvas.getContext('2d', { willReadFrequently: true });
    const resultImg = document.getElementById('resultImg');
    const statusEl = document.getElementById('status');
    const sourceInfo = document.getElementById('sourceInfo');
    const targetInfo = document.getElementById('targetInfo');

    let sourceImage = null;
    let targetImage = null;
    let sourceData = '';
    let targetData = '';
    let targetPreviewImage = null;
    let sourceName = 'source';
    let targetName = 'target';
    let sourceTool = 'brush';
    let sourceDrawing = false;
    let lastPoint = null;
    let undoStack = [];
    let previewTimer = null;
    let hasPreview = false;
    let sourceDisplayScale = 1;
    let targetDisplayScale = 1;
    let targetTool = 'move';
    let targetDragging = false;
    let targetDragOffset = { xPct: 0, yPct: 0 };
    let skinSample = null;
    let activePatch = 'primary';
    let maskBBoxCache = null;
    let processBusy = false;
    let pendingPreview = false;
    let patches = {
      primary: { enabled: true, xPct: 50, yPct: 35, scale: 100, rotation: 0, flip: false },
      secondary: { enabled: false, xPct: 58, yPct: 35, scale: 100, rotation: 0, flip: true }
    };

    function setStatus(text) { statusEl.innerHTML = text; }

    function setVal(id, v) {
      const el = document.getElementById(id);
      const min = Number(el.min || 0);
      const max = Number(el.max || 100);
      el.value = Math.max(min, Math.min(max, Number(v)));
      updateLabels();
    }

    function invalidatePreview() {
      targetPreviewImage = null;
      hasPreview = false;
    }

    function activePatchData() {
      return patches[activePatch];
    }

    function syncControlsFromPatch() {
      const p = activePatchData();
      document.getElementById('posX').value = p.xPct;
      document.getElementById('posY').value = p.yPct;
      document.getElementById('scale').value = p.scale;
      document.getElementById('rotation').value = p.rotation;
      document.getElementById('flipX').checked = !!p.flip;
      updateLabels();
      drawTargetMarker();
    }

    function syncPatchFromControls() {
      const p = activePatchData();
      p.xPct = Number(document.getElementById('posX').value);
      p.yPct = Number(document.getElementById('posY').value);
      p.scale = Number(document.getElementById('scale').value);
      p.rotation = Number(document.getElementById('rotation').value);
      p.flip = document.getElementById('flipX').checked;
    }

    function setActivePatch(name) {
      activePatch = name;
      document.getElementById('primaryPatchBtn').classList.toggle('ok', name === 'primary');
      document.getElementById('secondaryPatchBtn').classList.toggle('ok', name === 'secondary');
      if (name === 'secondary') patches.secondary.enabled = true;
      syncControlsFromPatch();
    }

    function updateLabels() {
      document.getElementById('brushValue').textContent = document.getElementById('brushSize').value;
      document.getElementById('sourceZoomValue').textContent = document.getElementById('sourceZoom').value + '%';
      document.getElementById('scaleValue').textContent = document.getElementById('scale').value + '%';
      document.getElementById('rotationValue').textContent = document.getElementById('rotation').value + '°';
      document.getElementById('featherValue').textContent = document.getElementById('feather').value;
      document.getElementById('contractValue').textContent = document.getElementById('contract').value;
      document.getElementById('opacityValue').textContent = document.getElementById('opacity').value + '%';
      document.getElementById('colorValue').textContent = document.getElementById('colorMatch').value + '%';
      document.getElementById('skinTintValue').textContent = document.getElementById('skinTint').value + '%';
      document.getElementById('protectValue').textContent = document.getElementById('protectDark').value + '%';
      document.getElementById('padValue').textContent = document.getElementById('pad').value;
      if (targetImage) {
        const x = Math.round(Number(document.getElementById('posX').value) / 100 * targetImage.naturalWidth);
        const y = Math.round(Number(document.getElementById('posY').value) / 100 * targetImage.naturalHeight);
        document.getElementById('xValue').textContent = x;
        document.getElementById('yValue').textContent = y;
        document.getElementById('posXNum').value = x;
        document.getElementById('posYNum').value = y;
      }
      document.getElementById('scaleNum').value = document.getElementById('scale').value;
      document.getElementById('rotationNum').value = document.getElementById('rotation').value;
    }

    function pushUndo() {
      if (!maskCanvas.width || !maskCanvas.height) return;
      undoStack.push(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
      if (undoStack.length > 20) undoStack.shift();
    }

    function getMaskBBox() {
      if (maskBBoxCache !== null) return maskBBoxCache;
      if (!maskCanvas.width || !maskCanvas.height) {
        maskBBoxCache = null;
        return null;
      }
      const data = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;
      let minX = maskCanvas.width, minY = maskCanvas.height, maxX = -1, maxY = -1;
      for (let y = 0; y < maskCanvas.height; y += 1) {
        for (let x = 0; x < maskCanvas.width; x += 1) {
          const i = (y * maskCanvas.width + x) * 4;
          if (data[i + 3] > 5 || data[i] > 5 || data[i + 1] > 5 || data[i + 2] > 5) {
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
          }
        }
      }
      if (maxX < minX || maxY < minY) {
        maskBBoxCache = null;
        return null;
      }
      const pad = 12;
      minX = Math.max(0, minX - pad);
      minY = Math.max(0, minY - pad);
      maxX = Math.min(maskCanvas.width - 1, maxX + pad);
      maxY = Math.min(maskCanvas.height - 1, maxY + pad);
      maskBBoxCache = { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
      return maskBBoxCache;
    }

    function livePatchBBox() {
      const bbox = getMaskBBox();
      if (!bbox) return null;
      const pad = Number(document.getElementById('pad').value);
      const grow = Math.max(12, pad + 12);
      const x = Math.max(0, bbox.x - grow);
      const y = Math.max(0, bbox.y - grow);
      const right = Math.min(maskCanvas.width, bbox.x + bbox.w + grow);
      const bottom = Math.min(maskCanvas.height, bbox.y + bbox.h + grow);
      return { x, y, w: Math.max(1, right - x), h: Math.max(1, bottom - y) };
    }

    function roundedRect(ctx, x, y, w, h, r) {
      const radius = Math.max(0, Math.min(r, w / 2, h / 2));
      ctx.beginPath();
      ctx.moveTo(x + radius, y);
      ctx.lineTo(x + w - radius, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
      ctx.lineTo(x + w, y + h - radius);
      ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
      ctx.lineTo(x + radius, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
      ctx.lineTo(x, y + radius);
      ctx.quadraticCurveTo(x, y, x + radius, y);
      ctx.closePath();
    }

    function percentile(values, pct) {
      if (!values.length) return 0;
      values.sort((a, b) => a - b);
      const idx = Math.min(values.length - 1, Math.max(0, Math.round((values.length - 1) * pct)));
      return values[idx];
    }

    function smoothStepNumber(edge0, edge1, value) {
      const denom = Math.max(edge1 - edge0, 1e-6);
      const t = Math.max(0, Math.min(1, (value - edge0) / denom));
      return t * t * (3 - 2 * t);
    }

    function makeInnerAlphaCanvas(bbox, feather, contract = 0) {
      const w = bbox.w;
      const h = bbox.h;
      const alphaCanvas = document.createElement('canvas');
      alphaCanvas.width = w;
      alphaCanvas.height = h;
      const alphaCtx = alphaCanvas.getContext('2d');
      const src = maskCtx.getImageData(bbox.x, bbox.y, w, h).data;
      const imageData = alphaCtx.createImageData(w, h);
      const out = imageData.data;
      const base = new Uint8ClampedArray(w * h);
      for (let i = 0, p = 0; i < src.length; i += 4, p += 1) {
        const m = Math.max(src[i], src[i + 1], src[i + 2], src[i + 3]);
        base[p] = m > 5 ? m : 0;
      }

      let alpha = base;
      if (feather > 0 || contract > 0) {
        const inf = 1e6;
        const diag = 1.4142;
        const dist = new Float32Array(w * h);
        for (let i = 0; i < dist.length; i += 1) dist[i] = base[i] > 5 ? inf : 0;
        for (let y = 0; y < h; y += 1) {
          for (let x = 0; x < w; x += 1) {
            const i = y * w + x;
            if (dist[i] === 0) continue;
            let d = dist[i];
            if (x > 0) d = Math.min(d, dist[i - 1] + 1);
            if (y > 0) d = Math.min(d, dist[i - w] + 1);
            if (x > 0 && y > 0) d = Math.min(d, dist[i - w - 1] + diag);
            if (x + 1 < w && y > 0) d = Math.min(d, dist[i - w + 1] + diag);
            dist[i] = d;
          }
        }
        for (let y = h - 1; y >= 0; y -= 1) {
          for (let x = w - 1; x >= 0; x -= 1) {
            const i = y * w + x;
            if (dist[i] === 0) continue;
            let d = dist[i];
            if (x + 1 < w) d = Math.min(d, dist[i + 1] + 1);
            if (y + 1 < h) d = Math.min(d, dist[i + w] + 1);
            if (x + 1 < w && y + 1 < h) d = Math.min(d, dist[i + w + 1] + diag);
            if (x > 0 && y + 1 < h) d = Math.min(d, dist[i + w - 1] + diag);
            dist[i] = d;
          }
        }
        const softened = new Uint8ClampedArray(w * h);
        const softWidth = Math.max(1, feather * 0.45);
        const shrink = Math.max(0, contract);
        for (let i = 0; i < softened.length; i += 1) {
          const innerDistance = dist[i] - shrink;
          if (base[i] <= 5 || innerDistance <= 0) {
            softened[i] = 0;
          } else {
            const inward = feather > 0 ? Math.max(0, Math.min(255, innerDistance / softWidth * 255)) : 255;
            softened[i] = Math.min(base[i], inward);
          }
        }
        alpha = softened;
      }

      const protect = Number(document.getElementById('protectDark').value) / 100;
      if (protect > 0 && sourceImage) {
        const patch = document.createElement('canvas');
        patch.width = w;
        patch.height = h;
        const patchCtx = patch.getContext('2d', { willReadFrequently: true });
        patchCtx.drawImage(sourceImage, bbox.x, bbox.y, w, h, 0, 0, w, h);
        const src = patchCtx.getImageData(0, 0, w, h).data;
        const selected = [];
        for (let i = 0, p = 0; i < src.length; i += 4, p += 1) {
          if (base[p] > 5) selected.push(src[i] * 0.299 + src[i + 1] * 0.587 + src[i + 2] * 0.114);
        }
        if (selected.length >= 20) {
          const p18 = percentile(selected.slice(), 0.18);
          const p88 = percentile(selected.slice(), 0.88);
          const contrast = Math.max(18, p88 - p18);
          const edge0 = Math.max(14, contrast * 0.18);
          const edge1 = Math.max(42, contrast * 0.72);
          const protectGain = 0.25 + 0.75 * Math.max(0, Math.min(1, protect));
          for (let i = 0, p = 0; i < src.length; i += 4, p += 1) {
            if (base[p] <= 5) continue;
            const gray = src[i] * 0.299 + src[i + 1] * 0.587 + src[i + 2] * 0.114;
            const hair = smoothStepNumber(edge0, edge1, p88 - gray);
            const hairAlpha = Math.max(0, Math.min(255, hair * base[p] * protectGain));
            alpha[p] = Math.max(alpha[p], hairAlpha);
          }
        }
      }

      for (let i = 0, p = 0; p < alpha.length; i += 4, p += 1) {
        out[i] = 255;
        out[i + 1] = 255;
        out[i + 2] = 255;
        out[i + 3] = alpha[p];
      }
      alphaCtx.putImageData(imageData, 0, 0);
      return alphaCanvas;
    }

    function redrawSource() {
      if (!sourceImage) return;
      const zoom = Number(document.getElementById('sourceZoom').value) / 100;
      const maxW = Math.max(340, window.innerWidth * 0.34);
      sourceDisplayScale = Math.min(maxW / sourceImage.naturalWidth, 1) * zoom;
      const w = Math.max(1, Math.round(sourceImage.naturalWidth * sourceDisplayScale));
      const h = Math.max(1, Math.round(sourceImage.naturalHeight * sourceDisplayScale));
      sourceCanvas.width = sourcePaintCanvas.width = w;
      sourceCanvas.height = sourcePaintCanvas.height = h;
      sourceCtx.clearRect(0, 0, w, h);
      sourceCtx.drawImage(sourceImage, 0, 0, w, h);
      sourcePaintCtx.clearRect(0, 0, w, h);
      if (maskCanvas.width) {
        sourcePaintCtx.save();
        sourcePaintCtx.globalAlpha = 0.44;
        sourcePaintCtx.drawImage(maskCanvas, 0, 0, w, h);
        sourcePaintCtx.restore();
      }
      updateLabels();
    }

    function drawFastPatchPreview() {
      if (!sourceImage || !targetImage) return;
      const bbox = livePatchBBox();
      if (!bbox) return;
      const opacity = Number(document.getElementById('opacity').value) / 100;
      const feather = Number(document.getElementById('feather').value);
      const contract = Number(document.getElementById('contract').value);
      const alphaCanvas = makeInnerAlphaCanvas(bbox, feather, contract);
      for (const [name, p] of Object.entries(patches)) {
        if (!p.enabled) continue;
        const x = p.xPct / 100 * targetCanvas.width;
        const y = p.yPct / 100 * targetCanvas.height;
        const scale = Math.max(0.05, p.scale / 100) * targetDisplayScale;
        targetCtx.save();
        targetCtx.globalAlpha = Math.max(0, Math.min(1, opacity));
        targetCtx.translate(x, y);
        targetCtx.rotate(p.rotation * Math.PI / 180);
        targetCtx.scale(p.flip ? -scale : scale, scale);

        const patch = document.createElement('canvas');
        patch.width = bbox.w;
        patch.height = bbox.h;
        const patchCtx = patch.getContext('2d');
        patchCtx.drawImage(sourceImage, bbox.x, bbox.y, bbox.w, bbox.h, 0, 0, bbox.w, bbox.h);

        patchCtx.globalCompositeOperation = 'destination-in';
        patchCtx.drawImage(alphaCanvas, 0, 0);
        targetCtx.drawImage(patch, -bbox.w / 2, -bbox.h / 2);
        targetCtx.restore();
      }
    }

    function redrawTarget(options = {}) {
      if (!targetImage) return;
      const maxW = Math.max(480, window.innerWidth * 0.42);
      targetDisplayScale = Math.min(maxW / targetImage.naturalWidth, 1);
      const w = Math.max(1, Math.round(targetImage.naturalWidth * targetDisplayScale));
      const h = Math.max(1, Math.round(targetImage.naturalHeight * targetDisplayScale));
      targetCanvas.width = targetHitCanvas.width = w;
      targetCanvas.height = targetHitCanvas.height = h;
      targetCtx.clearRect(0, 0, w, h);
      targetCtx.drawImage(targetImage, 0, 0, w, h);
      drawFastPatchPreview();
      drawTargetMarker();
      updateLabels();
    }

    function drawTargetMarker() {
      if (!targetImage) return;
      const w = targetCanvas.width;
      const h = targetCanvas.height;
      targetHitCtx.clearRect(0, 0, w, h);
      if (!targetDragging) return;
      const p = activePatchData();
      if (!p.enabled) return;
      const x = p.xPct / 100 * w;
      const y = p.yPct / 100 * h;
      targetHitCtx.save();
      targetHitCtx.strokeStyle = '#4da3ff';
      targetHitCtx.globalAlpha = 0.45;
      targetHitCtx.lineWidth = 1.5;
      targetHitCtx.beginPath();
      targetHitCtx.arc(x, y, 5, 0, Math.PI * 2);
      targetHitCtx.stroke();
      targetHitCtx.restore();
    }

    function loadImageData(dataUrl, name, kind) {
      const img = new Image();
      img.onload = () => {
        if (kind === 'source') {
          sourceImage = img;
          sourceData = dataUrl;
          sourceName = name || 'source';
          maskCanvas.width = img.naturalWidth;
          maskCanvas.height = img.naturalHeight;
          maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
          maskBBoxCache = null;
          invalidatePreview();
          undoStack = [];
          sourceInfo.textContent = `${img.naturalWidth} x ${img.naturalHeight}`;
          redrawSource();
          redrawTarget({ fast: true });
        } else {
          targetImage = img;
          targetData = dataUrl;
          targetPreviewImage = null;
          targetOriginalCanvas.width = img.naturalWidth;
          targetOriginalCanvas.height = img.naturalHeight;
          targetOriginalCtx.clearRect(0, 0, targetOriginalCanvas.width, targetOriginalCanvas.height);
          targetOriginalCtx.drawImage(img, 0, 0);
          targetName = name || 'target';
          targetInfo.textContent = `${img.naturalWidth} x ${img.naturalHeight}`;
          patches.primary.xPct = 50;
          patches.primary.yPct = 35;
          patches.secondary.xPct = 58;
          patches.secondary.yPct = 35;
          setActivePatch('primary');
          redrawTarget({ fast: true });
        }
        hasPreview = false;
        resultImg.removeAttribute('src');
        setStatus('图片已载入。');
      };
      img.src = dataUrl;
    }

    async function loadPath(path, kind) {
      if (!path) return;
      setStatus('正在打开图片...');
      const resp = await fetch('/api/load_path', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({path})
      });
      const data = await resp.json();
      if (!resp.ok) {
        setStatus(data.detail || '打开失败');
        return;
      }
      loadImageData(data.data_url, data.name, kind);
    }

    document.getElementById('loadSourcePath').onclick = () => loadPath(document.getElementById('sourcePath').value.trim(), 'source');
    document.getElementById('loadTargetPath').onclick = () => loadPath(document.getElementById('targetPath').value.trim(), 'target');
    document.getElementById('fitSource').onclick = () => { setVal('sourceZoom', 100); redrawSource(); };
    document.getElementById('fitTarget').onclick = () => redrawTarget();

    function readFileInput(input, kind) {
      const file = input.files && input.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => loadImageData(reader.result, file.name, kind);
      reader.readAsDataURL(file);
    }
    document.getElementById('sourceFile').addEventListener('change', ev => readFileInput(ev.target, 'source'));
    document.getElementById('targetFile').addEventListener('change', ev => readFileInput(ev.target, 'target'));

    function sourcePoint(ev) {
      const rect = sourcePaintCanvas.getBoundingClientRect();
      return {
        x: (ev.clientX - rect.left) * maskCanvas.width / rect.width,
        y: (ev.clientY - rect.top) * maskCanvas.height / rect.height
      };
    }

    function drawSourceMask(point) {
      const size = Number(document.getElementById('brushSize').value);
      maskCtx.save();
      maskCtx.lineCap = 'round';
      maskCtx.lineJoin = 'round';
      maskCtx.lineWidth = size;
      if (sourceTool === 'eraser') {
        maskCtx.globalCompositeOperation = 'destination-out';
      } else {
        maskCtx.globalCompositeOperation = 'source-over';
      }
      maskCtx.strokeStyle = 'rgba(0,160,255,1)';
      maskCtx.beginPath();
      if (lastPoint) maskCtx.moveTo(lastPoint.x, lastPoint.y);
      else maskCtx.moveTo(point.x, point.y);
      maskCtx.lineTo(point.x, point.y);
      maskCtx.stroke();
      maskCtx.restore();
      lastPoint = point;
      maskBBoxCache = null;
      invalidatePreview();
      redrawSource();
    }

    sourcePaintCanvas.addEventListener('pointerdown', ev => {
      if (!sourceImage) return;
      ev.preventDefault();
      pushUndo();
      sourceDrawing = true;
      lastPoint = sourcePoint(ev);
      drawSourceMask(lastPoint);
      sourcePaintCanvas.setPointerCapture(ev.pointerId);
    });
    sourcePaintCanvas.addEventListener('pointermove', ev => {
      if (!sourceDrawing) return;
      ev.preventDefault();
      drawSourceMask(sourcePoint(ev));
    });
    sourcePaintCanvas.addEventListener('pointerup', ev => {
      sourceDrawing = false;
      lastPoint = null;
      sourcePaintCanvas.releasePointerCapture(ev.pointerId);
      redrawTarget({ fast: true });
      schedulePreview();
    });
    sourcePaintCanvas.addEventListener('pointercancel', () => {
      sourceDrawing = false;
      lastPoint = null;
    });

    function targetPointPct(ev) {
      const rect = targetHitCanvas.getBoundingClientRect();
      const xPct = (ev.clientX - rect.left) / rect.width * 100;
      const yPct = (ev.clientY - rect.top) / rect.height * 100;
      return {
        xPct,
        yPct,
        xCanvas: Math.round((ev.clientX - rect.left) / rect.width * targetCanvas.width),
        yCanvas: Math.round((ev.clientY - rect.top) / rect.height * targetCanvas.height),
        xTarget: Math.round(xPct / 100 * (targetImage ? targetImage.naturalWidth : 0)),
        yTarget: Math.round(yPct / 100 * (targetImage ? targetImage.naturalHeight : 0))
      };
    }

    function moveActivePatchTo(point) {
      const p = activePatchData();
      p.xPct = Math.max(0, Math.min(100, point.xPct - targetDragOffset.xPct));
      p.yPct = Math.max(0, Math.min(100, point.yPct - targetDragOffset.yPct));
      syncControlsFromPatch();
    }

    function patchHitRadius(p) {
      const bbox = getMaskBBox();
      if (!bbox) return 34;
      const displayMax = Math.max(bbox.w, bbox.h) * (p.scale / 100) * targetDisplayScale;
      return Math.max(28, Math.min(160, displayMax * 0.55));
    }

    function findPatchAt(point) {
      let best = null;
      let bestDist = Infinity;
      for (const [name, p] of Object.entries(patches)) {
        if (!p.enabled) continue;
        const x = p.xPct / 100 * targetCanvas.width;
        const y = p.yPct / 100 * targetCanvas.height;
        const dx = point.xCanvas - x;
        const dy = point.yCanvas - y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist <= patchHitRadius(p) && dist < bestDist) {
          best = name;
          bestDist = dist;
        }
      }
      return best;
    }

    function sampleTargetSkin(point) {
      if (!targetImage) return;
      const x = Math.max(0, Math.min(targetOriginalCanvas.width - 1, Math.round(point.xPct / 100 * targetOriginalCanvas.width)));
      const y = Math.max(0, Math.min(targetOriginalCanvas.height - 1, Math.round(point.yPct / 100 * targetOriginalCanvas.height)));
      const radius = 8;
      const sx = Math.max(0, x - radius);
      const sy = Math.max(0, y - radius);
      const sw = Math.min(targetOriginalCanvas.width - sx, radius * 2 + 1);
      const sh = Math.min(targetOriginalCanvas.height - sy, radius * 2 + 1);
      const data = targetOriginalCtx.getImageData(sx, sy, sw, sh).data;
      const pixels = [];
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        if (lum > 30 && lum < 245) pixels.push([r, g, b, lum]);
      }
      pixels.sort((a, b) => a[3] - b[3]);
      const keep = pixels.length ? pixels.slice(
        Math.floor(pixels.length * 0.15),
        Math.max(Math.floor(pixels.length * 0.85), Math.floor(pixels.length * 0.15) + 1)
      ) : [[data[0], data[1], data[2], 0]];
      const avg = keep.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]], [0, 0, 0])
        .map(v => Math.round(v / keep.length));
      skinSample = [avg[0], avg[1], avg[2]];
      const css = `rgb(${avg[0]}, ${avg[1]}, ${avg[2]})`;
      document.getElementById('skinSwatch').style.background = css;
      document.getElementById('skinText').textContent = `已吸取 ${avg[0]}, ${avg[1]}, ${avg[2]}`;
      schedulePreview();
    }

    targetHitCanvas.addEventListener('pointerdown', ev => {
      if (!targetImage) return;
      ev.preventDefault();
      const point = targetPointPct(ev);
      if (targetTool === 'eyedropper') {
        sampleTargetSkin(point);
        return;
      }
      const hitPatch = findPatchAt(point);
      if (hitPatch) {
        setActivePatch(hitPatch);
        const p = activePatchData();
        targetDragOffset = { xPct: point.xPct - p.xPct, yPct: point.yPct - p.yPct };
      } else {
        targetDragOffset = { xPct: 0, yPct: 0 };
      }
      targetDragging = true;
      targetHitCanvas.classList.add('dragging');
      moveActivePatchTo(point);
      redrawTarget({ fast: true });
      targetHitCanvas.setPointerCapture(ev.pointerId);
    });
    targetHitCanvas.addEventListener('pointermove', ev => {
      if (!targetDragging || targetTool !== 'move') return;
      ev.preventDefault();
      moveActivePatchTo(targetPointPct(ev));
      redrawTarget({ fast: true });
    });
    targetHitCanvas.addEventListener('pointerup', ev => {
      if (targetDragging) schedulePreview();
      targetDragging = false;
      targetDragOffset = { xPct: 0, yPct: 0 };
      targetHitCanvas.classList.remove('dragging');
      try { targetHitCanvas.releasePointerCapture(ev.pointerId); } catch (_) {}
    });
    targetHitCanvas.addEventListener('pointercancel', () => {
      targetDragging = false;
      targetDragOffset = { xPct: 0, yPct: 0 };
      targetHitCanvas.classList.remove('dragging');
    });

    function setTargetTool(tool) {
      targetTool = tool;
      targetHitCanvas.style.cursor = tool === 'move' ? 'grab' : 'crosshair';
      document.getElementById('moveTargetBtn').classList.toggle('ok', tool === 'move');
      document.getElementById('eyedropperBtn').classList.toggle('ok', tool === 'eyedropper');
    }
    document.getElementById('moveTargetBtn').onclick = () => setTargetTool('move');
    document.getElementById('eyedropperBtn').onclick = () => setTargetTool('eyedropper');
    document.getElementById('primaryPatchBtn').onclick = () => setActivePatch('primary');
    document.getElementById('secondaryPatchBtn').onclick = () => setActivePatch('secondary');
    document.getElementById('copyPatchBtn').onclick = () => {
      syncPatchFromControls();
      const src = activePatchData();
      patches.secondary = {
        enabled: true,
        xPct: Math.max(0, Math.min(100, 100 - src.xPct)),
        yPct: src.yPct,
        scale: src.scale,
        rotation: -src.rotation,
        flip: !src.flip
      };
      setActivePatch('secondary');
      redrawTarget({ fast: true });
      schedulePreview();
    };
    document.getElementById('removeSecondBtn').onclick = () => {
      patches.secondary.enabled = false;
      setActivePatch('primary');
      redrawTarget({ fast: true });
      schedulePreview();
    };

    document.getElementById('brushBtn').onclick = () => {
      sourceTool = 'brush';
      document.getElementById('brushBtn').classList.add('ok');
      document.getElementById('eraserBtn').classList.remove('ok');
    };
    document.getElementById('eraserBtn').onclick = () => {
      sourceTool = 'eraser';
      document.getElementById('eraserBtn').classList.add('ok');
      document.getElementById('brushBtn').classList.remove('ok');
    };
    document.getElementById('undoBtn').onclick = () => {
      const item = undoStack.pop();
      if (item) {
        maskCtx.putImageData(item, 0, 0);
        maskBBoxCache = null;
        invalidatePreview();
        redrawSource();
        redrawTarget({ fast: true });
        schedulePreview();
      }
    };
    document.getElementById('clearMaskBtn').onclick = () => {
      if (!maskCanvas.width) return;
      pushUndo();
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      maskBBoxCache = null;
      invalidatePreview();
      redrawSource();
      redrawTarget({ fast: true });
      resultImg.removeAttribute('src');
    };

    function nudge(dx, dy) {
      setVal('posX', Number(document.getElementById('posX').value) + dx);
      setVal('posY', Number(document.getElementById('posY').value) + dy);
      syncPatchFromControls();
      redrawTarget({ fast: true });
      schedulePreview();
    }
    document.getElementById('leftBtn').onclick = () => nudge(-1, 0);
    document.getElementById('rightBtn').onclick = () => nudge(1, 0);
    document.getElementById('upBtn').onclick = () => nudge(0, -1);
    document.getElementById('downBtn').onclick = () => nudge(0, 1);
    document.getElementById('smallBtn').onclick = () => {
      setVal('scale', Number(document.getElementById('scale').value) - 5);
      syncPatchFromControls();
      redrawTarget({ fast: true });
      schedulePreview();
    };
    document.getElementById('bigBtn').onclick = () => {
      setVal('scale', Number(document.getElementById('scale').value) + 5);
      syncPatchFromControls();
      redrawTarget({ fast: true });
      schedulePreview();
    };

    for (const id of ['brushSize', 'sourceZoom']) {
      document.getElementById(id).addEventListener('input', () => {
        updateLabels();
        if (id === 'sourceZoom') redrawSource();
      });
    }
    for (const id of ['posX', 'posY', 'scale', 'rotation']) {
      document.getElementById(id).addEventListener('input', () => {
        syncPatchFromControls();
        updateLabels();
        redrawTarget({ fast: true });
        schedulePreview();
      });
      document.getElementById(id).addEventListener('change', () => {
        syncPatchFromControls();
        updateLabels();
        redrawTarget({ fast: true });
        schedulePreview();
      });
    }
    document.getElementById('flipX').addEventListener('change', () => {
      syncPatchFromControls();
      redrawTarget({ fast: true });
      schedulePreview();
    });
    document.getElementById('posXNum').addEventListener('change', () => {
      if (!targetImage) return;
      setVal('posX', Number(document.getElementById('posXNum').value) / targetImage.naturalWidth * 100);
      syncPatchFromControls();
      redrawTarget({ fast: true });
      schedulePreview();
    });
    document.getElementById('posYNum').addEventListener('change', () => {
      if (!targetImage) return;
      setVal('posY', Number(document.getElementById('posYNum').value) / targetImage.naturalHeight * 100);
      syncPatchFromControls();
      redrawTarget({ fast: true });
      schedulePreview();
    });
    document.getElementById('scaleNum').addEventListener('change', () => {
      setVal('scale', Number(document.getElementById('scaleNum').value));
      syncPatchFromControls();
      redrawTarget({ fast: true });
      schedulePreview();
    });
    document.getElementById('rotationNum').addEventListener('change', () => {
      setVal('rotation', Number(document.getElementById('rotationNum').value));
      syncPatchFromControls();
      redrawTarget({ fast: true });
      schedulePreview();
    });
    for (const id of ['blendMode', 'feather', 'contract', 'opacity', 'colorMatch', 'skinTint', 'protectDark', 'pad']) {
      const onParamChange = () => {
        updateLabels();
        redrawTarget({ fast: true });
        schedulePreview();
      };
      document.getElementById(id).addEventListener('input', onParamChange);
      document.getElementById(id).addEventListener('change', onParamChange);
    }

    function payload(save) {
      syncPatchFromControls();
      const placements = [];
      for (const [name, p] of Object.entries(patches)) {
        if (!p.enabled) continue;
        placements.push({
          name,
          x: targetImage ? p.xPct / 100 * targetImage.naturalWidth : 0,
          y: targetImage ? p.yPct / 100 * targetImage.naturalHeight : 0,
          scale: p.scale / 100,
          rotation: p.rotation,
          flip_x: !!p.flip
        });
      }
      return {
        source_data: sourceData,
        source_mask_data: maskCanvas.toDataURL('image/png'),
        target_data: targetData,
        source_name: sourceName,
        target_name: targetName,
        patches: placements,
        blend_mode: document.getElementById('blendMode').value,
        feather: Number(document.getElementById('feather').value),
        contract: Number(document.getElementById('contract').value),
        opacity: Number(document.getElementById('opacity').value) / 100,
        color_match: Number(document.getElementById('colorMatch').value) / 100,
        skin_tint: Number(document.getElementById('skinTint').value) / 100,
        old_brow_cover: 0,
        skin_sample: skinSample,
        protect_dark: Number(document.getElementById('protectDark').value) / 100,
        pad: Number(document.getElementById('pad').value),
        output_dir: document.getElementById('outputDir').value.trim(),
        output_name: document.getElementById('outputName').value.trim(),
        save
      };
    }

    function patchPayload(save) {
      return {
        source_data: sourceData,
        source_mask_data: maskCanvas.toDataURL('image/png'),
        source_name: sourceName,
        patch_mode: document.getElementById('patchMode').value,
        feather: Number(document.getElementById('feather').value),
        contract: Number(document.getElementById('contract').value),
        protect_dark: Number(document.getElementById('protectDark').value) / 100,
        pad: Number(document.getElementById('pad').value),
        output_dir: document.getElementById('patchOutputDir').value.trim(),
        output_name: document.getElementById('patchOutputName').value.trim(),
        save
      };
    }

    function tunedPatchPayload(save) {
      syncPatchFromControls();
      const p = activePatchData();
      return {
        source_data: sourceData,
        source_mask_data: maskCanvas.toDataURL('image/png'),
        target_data: targetData,
        source_name: sourceName,
        target_name: targetName,
        patches: [{
          name: activePatch,
          x: targetImage ? p.xPct / 100 * targetImage.naturalWidth : 0,
          y: targetImage ? p.yPct / 100 * targetImage.naturalHeight : 0,
          scale: p.scale / 100,
          rotation: p.rotation,
          flip_x: !!p.flip
        }],
        blend_mode: document.getElementById('blendMode').value,
        feather: Number(document.getElementById('feather').value),
        contract: Number(document.getElementById('contract').value),
        opacity: Number(document.getElementById('opacity').value) / 100,
        color_match: Number(document.getElementById('colorMatch').value) / 100,
        skin_tint: Number(document.getElementById('skinTint').value) / 100,
        old_brow_cover: 0,
        skin_sample: skinSample,
        protect_dark: Number(document.getElementById('protectDark').value) / 100,
        pad: Number(document.getElementById('pad').value),
        output_dir: document.getElementById('patchOutputDir').value.trim(),
        output_name: document.getElementById('patchOutputName').value.trim(),
        save
      };
    }

    async function process(save) {
      if (!sourceImage || !targetImage) {
        setStatus('来源图和目标图都要先载入。');
        return;
      }
      if (!getMaskBBox()) {
        setStatus('来源选区为空，请先在来源图上涂出眉毛区域。');
        return;
      }
      if (processBusy && !save) {
        pendingPreview = true;
        return;
      }
      processBusy = true;
      setStatus(save ? '正在保存当前合成...' : '正在更新预览...');
      document.getElementById('previewBtn').disabled = true;
      document.getElementById('saveBtn').disabled = true;
      document.getElementById('savePatchBtn').disabled = true;
      document.getElementById('saveTunedPatchBtn').disabled = true;
      try {
        const resp = await fetch('/api/blend', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload(save))
        });
        const data = await resp.json();
        if (!resp.ok) {
          setStatus(data.detail || '处理失败');
          return;
        }
        resultImg.src = data.data_url;
        resultImg.style.display = 'block';
        hasPreview = true;
        const saved = data.path ? `<br><strong>保存：</strong>${data.path}` : '<br><strong>状态：</strong>仅预览，未保存';
        setStatus(`<strong>输出：</strong>${data.width} x ${data.height}${saved}`);
      } finally {
        processBusy = false;
        document.getElementById('previewBtn').disabled = false;
        document.getElementById('saveBtn').disabled = false;
        document.getElementById('savePatchBtn').disabled = false;
        document.getElementById('saveTunedPatchBtn').disabled = false;
        if (pendingPreview && !save) {
          pendingPreview = false;
          schedulePreview();
        }
      }
    }

    async function exportPatch(save) {
      if (!sourceImage) {
        setStatus('请先载入来源图。');
        return;
      }
      if (!getMaskBBox()) {
        setStatus('来源选区为空，请先在来源图上涂出眉毛区域。');
        return;
      }
      setStatus('正在保存透明眉毛贴片...');
      document.getElementById('savePatchBtn').disabled = true;
      document.getElementById('saveTunedPatchBtn').disabled = true;
      try {
        const resp = await fetch('/api/export_patch', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(patchPayload(save))
        });
        const data = await resp.json();
        if (!resp.ok) {
          setStatus(data.detail || '贴片导出失败');
          return;
        }
        resultImg.src = data.data_url;
        resultImg.style.display = 'block';
        const saved = data.path ? `<br><strong>贴片：</strong>${data.path}` : '<br><strong>状态：</strong>仅预览，未保存';
        setStatus(`<strong>透明贴片：</strong>${data.width} x ${data.height}${saved}`);
      } finally {
        document.getElementById('savePatchBtn').disabled = false;
        document.getElementById('saveTunedPatchBtn').disabled = false;
      }
    }

    async function exportTunedPatch(save) {
      if (!sourceImage || !targetImage) {
        setStatus('来源图和目标图都要先载入，才能保存当前调好贴片。');
        return;
      }
      if (!getMaskBBox()) {
        setStatus('来源选区为空，请先在来源图上涂出眉毛区域。');
        return;
      }
      setStatus('正在保存当前调好贴片...');
      document.getElementById('savePatchBtn').disabled = true;
      document.getElementById('saveTunedPatchBtn').disabled = true;
      try {
        const resp = await fetch('/api/export_tuned_patch', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(tunedPatchPayload(save))
        });
        const data = await resp.json();
        if (!resp.ok) {
          setStatus(data.detail || '调好贴片导出失败');
          return;
        }
        resultImg.src = data.data_url;
        resultImg.style.display = 'block';
        const saved = data.path ? `<br><strong>调好贴片：</strong>${data.path}` : '<br><strong>状态：</strong>仅预览，未保存';
        setStatus(`<strong>调好贴片：</strong>${data.width} x ${data.height}${saved}`);
      } finally {
        document.getElementById('savePatchBtn').disabled = false;
        document.getElementById('saveTunedPatchBtn').disabled = false;
      }
    }

    function schedulePreview() {
      if (!sourceImage || !targetImage || !getMaskBBox()) return;
      clearTimeout(previewTimer);
      previewTimer = setTimeout(() => process(false), 350);
    }

    document.getElementById('previewBtn').onclick = () => process(false);
    document.getElementById('saveBtn').onclick = () => process(true);
    document.getElementById('savePatchBtn').onclick = () => exportPatch(true);
    document.getElementById('saveTunedPatchBtn').onclick = () => exportTunedPatch(true);
    window.addEventListener('resize', () => {
      redrawSource();
      redrawTarget();
    });
    updateLabels();
  </script>
</body>
</html>
"""


class PathRequest(BaseModel):
    path: str


class PatchPlacement(BaseModel):
    name: str = "patch"
    x: float = 0
    y: float = 0
    scale: float = 1.0
    rotation: float = 0.0
    flip_x: bool = False


class BlendRequest(BaseModel):
    source_data: str
    source_mask_data: str
    target_data: str
    source_name: str = "source"
    target_name: str = "target"
    patches: list[PatchPlacement] = Field(default_factory=list)
    x: float = 0
    y: float = 0
    scale: float = 1.0
    rotation: float = 0.0
    flip_x: bool = False
    blend_mode: str = "alpha"
    feather: int = 18
    contract: int = 5
    opacity: float = 1.0
    color_match: float = 0.65
    skin_tint: float = 0.70
    old_brow_cover: float = 0.0
    skin_sample: Optional[list[float]] = None
    protect_dark: float = 0.70
    pad: int = 0
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    output_name: str = ""
    save: bool = False


class ExportPatchRequest(BaseModel):
    source_data: str
    source_mask_data: str
    source_name: str = "source"
    patch_mode: str = "hair_only"
    feather: int = 18
    contract: int = 5
    protect_dark: float = 0.70
    pad: int = 0
    output_dir: str = str(DEFAULT_PATCH_DIR)
    output_name: str = ""
    save: bool = True


app = FastAPI(title="PS-style eyebrow patch blending UI")


def decode_data_url(value: str) -> Image.Image:
    match = re.match(r"^data:image/[^;]+;base64,(.*)$", value, re.S)
    if not match:
        raise HTTPException(status_code=400, detail="图片数据格式不对")
    try:
        raw = base64.b64decode(match.group(1))
        return Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"图片解码失败: {exc}") from exc


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def data_url_from_image(path: Path):
    try:
        image = Image.open(path).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法读取图片: {exc}") from exc
    return {
        "name": path.name,
        "width": image.width,
        "height": image.height,
        "data_url": image_to_data_url(image),
    }


def normalize_mask(mask_image: Image.Image, size):
    mask = mask_image.convert("RGBA").resize(size, Image.Resampling.BILINEAR)
    arr = np.array(mask)
    alpha = arr[:, :, 3].astype(np.float32)
    color = arr[:, :, :3].max(axis=2).astype(np.float32)
    mask = np.maximum(alpha, color)
    mask[mask < 5] = 0
    return np.clip(mask, 0, 255).astype(np.uint8)


def inner_feather_mask(mask: np.ndarray, feather: int):
    if feather <= 0:
        return mask.copy()
    hard = (mask > 5).astype(np.uint8)
    if np.count_nonzero(hard) == 0:
        return mask.copy()

    distance = cv2.distanceTransform(hard, cv2.DIST_L2, 3)
    soft_width = max(1.0, float(feather) * 0.45)
    inward_alpha = np.clip(distance / soft_width, 0.0, 1.0)
    limited = np.minimum(mask.astype(np.float32) / 255.0, inward_alpha)
    return np.clip(limited * 255.0, 0, 255).astype(np.uint8)


def adjust_mask(mask: np.ndarray, contract: int, feather: int):
    result = mask.copy()
    if contract > 0:
        ksize = max(1, int(contract) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        result = cv2.erode(result, kernel, iterations=1)
    if feather > 0:
        result = inner_feather_mask(result, feather)
    return np.clip(result, 0, 255).astype(np.uint8)


def mask_bbox_and_fill_ratio(mask: np.ndarray):
    ys, xs = np.where(mask > 5)
    if not len(xs):
        return None, 0.0
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    area = max(1, (x2 - x1) * (y2 - y1))
    return (x1, y1, x2, y2), float(len(xs)) / float(area)


def make_patch_region_mask(mask: np.ndarray, contract: int = 0):
    binary = (mask > 5).astype(np.uint8) * 255
    bbox, fill_ratio = mask_bbox_and_fill_ratio(binary)
    if bbox is None:
        raise HTTPException(status_code=400, detail="来源选区为空，请先涂眉毛区域")

    # The PS reference video copies a whole skin patch, not isolated dark strokes.
    # If the user painted sparse strokes over the brow, expand/fill them into a
    # small soft patch so feathering does not turn eyebrow pixels into a black haze.
    region = binary
    if fill_ratio < 0.42:
        grow = 16
        close_size = max(5, min(45, grow // 2 * 2 + 1))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(region)
        min_area = max(12.0, mask.shape[0] * mask.shape[1] * 0.00001)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            hull = cv2.convexHull(contour)
            cv2.drawContours(filled, [hull], -1, 255, thickness=-1)
        if np.count_nonzero(filled) > 0:
            region = filled
        grow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1))
        region = cv2.dilate(region, grow_kernel, iterations=1)

    if contract > 0:
        ksize = max(1, int(contract) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        region = cv2.erode(region, kernel, iterations=1)
    return region


def make_soft_patch_mask(mask: np.ndarray, contract: int, feather: int):
    region = make_patch_region_mask(mask, contract)
    return adjust_mask(region, 0, feather)


def protect_brow_line_alpha(
    source_rgba: np.ndarray,
    region_mask: np.ndarray,
    soft_mask: np.ndarray,
    protect_dark: float,
):
    protect_dark = max(0.0, min(float(protect_dark), 1.0))
    if protect_dark <= 0:
        return soft_mask
    try:
        line_alpha = hair_line_alpha(source_rgba, region_mask)
    except HTTPException:
        return soft_mask
    protect_gain = 0.25 + 0.75 * protect_dark
    protected = line_alpha.astype(np.float32) * protect_gain
    merged = np.maximum(soft_mask.astype(np.float32), protected)
    merged = np.minimum(merged, region_mask.astype(np.float32))
    return np.clip(merged, 0, 255).astype(np.uint8)


def make_protected_soft_patch_mask(
    source_rgba: np.ndarray,
    mask: np.ndarray,
    contract: int,
    feather: int,
    protect_dark: float,
):
    region = make_patch_region_mask(mask, contract)
    soft = adjust_mask(region, 0, feather)
    return protect_brow_line_alpha(source_rgba, region, soft, protect_dark)


def crop_patch(source_rgba: np.ndarray, mask: np.ndarray, pad=18):
    ys, xs = np.where(mask > 2)
    if not len(xs):
        raise HTTPException(status_code=400, detail="来源选区为空，请先涂眉毛区域")
    h, w = mask.shape
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w, int(xs.max()) + pad + 1)
    y2 = min(h, int(ys.max()) + pad + 1)
    patch = source_rgba[y1:y2, x1:x2].copy()
    alpha = mask[y1:y2, x1:x2].copy()
    patch[:, :, 3] = alpha
    patch[:, :, :3][alpha == 0] = 0
    return patch


def crop_rgba_by_alpha(rgba: np.ndarray, alpha: np.ndarray, pad=18):
    ys, xs = np.where(alpha > 2)
    if not len(xs):
        raise HTTPException(status_code=400, detail="来源选区里没有提取到眉毛线条")
    h, w = alpha.shape
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w, int(xs.max()) + pad + 1)
    y2 = min(h, int(ys.max()) + pad + 1)
    patch = rgba[y1:y2, x1:x2].copy()
    patch_alpha = alpha[y1:y2, x1:x2].copy()
    patch[:, :, 3] = patch_alpha
    patch[:, :, :3][patch_alpha == 0] = 0
    return patch


def decontaminate_hair_patch(source_rgba: np.ndarray, alpha: np.ndarray):
    rgb = source_rgba[:, :, :3].astype(np.float32)
    gray = cv2.cvtColor(source_rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    strong = alpha > 150
    if np.count_nonzero(strong) < 20:
        strong = alpha > 70
    if np.count_nonzero(strong) >= 20:
        dark_cutoff = np.percentile(gray[strong], 55)
        hair_mask = strong & (gray <= dark_cutoff)
        if np.count_nonzero(hair_mask) < 10:
            hair_mask = strong
        hair_color = np.median(rgb[hair_mask], axis=0)
    else:
        hair_color = np.array([70.0, 50.0, 40.0], dtype=np.float32)

    a = np.clip(alpha.astype(np.float32) / 255.0, 0.0, 1.0)
    preserve = smoothstep(0.68, 0.98, a)[..., None]
    out = hair_color.reshape(1, 1, 3) * (1.0 - preserve) + rgb * preserve
    out_rgba = source_rgba.copy()
    out_rgba[:, :, :3] = np.clip(out, 0, 255).astype(np.uint8)
    out_rgba[:, :, :3][alpha == 0] = 0
    out_rgba[:, :, 3] = alpha
    return out_rgba


def hair_line_alpha(source_rgba: np.ndarray, region_mask: np.ndarray):
    rgb = source_rgba[:, :, :3].astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mask_float = np.clip(region_mask.astype(np.float32) / 255.0, 0.0, 1.0)
    inside = mask_float > 0.03
    if np.count_nonzero(inside) < 20:
        raise HTTPException(status_code=400, detail="来源选区太小，请多涂一点眉毛区域")

    selected = gray[inside]
    # Use the bright side of the selected region as local skin/paper reference,
    # then keep pixels that are darker than that reference. This preserves fine
    # hair tips without bringing the whole skin patch into the transparent PNG.
    reference = float(np.percentile(selected, 88))
    dark_delta = reference - gray
    contrast = max(18.0, float(np.percentile(selected, 88) - np.percentile(selected, 18)))
    edge0 = max(14.0, contrast * 0.18)
    edge1 = max(42.0, contrast * 0.72)
    alpha = smoothstep(edge0, edge1, dark_delta)
    alpha = np.power(alpha, 1.15)
    alpha = alpha * mask_float

    # Avoid exporting dark unrelated edges if the user selected a wide area:
    # softly prefer thin/dark strokes and let the manual mask remain the hard boundary.
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (3, 3), 0)
    return np.clip(alpha * 255.0, 0, 255).astype(np.uint8)


def transform_patch(patch: np.ndarray, scale: float, rotation: float, flip_x: bool = False):
    pil = Image.fromarray(patch, "RGBA")
    if flip_x:
        pil = ImageOps.mirror(pil)
    scale = max(0.05, min(float(scale), 5.0))
    new_size = (
        max(1, int(round(pil.width * scale))),
        max(1, int(round(pil.height * scale))),
    )
    pil = pil.resize(new_size, Image.Resampling.LANCZOS)
    pil = pil.rotate(float(rotation), expand=True, resample=Image.Resampling.BICUBIC)
    return np.array(pil)


def patch_canvas(target_shape, patch: np.ndarray, center_x: float, center_y: float):
    th, tw = target_shape[:2]
    canvas = np.zeros((th, tw, 4), dtype=np.uint8)
    ph, pw = patch.shape[:2]
    x1 = int(round(center_x - pw / 2))
    y1 = int(round(center_y - ph / 2))
    x2 = x1 + pw
    y2 = y1 + ph

    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = pw - max(0, x2 - tw)
    sy2 = ph - max(0, y2 - th)
    dx1 = max(0, x1)
    dy1 = max(0, y1)
    dx2 = dx1 + max(0, sx2 - sx1)
    dy2 = dy1 + max(0, sy2 - sy1)
    if dx2 <= dx1 or dy2 <= dy1:
        raise HTTPException(status_code=400, detail="贴片完全在目标图外")
    canvas[dy1:dy2, dx1:dx2] = patch[sy1:sy2, sx1:sx2]
    return canvas


def smoothstep(edge0: float, edge1: float, value):
    denom = max(float(edge1) - float(edge0), 1e-6)
    t = np.clip((value.astype(np.float32) - float(edge0)) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def estimate_skin_fill(target_rgb, alpha_mask, skin_weight, skin_sample=None):
    if skin_sample:
        return np.array(skin_sample[:3], dtype=np.float32)

    target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
    candidate_mask = alpha_mask & (skin_weight > 0.35)
    if np.count_nonzero(candidate_mask) < 20:
        candidate_mask = alpha_mask
    values = target_gray[candidate_mask]
    if values.size < 20:
        return np.array([210.0, 180.0, 160.0], dtype=np.float32)
    cutoff = np.percentile(values, 55)
    bright_mask = candidate_mask & (target_gray >= cutoff)
    if np.count_nonzero(bright_mask) < 20:
        bright_mask = candidate_mask
    return target_rgb[bright_mask].astype(np.float32).mean(axis=0)


def color_match_patch(
    patch_rgb,
    target_rgb,
    alpha,
    strength,
    protect_dark,
    skin_sample=None,
    skin_tint=0.0,
    old_brow_cover=0.0,
):
    strength = max(0.0, min(float(strength), 1.0))
    skin_tint = max(0.0, min(float(skin_tint), 1.0))
    old_brow_cover = 0.0
    if strength <= 0 and skin_tint <= 0:
        return patch_rgb
    alpha_mask = alpha > 20
    if np.count_nonzero(alpha_mask) < 20:
        return patch_rgb

    patch_lab = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    selected = gray[alpha_mask]
    if selected.size < 20:
        return patch_rgb
    protect_dark = max(0.0, min(float(protect_dark), 1.0))
    low = np.percentile(selected, 30)
    mid = np.percentile(selected, 50)
    high = np.percentile(selected, 82)
    skin_start = mid + (high - mid) * protect_dark * 0.25
    skin_weight = smoothstep(skin_start, high, gray)
    skin_mask = alpha_mask & (skin_weight > 0.35)
    if np.count_nonzero(skin_mask) < 20:
        skin_mask = alpha_mask

    src_mean = patch_lab[skin_mask].mean(axis=0)
    src_std = patch_lab[skin_mask].std(axis=0) + 1e-6
    dst_mean = target_lab[skin_mask].mean(axis=0)
    dst_std = target_lab[skin_mask].std(axis=0) + 1e-6
    matched = (patch_lab - src_mean) / src_std * dst_std + dst_mean
    if skin_sample and skin_tint > 0:
        sample_rgb = np.array(skin_sample[:3], dtype=np.uint8).reshape(1, 1, 3)
        sample_lab = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
        sample_matched = patch_lab + (sample_lab - src_mean)
        matched = matched * (1.0 - skin_tint) + sample_matched * skin_tint

    dark_weight = smoothstep(low, high, gray)
    dark_weight = np.clip(dark_weight + (1.0 - protect_dark) * 0.35, 0, 1)
    effective_strength = max(strength, skin_tint if skin_sample else 0.0)
    blend_weight = np.clip(effective_strength * dark_weight, 0.0, 1.0)[..., None]
    out_lab = patch_lab * (1 - blend_weight) + matched * blend_weight
    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB).astype(np.float32)

    # Make imported skin disappear into the target while keeping dark brow strokes.
    # This is closer to the PS workflow for eyebrow transfer than plain color shifting.
    alpha_frac = alpha.astype(np.float32) / 255.0
    skin_fill = estimate_skin_fill(target_rgb, alpha_mask, skin_weight, skin_sample)
    skin_fill_rgb = np.zeros_like(target_rgb, dtype=np.float32)
    skin_fill_rgb[:, :] = skin_fill
    cover_weight = (old_brow_cover * skin_weight * alpha_frac)[..., None]
    covered_target = target_rgb.astype(np.float32) * (1.0 - cover_weight) + skin_fill_rgb * cover_weight

    skin_fade = (skin_tint * skin_weight * alpha_frac)[..., None]
    out_rgb = out_rgb * (1.0 - skin_fade) + covered_target * skin_fade
    return np.clip(out_rgb, 0, 255).astype(np.uint8)


def alpha_blend(target_rgba, source_canvas, req: BlendRequest):
    target_rgb = target_rgba[:, :, :3]
    patch_rgb = source_canvas[:, :, :3]
    alpha = source_canvas[:, :, 3]
    patch_rgb = color_match_patch(
        patch_rgb,
        target_rgb,
        alpha,
        req.color_match,
        req.protect_dark,
        req.skin_sample,
        req.skin_tint,
        req.old_brow_cover,
    )
    a = (alpha.astype(np.float32) / 255.0) * max(0.0, min(req.opacity, 1.0))
    out_rgb = (patch_rgb.astype(np.float32) * a[..., None] + target_rgb.astype(np.float32) * (1 - a[..., None]))
    out = target_rgba.copy()
    out[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    return out


def poisson_blend(target_rgba, source_canvas, req: BlendRequest):
    target_rgb = target_rgba[:, :, :3]
    patch_rgb = source_canvas[:, :, :3]
    alpha = source_canvas[:, :, 3]
    patch_rgb = color_match_patch(
        patch_rgb,
        target_rgb,
        alpha,
        req.color_match,
        req.protect_dark,
        req.skin_sample,
        req.skin_tint,
        req.old_brow_cover,
    )
    mask = (alpha > 8).astype(np.uint8) * 255
    if np.count_nonzero(mask) < 20:
        return target_rgba
    ys, xs = np.where(mask > 0)
    center = (int(round((xs.min() + xs.max()) / 2)), int(round((ys.min() + ys.max()) / 2)))
    mode = cv2.MIXED_CLONE if req.blend_mode == "mixed_clone" else cv2.NORMAL_CLONE
    try:
        cloned = cv2.seamlessClone(
            cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR),
            mask,
            center,
            mode,
        )
        clone_rgb = cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB)
        a = (alpha.astype(np.float32) / 255.0) * max(0.0, min(req.opacity, 1.0))
        out_rgb = clone_rgb.astype(np.float32) * a[..., None] + target_rgb.astype(np.float32) * (1 - a[..., None])
        out = target_rgba.copy()
        out[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        return out
    except Exception:
        return alpha_blend(target_rgba, source_canvas, req)


def build_blend(req: BlendRequest):
    source = np.array(decode_data_url(req.source_data))
    target = np.array(decode_data_url(req.target_data))
    mask_image = decode_data_url(req.source_mask_data)
    source_mask = normalize_mask(mask_image, (source.shape[1], source.shape[0]))
    source_mask = make_protected_soft_patch_mask(
        source,
        source_mask,
        req.contract,
        req.feather,
        req.protect_dark,
    )
    base_patch = crop_patch(source, source_mask, pad=max(8, int(req.pad) + 12))
    placements = req.patches or [
        PatchPlacement(
            name="patch",
            x=req.x,
            y=req.y,
            scale=req.scale,
            rotation=req.rotation,
            flip_x=req.flip_x,
        )
    ]
    out = target
    for placement in placements:
        patch = transform_patch(
            base_patch,
            placement.scale,
            placement.rotation,
            placement.flip_x,
        )
        canvas = patch_canvas(out.shape, patch, placement.x, placement.y)
        if req.blend_mode == "alpha":
            out = alpha_blend(out, canvas, req)
        else:
            out = poisson_blend(out, canvas, req)
    return Image.fromarray(out, "RGBA")


def safe_output_name(name: str, source_name: str, target_name: str):
    if name:
        stem = Path(name).stem
    else:
        stem = f"{Path(target_name).stem or 'target'}_brow_blend_{time.strftime('%Y%m%d_%H%M%S')}"
    stem = re.sub(r"[\\/:*?\"<>|]+", "_", stem).strip() or "brow_blend"
    return stem + ".png"


def safe_patch_name(name: str, source_name: str, mode: str):
    if name:
        stem = Path(name).stem
    else:
        mode_label = "line" if mode == "hair_only" else "soft"
        stem = f"{Path(source_name).stem or 'source'}_brow_patch_{mode_label}_{time.strftime('%Y%m%d_%H%M%S')}"
    stem = re.sub(r"[\\/:*?\"<>|]+", "_", stem).strip() or "brow_patch"
    return stem + ".png"


def safe_tuned_patch_name(name: str, source_name: str, target_name: str):
    if name:
        stem = Path(name).stem
    else:
        stem = (
            f"{Path(source_name).stem or 'source'}_to_"
            f"{Path(target_name).stem or 'target'}_tuned_patch_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    stem = re.sub(r"[\\/:*?\"<>|]+", "_", stem).strip() or "brow_tuned_patch"
    return stem + ".png"


def build_export_patch(req: ExportPatchRequest):
    source = np.array(decode_data_url(req.source_data))
    mask_image = decode_data_url(req.source_mask_data)
    source_mask = normalize_mask(mask_image, (source.shape[1], source.shape[0]))

    if req.patch_mode == "soft_patch":
        adjusted_mask = make_protected_soft_patch_mask(
            source,
            source_mask,
            req.contract,
            req.feather,
            req.protect_dark,
        )
        return Image.fromarray(crop_patch(source, adjusted_mask, pad=max(8, int(req.pad) + 12)), "RGBA")

    region_mask = adjust_mask(source_mask, 0, max(0, min(int(req.feather), 10)))
    alpha = hair_line_alpha(source, region_mask)
    hair_rgba = decontaminate_hair_patch(source, alpha)
    return Image.fromarray(crop_rgba_by_alpha(hair_rgba, alpha, pad=max(8, int(req.pad) + 12)), "RGBA")


def build_tuned_patch(req: BlendRequest):
    source = np.array(decode_data_url(req.source_data))
    target = np.array(decode_data_url(req.target_data))
    mask_image = decode_data_url(req.source_mask_data)
    source_mask = normalize_mask(mask_image, (source.shape[1], source.shape[0]))
    source_mask = make_protected_soft_patch_mask(
        source,
        source_mask,
        req.contract,
        req.feather,
        req.protect_dark,
    )
    base_patch = crop_patch(source, source_mask, pad=max(8, int(req.pad) + 12))
    placement = (req.patches or [
        PatchPlacement(name="patch", x=req.x, y=req.y, scale=req.scale, rotation=req.rotation, flip_x=req.flip_x)
    ])[0]
    patch = transform_patch(base_patch, placement.scale, placement.rotation, placement.flip_x)
    canvas = patch_canvas(target.shape, patch, placement.x, placement.y)
    alpha = canvas[:, :, 3].copy()
    if np.count_nonzero(alpha > 2) < 20:
        raise HTTPException(status_code=400, detail="当前贴片太小或在目标图外")
    target_rgb = target[:, :, :3]
    patch_rgb = color_match_patch(
        canvas[:, :, :3],
        target_rgb,
        alpha,
        req.color_match,
        req.protect_dark,
        req.skin_sample,
        req.skin_tint,
        req.old_brow_cover,
    )
    alpha = np.clip(alpha.astype(np.float32) * max(0.0, min(req.opacity, 1.0)), 0, 255).astype(np.uint8)
    tuned = canvas.copy()
    tuned[:, :, :3] = patch_rgb
    tuned[:, :, 3] = alpha
    tuned[:, :, :3][alpha == 0] = 0
    return Image.fromarray(crop_rgba_by_alpha(tuned, alpha, pad=max(8, int(req.pad) + 12)), "RGBA")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.post("/api/load_path")
def load_path(req: PathRequest):
    path = Path(os.path.expanduser(req.path)).resolve()
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {path}")
    return data_url_from_image(path)


@app.post("/api/blend")
def blend(req: BlendRequest):
    result = build_blend(req)
    out_path = ""
    if req.save:
        out_dir = Path(os.path.expanduser(req.output_dir or str(DEFAULT_OUTPUT_DIR))).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / safe_output_name(req.output_name, req.source_name, req.target_name))
        result.save(out_path)
    return {
        "width": result.width,
        "height": result.height,
        "path": out_path,
        "data_url": image_to_data_url(result),
    }


@app.post("/api/export_patch")
def export_patch(req: ExportPatchRequest):
    result = build_export_patch(req)
    out_path = ""
    if req.save:
        out_dir = Path(os.path.expanduser(req.output_dir or str(DEFAULT_PATCH_DIR))).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / safe_patch_name(req.output_name, req.source_name, req.patch_mode))
        result.save(out_path)
    return {
        "width": result.width,
        "height": result.height,
        "path": out_path,
        "data_url": image_to_data_url(result),
    }


@app.post("/api/export_tuned_patch")
def export_tuned_patch(req: BlendRequest):
    result = build_tuned_patch(req)
    out_path = ""
    if req.save:
        out_dir = Path(os.path.expanduser(req.output_dir or str(DEFAULT_PATCH_DIR))).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / safe_tuned_patch_name(req.output_name, req.source_name, req.target_name))
        result.save(out_path)
    return {
        "width": result.width,
        "height": result.height,
        "path": out_path,
        "data_url": image_to_data_url(result),
    }


def main():
    parser = argparse.ArgumentParser(description="PS式换眉贴合工具")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8799)
    args = parser.parse_args()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_PATCH_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
