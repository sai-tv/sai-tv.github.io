---
layout: post
title: "Week 1: Inspecting GPT-2 Small with TransformerLens"
date: 2026-04-03 00:00:00-0000
description: Loading GPT-2 small, capturing residual-stream activations at selected layers, and visualising attention patterns across early, mid, and late heads.
tags: mechanistic-interpretability transformers attention gpt2
categories: interpretability
giscus_comments: true
related_posts: false
---

I really want to try understanding LLMs: interpretability seems the place to start.

This one's a first hands-on inspection of **GPT-2 small** (117 M parameters)
using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
This week, I wanted to:

1. Load the model and verify tensor shapes at key hook points.
2. Run three fixed prompts through `run_with_cache` and capture residual-stream
   activations.
3. Save a compact activation summary to disk for later analysis and have a little 
   interactive feature on this post where you can see activations of different
   layers and heads.

The notebook below contains the full code, observations, and then a little interactive plot
where you can play around with layers and heads. Patterns on first pass seem pretty all over
the place; perhaps more analysis will clarify things.

Two things sort of stood out: there's an attention sink at the end-of-text token where a majority
of the attention is directed in many of the heads and layers: normalizing without
this portion gives somewhat of a more informative summary. Also, self-attention seems to 
also be a dominant phenomenon for tokens in many layers: there's other ones that look at
adjacent words and somewhat "understand" (?) their relationships? Curious. Is this how we learn too...

The full code and observations are in the notebook below — click to expand.

{::nomarkdown}
<details>
<summary style="cursor:pointer; font-weight:bold; padding:6px 0;">View notebook</summary>
{% assign jupyter_path = 'assets/jupyter/week1_inspection.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/week1_inspection.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Notebook not found — make sure <code>week1_inspection.ipynb</code> is in <code>assets/jupyter/</code>.</p>
{% endif %}
</details>
{:/nomarkdown}

---

## Interactive attention explorer

Browse all 12 layers and 12 heads across three prompts, or view averages.
Hover a cell for the exact weight.

<div
  id="attn-explorer"
  data-src="{{ '/assets/json/attention_data.json' | relative_url }}"
  style="max-width:540px; margin:2rem auto; font-family:monospace; font-size:0.9rem;">

  <div style="display:flex; flex-wrap:wrap; gap:16px; margin-bottom:14px;">
    <label style="display:flex; flex-direction:column; gap:4px;">
      Prompt
      <select id="prompt-select" style="font-family:monospace; font-size:0.85rem; padding:3px 6px;"></select>
    </label>
    <label style="display:flex; flex-direction:column; gap:4px;">
      View
      <select id="mode-select" style="font-family:monospace; font-size:0.85rem; padding:3px 6px;">
        <option value="single">Single head</option>
        <option value="avg-heads">Avg across heads (fix layer)</option>
        <option value="avg-layers">Avg across layers (fix head)</option>
      </select>
    </label>
    <label style="display:flex; flex-direction:column; gap:4px; justify-content:flex-end;">
      <span style="display:flex; align-items:center; gap:6px; cursor:pointer;">
        <input type="checkbox" id="exclude-bos" style="cursor:pointer;">
        Exclude &lt;|endoftext|&gt;
      </span>
    </label>
  </div>

  <div style="display:flex; gap:32px; margin-bottom:14px;">
    <label id="layer-label" style="display:flex; flex-direction:column; gap:4px; align-items:center;">
      Layer
      <input type="range" id="layer-slider" min="0" max="11" value="0" style="width:160px;">
      <span id="layer-val">0</span>
    </label>
    <label id="head-label" style="display:flex; flex-direction:column; gap:4px; align-items:center;">
      Head
      <input type="range" id="head-slider" min="0" max="11" value="0" style="width:160px;">
      <span id="head-val">0</span>
    </label>
  </div>

  <canvas id="attn-canvas"></canvas>
  <div id="attn-tooltip" style="position:absolute; display:none; pointer-events:none; background:rgba(0,0,0,0.85); color:#fff; padding:6px 8px; border-radius:4px; font-size:12px; line-height:1.2; z-index:10;"></div>
  <p id="attn-status" style="margin-top:10px; color:#666;"></p>
</div>

<script>
(function () {
  var allData   = null;
  var promptIdx = 0;
  var hoverCell = null;
  var layout    = null;
  var curMatrix = null; // the matrix currently on screen (used by tooltip)

  function setStatus(msg, isError) {
    var el = document.getElementById('attn-status');
    if (!el) return;
    el.textContent = msg || '';
    el.style.color = isError ? '#b00020' : '#666';
  }

  function currentLayer()      { return +document.getElementById('layer-slider').value; }
  function currentHead()       { return +document.getElementById('head-slider').value;  }
  function currentMode()       { return document.getElementById('mode-select').value;   }
  function excludingBos()      { return document.getElementById('exclude-bos').checked; }

  // Zero out column 0 (BOS/endoftext key) then renormalize each row to sum to 1
  function maskBos(matrix) {
    return matrix.map(function (row) {
      var masked = row.map(function (v, k) { return k === 0 ? 0 : v; });
      var rowSum = masked.reduce(function (a, b) { return a + b; }, 0);
      return rowSum > 0 ? masked.map(function (v) { return v / rowSum; }) : masked;
    });
  }

  // Elementwise average of an array of same-shape 2-D arrays
  function avgMatrices(matrices) {
    var n = matrices[0].length;
    var result = [];
    for (var q = 0; q < n; q++) {
      result[q] = [];
      for (var k = 0; k < n; k++) {
        var sum = 0;
        for (var m = 0; m < matrices.length; m++) sum += matrices[m][q][k];
        result[q][k] = sum / matrices.length;
      }
    }
    return result;
  }

  function getMatrix(entry, layer, head, mode) {
    if (mode === 'avg-heads') {
      // average all 12 heads at this layer
      var mats = entry.patterns[String(layer)]; // array of 12 matrices
      return avgMatrices(mats);
    }
    if (mode === 'avg-layers') {
      // average all 12 layers at this head
      var mats = [];
      for (var l = 0; l < 12; l++) mats.push(entry.patterns[String(l)][head]);
      return avgMatrices(mats);
    }
    return entry.patterns[String(layer)][head];
  }

  function titleFor(entry, layer, head, mode) {
    var base = '\u201C' + entry.prompt + '\u201D';
    if (mode === 'avg-heads') return 'Avg across all heads  |  Layer ' + layer + '  |  ' + base;
    if (mode === 'avg-layers') return 'Avg across all layers  |  Head ' + head + '  |  ' + base;
    return 'Layer ' + layer + ' \xB7 Head ' + head + '  |  ' + base;
  }

  // Blues colormap: 5-stop interpolation
  var BLUES = [[247,251,255],[198,219,239],[107,174,214],[33,113,181],[8,48,107]];
  function blues(v) {
    var t = Math.max(0, Math.min(1, v)) * (BLUES.length - 1);
    var i = Math.min(Math.floor(t), BLUES.length - 2);
    var f = t - i;
    var a = BLUES[i], b = BLUES[i + 1];
    return 'rgb(' + Math.round(a[0]+f*(b[0]-a[0])) + ','
                  + Math.round(a[1]+f*(b[1]-a[1])) + ','
                  + Math.round(a[2]+f*(b[2]-a[2])) + ')';
  }

  function updateSliderVisibility(mode) {
    var layerLabel = document.getElementById('layer-label');
    var headLabel  = document.getElementById('head-label');
    layerLabel.style.opacity = (mode === 'avg-layers') ? '0.35' : '1';
    headLabel.style.opacity  = (mode === 'avg-heads')  ? '0.35' : '1';
    document.getElementById('layer-slider').disabled = (mode === 'avg-layers');
    document.getElementById('head-slider').disabled  = (mode === 'avg-heads');
  }

  function draw() {
    if (!allData) return;
    var layer  = currentLayer();
    var head   = currentHead();
    var mode   = currentMode();
    var entry  = allData[promptIdx];
    var tokens = entry.tokens;
    var n      = tokens.length;

    curMatrix = getMatrix(entry, layer, head, mode);
    if (excludingBos()) curMatrix = maskBos(curMatrix);
    updateSliderVisibility(mode);

    var canvas = document.getElementById('attn-canvas');
    var ctx    = canvas.getContext('2d');
    var cell   = 34;
    var left   = 130, top = 56, right = 24, bottom = 120;
    var gridW  = n * cell, gridH = n * cell;

    canvas.width  = left + gridW + right;
    canvas.height = top  + gridH + bottom;
    layout = { left: left, top: top, cell: cell, n: n };

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#222';
    ctx.font = '12px sans-serif';
    ctx.textBaseline = 'middle';
    ctx.fillText(titleFor(entry, layer, head, mode), left, 20);

    for (var q = 0; q < n; q++) {
      for (var k = 0; k < n; k++) {
        var x = left + k * cell;
        var y = top  + q * cell;
        ctx.fillStyle = blues(curMatrix[q][k]);
        ctx.fillRect(x, y, cell, cell);
        ctx.strokeStyle = 'rgba(0,0,0,0.08)';
        ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      }
    }

    if (hoverCell && hoverCell.q >= 0 && hoverCell.k >= 0 && hoverCell.q < n && hoverCell.k < n) {
      ctx.strokeStyle = '#111';
      ctx.lineWidth = 2;
      ctx.strokeRect(left + hoverCell.k * cell + 1, top + hoverCell.q * cell + 1, cell - 2, cell - 2);
      ctx.lineWidth = 1;
    }

    ctx.fillStyle = '#222';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'right';
    for (var r = 0; r < n; r++)
      ctx.fillText(tokens[r], left - 10, top + r * cell + cell / 2);

    ctx.save();
    ctx.textAlign = 'left';
    for (var c = 0; c < n; c++) {
      var tx = left + c * cell + 4, ty = top + gridH + 6;
      ctx.translate(tx, ty);
      ctx.rotate(Math.PI / 4);
      ctx.fillText(tokens[c], 0, 0);
      ctx.rotate(-Math.PI / 4);
      ctx.translate(-tx, -ty);
    }
    ctx.restore();

    ctx.fillStyle = '#333';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Key (source)', left + gridW / 2, canvas.height - 18);
    ctx.save();
    ctx.translate(20, top + gridH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Query (destination)', 0, 0);
    ctx.restore();
  }

  function handleHover(evt) {
    if (!allData || !layout || !curMatrix) return;
    var tooltip = document.getElementById('attn-tooltip');
    var canvas  = document.getElementById('attn-canvas');
    var rect    = canvas.getBoundingClientRect();
    var sx = canvas.width / rect.width, sy = canvas.height / rect.height;
    var x  = (evt.clientX - rect.left) * sx;
    var y  = (evt.clientY - rect.top)  * sy;
    var k  = Math.floor((x - layout.left) / layout.cell);
    var q  = Math.floor((y - layout.top)  / layout.cell);

    if (k < 0 || q < 0 || k >= layout.n || q >= layout.n) {
      hoverCell = null;
      tooltip.style.display = 'none';
      draw();
      return;
    }

    hoverCell = { q: q, k: k };
    var tokens = allData[promptIdx].tokens;
    tooltip.textContent = tokens[q] + ' \u2192 ' + tokens[k] + ': ' + curMatrix[q][k].toFixed(3);
    tooltip.style.display = 'block';
    tooltip.style.left = (evt.clientX - rect.left + 12) + 'px';
    tooltip.style.top  = (evt.clientY - rect.top  + 12) + 'px';
    draw();
  }

  function init() {
    var jsonPath = document.getElementById('attn-explorer').dataset.src;
    var canvas   = document.getElementById('attn-canvas');

    setStatus('Loading attention data...');
    fetch(jsonPath)
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (json) {
        allData = json;
        var sel = document.getElementById('prompt-select');
        json.forEach(function (entry, idx) {
          var opt = document.createElement('option');
          opt.value = idx;
          opt.textContent = entry.prompt;
          sel.appendChild(opt);
        });
        setStatus('');
        draw();
      })
      .catch(function (e) { setStatus('Failed to load attention data: ' + e.message, true); });

    document.getElementById('prompt-select').addEventListener('change', function () { promptIdx = +this.value; hoverCell = null; draw(); });
    document.getElementById('mode-select').addEventListener('change',   function () { hoverCell = null; draw(); });
    document.getElementById('exclude-bos').addEventListener('change',   function () { hoverCell = null; draw(); });
    document.getElementById('layer-slider').addEventListener('input',   function () { document.getElementById('layer-val').textContent = this.value; hoverCell = null; draw(); });
    document.getElementById('head-slider').addEventListener('input',    function () { document.getElementById('head-val').textContent  = this.value; hoverCell = null; draw(); });

    canvas.addEventListener('mousemove',  handleHover);
    canvas.addEventListener('mouseleave', function () { hoverCell = null; document.getElementById('attn-tooltip').style.display = 'none'; draw(); });
  }

  document.readyState === 'loading'
    ? document.addEventListener('DOMContentLoaded', init)
    : init();
})();
</script>
