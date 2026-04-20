---
layout: post
title: "Week 3: Circuit Tracing Gemma 2B with circuit-tracer"
date: 2026-04-07 00:00:00-0000
description: Using circuit-tracer and GemmaScope transcoders to compare Dallas and Houston prompts, inspect top internal features, and export interactive attribution graphs.
tags: mechanistic-interpretability gemma circuit-tracing attribution-graphs transcoders
categories: interpretability
giscus_comments: true
related_posts: false
---

This week moved from activation summaries to **feature-level circuit tracing**.

Instead of just asking where the residual stream changes, I wanted to know which
specific internal features help Gemma 2B answer a factual prompt. I used
[`circuit-tracer`](https://github.com/safety-research/circuit-tracer) with the
GemmaScope transcoder set and compared two nearly identical prompts:

1. `The capital of the state containing Dallas is`
2. `The capital of the state containing Houston is`

That setup is useful because Dallas and Houston should both route through the
same state-level fact (**Texas**) and then the same capital-city fact
(**Austin**). If the model has reusable internal representations for those
relations, the two prompt circuits should overlap substantially.

The week 3 notebook does four things:

- loads Gemma 2B with `circuit-tracer`
- computes attribution graphs for Dallas and Houston prompts
- ranks the most influential internal features
- exports interactive graph files for local inspection

On this run, both graphs are clean and fairly compact. The Dallas graph prunes
to **296 nodes / 17,024 edges**; the Houston graph prunes to **301 nodes /
17,172 edges**. In both cases the top retained output logit is **`Austin`**, and
**`Texas`** also appears among the salient target logits.

The strongest features are also highly stable across prompts. Among the top 40
features, ignoring token position, the Dallas and Houston graphs share **31**
`(layer, feature)` pairs for a **Jaccard overlap of 0.674**. The same two
features dominate both runs:

- `L20 · feature 15589`
- `L24 · feature 6044`

Several other recurring features reappear across both prompts, including
`L21 · 5943`, `L14 · 2268`, `L4 · 13154`, `L16 · 25`, and `L7 · 6861`. That is
exactly the pattern I hoped to see: the model seems to be reusing a fairly
stable factual scaffold rather than building two unrelated circuits for Dallas
and Houston.

Layer-wise influence is also informative. The biggest influence mass sits in
**layers 0, 24, 20, and 4** in both runs, with Dallas additionally leaning a bit
more on layer 2 and Houston leaning more on layer 18. My read is that early
layers still supply token- and syntax-level scaffolding, while later layers are
doing the heavier factual composition.

Here is the layer-level summary plot produced by the notebook:

{% include figure.liquid path="assets/img/week3_gemma_layer_influence.png" class="img-fluid rounded z-depth-1" %}

I also wanted the graph itself in the post rather than only as a local export, so
I copied the `circuit-tracer` frontend into a static viewer directory and wired it
to the Dallas/Houston graph JSON files. The embed below opens on the Dallas graph,
and the prompt selector can switch to Houston:

{::nomarkdown}
<div style="margin: 2rem 0;">
  <iframe
    src="{{ '/assets/circuit-tracer/week3/index.html?slug=week3-gemma-dallas' | relative_url }}"
    title="Week 3 Gemma circuit graph"
    style="width: 100%; height: 900px; border: 1px solid #ddd; border-radius: 8px;">
  </iframe>
</div>
<p>
  If the embedded viewer does not load, open
  <a href="{{ '/assets/circuit-tracer/week3/index.html?slug=week3-gemma-dallas' | relative_url }}">the full viewer page</a>
  directly.
</p>
{:/nomarkdown}

The full code and outputs are in the notebook below.

{::nomarkdown}
<details>
<summary style="cursor:pointer; font-weight:bold; padding:6px 0;">View notebook</summary>
{% assign jupyter_path = '/assets/jupyter/week3_gemma_circuit_tracing.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists /assets/jupyter/week3_gemma_circuit_tracing.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Notebook not found - make sure <code>week3_gemma_circuit_tracing.ipynb</code> is present in the site root or update <code>jupyter_path</code>.</p>
{% endif %}
</details>
{:/nomarkdown}

---

## Takeaway

- The Dallas and Houston prompts reuse a large shared core circuit rather than
  producing unrelated explanations.
- The strongest shared features sit in middle and late layers, which is where I
  would expect the model to assemble the factual relation.
- The graph export step matters: once the static summary identifies stable
  features, the interactive viewer is the right place to inspect the actual
  circuit structure node by node.
