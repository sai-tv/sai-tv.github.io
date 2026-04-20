---
layout: post
title: "Week 4: Circuit Tracing Homonym Disambiguation in Gemma 2B"
date: 2026-04-08 00:00:00-0000
description: Auditing homonym prompts in Gemma 2B, selecting the strongest same-word case, and tracing where the model's internal mappings split between two senses.
tags: mechanistic-interpretability gemma circuit-tracing homonyms lexical-disambiguation
categories: interpretability
giscus_comments: true
related_posts: false
---

Week 4 focuses on something I learnt in 3rd grade: homonyms.

**homonyms**: the same word that means different things, depending on context, something that LLMs seem to be
doing pretty well with.

Both prompts can target the **same next token** while only the sense
changes. So the main mechanistic question becomes:

How much of Gemma 2B's internal computation is shared when it maps the same word
surface form from two different contexts, and where does the model start to
separate those meanings?

The notebook starts with an audit over several candidate homonyms:

- `bank`
- `bat`
- `bark`
- `watch`

For each word, it evaluates two sense-specific prompts and checks whether the
intended target token is actually well supported in both contexts. That matters
because a circuit is much easier to interpret when the model is naturally
leaning toward the intended word.

Once the notebook selects the strongest word, it traces one prompt for each
sense while forcing the same target token id in both runs. That keeps the graph
centered on the chosen word instead of drifting toward punctuation or some other
higher-probability fallback token.

The comparison then uses three overlap views:

- **Top-40 feature Jaccard**: do the same `(layer, feature_idx)` pairs recur?
- **Weighted feature-vector cosine**: does the total influence mass land on the same latent directions?
- **Internal target-subgraph weighted edge overlap**: after dropping prompt token nodes, how similar is the retained internal routing into the shared target word?

That last metric is the most useful one for this week. Because the output token
is the same in both prompts, any difference in the internal target-subgraph is
evidence about **sense disambiguation**, not just about different output
spellings.

Here is the fixed-path layer influence plot for the selected word:

{% capture layer_exists %}{% file_exists assets/img/week4_homonym_selected_word_layer_influence.png %}{% endcapture %}
{% if layer_exists == 'true' %}
{% include figure.liquid path="assets/img/week4_homonym_selected_word_layer_influence.png" alt="Week 4 homonym selected-word layer influence" class="img-fluid rounded z-depth-1" max-width="100%" %}
{% endif %}

The embed below opens on the finance-sense graph,
and the prompt selector can switch to the river-sense graph:

{::nomarkdown}
<div style="margin: 2rem 0;">
  <iframe
    src="{{ '/assets/circuit-tracer/week4_graph_viewer/index.html?slug=week4-gemma-homonym-bank-finance' | relative_url }}"
    title="Week 4 Gemma homonym circuit graph"
    style="width: 100%; height: 900px; border: 1px solid #ddd; border-radius: 8px;">
  </iframe>
</div>
<p>
  If the embedded viewer does not load, open
  <a href="{{ '/assets/circuit-tracer/week4_graph_viewer/index.html?slug=week4-gemma-homonym-bank-finance' | relative_url }}">the full viewer page</a>
  directly.
</p>
{:/nomarkdown}

The notebook stores the full summary, including the selected word and overlap
metrics, in `week4_gemma_homonym_circuit_summary.pt`.

The full code and outputs are in the notebook below.

{::nomarkdown}
<details>
<summary style="cursor:pointer; font-weight:bold; padding:6px 0;">View notebook</summary>
{% assign jupyter_path = 'assets/jupyter/week4_gemma_homonym_circuit_tracing.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/week4_gemma_homonym_circuit_tracing.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Notebook not found - make sure <code>week4_gemma_homonym_circuit_tracing.ipynb</code> is present in the site root or update <code>jupyter_path</code>.</p>
{% endif %}
</details>
{:/nomarkdown}

---


## What you may notice

- It's crazy how the model truly understands context: immediately after "deposits" the model makes the connection to a financial institution and looks for the financial "bank"
- I guess with river, and the preposition "on", it starts looking for locations, directions, etc. and then narrows down on the word "bank" for a location by the river.
- The two graphs seem to split early, with larger splits at the end of the model: more contextual changes hit harder at the end I guess. But the context of "deposit" and "paycheck" seem to guide the model in the right direction almost immediately. The "on" combined with "river" gives it a completely different context as well. Different scaffolds.
- Each of the words is the same combination of letters, but is in a completely different mapping: different from how a 3rd grader would understand and get confused I guess: context is given much more importance here, more robotic.