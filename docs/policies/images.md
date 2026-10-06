---
title: "What dgem Can Do With Images"
description: "Strengths and limits of DiffusionGemma on images: fast categorical questions (presence, 3x3 location, relations, count, quality) in one pass, with hesitation to route hard cases to Gemini; not bounding boxes, occlusion or screenshot grounding."
---

# What dgem Can Do With Images

DiffusionGemma reads an image with Gemma 4's vision tower and answers a policy's questions about it in one forward
pass. This page says which image questions that works for, based on our tests on photos, synthetic UI scenes and
app/web screenshots. Re-check on your own images with `dgem bench-vision` (below).

## The short version

- **Use dgem for fast categorical questions about an image.** It can answer several questions together in one pass,
  much faster than asking a large multimodal model such as Gemini.
- **It is less accurate than Gemini on every question we tested.** Its value is speed and cost, plus a hesitation score
  that says when to ask Gemini instead.
- **Do not use it for coordinates.** Use a detector or Gemini for bounding boxes and masks, for "is this hidden or cut
  off?", or for locating elements in app and web screenshots.

## Supported questions

| Question | Photos | Synthetic UI scenes | App and web screenshots |
| :--- | :---: | :---: | :---: |
| Is the described thing in the image? | Strong | Strong | Usable |
| Is it left of / right of / above / below another thing? | Strong | Usable with a cascade | Not supported |
| Which cell of a 3×3 grid holds it? | Usable with a cascade | Usable with a cascade | Not supported |
| How many elements (1–4)? | — | Usable with a cascade | — |
| Image quality (clean, blurred, noisy, compressed, low contrast)? | — | Usable with a cascade | — |
| Is part of it hidden or cut off? | Not supported | Not supported | — |

Notes:
- On a blank image, every answer collapses to a fixed default (absent, centre cell, one element, clean). The answers
  come from the image, not the prompt.
- Answers are mostly, but not always, stable across identical requests.

**Not supported:**
- Occlusion: it answers "not hidden" almost every time.
- Grid cell and relations on app or web screenshots.
- Boxes. dgem's own boxes are coarse on photos and poor on screenshots, and they move when the image is flipped or
  padded.

## Hesitation tells you when to ask Gemini

Each answer comes with a hesitation score (normalized entropy). High hesitation picks out wrong answers, more
reliably for some questions than others. The Stage-2 cascade uses this:
- slots at or above an entropy threshold go to Gemini (`gemini-3.8-flash` by default);
- the image goes to Gemini too;
- `cascade_mode: "entropy"` turns it on, on `POST /api/decide`, the MCP decide tools and Studio batch runs.

In our tests the cascade at the default 0.35 nats raised accuracy well above dgem alone while still answering about
half of the requests without Gemini, so the median request stayed fast; lower thresholds send more requests to Gemini
and approach its accuracy and latency.

Guidance:
- **Ask only the questions you need.** Gemini is called whenever *any* slot of a request is unsure, so every extra
  question raises the share of requests that reach Gemini, and the median latency with it.
- **Keep the threshold at the default 0.35 nats unless you have measured otherwise.** At low thresholds, giving Gemini
  dgem's full answer and distribution as "the candidate" anchored it on dgem's wrong answers. The default
  `stage2_prior: "soft"` only tells Gemini that a fast first stage leaned toward an answer and may be wrong
  ([authoring guide](authoring.md)).

## By domain

Results differ by domain, so check yours. In our tests on mobile UI, web UI, document pages and synthetic circuit
boards, with presence, 3×3 grid cell and relation questions:

| Domain | dgem alone | With a Gemini cascade | Not supported | Skip Gemini on "absent"? |
| :--- | :--- | :--- | :--- | :--- |
| Web UI | present | — | grid cell, relation | no (loses some targets) |
| Mobile UI | — | present, grid cell | relation | yes |
| Document pages | — | grid cell | present, relation | no (saves little) |
| Circuit-board defects | — | grid cell | present, relation | no (misses defects) |

## Boxes, masks and screenshots: use something else

| Need | Use |
| :--- | :--- |
| Box for an object described in a photo | An open-vocabulary detector such as Grounding DINO, or Gemini |
| Box for a UI element from an instruction ("raise the temperature") | Gemini |
| Grade or label boxes without ground truth | Gemini as a judge (`dgem bench-bbox-judge`) |
| Faster Gemini boxes | `dgem locate`, `POST /api/locate` or MCP `locate_object` (defaults: LOW thinking, no hint, no skip). LOW thinking lowers latency; skip Gemini when dgem confidently says absent only on photos and mobile UI. dgem's grid-cell hint helped on photos but not on UI, documents or circuit boards. |
| Masks | SAM prompted with the Gemini (or Grounding DINO) box |

## Validate on your images

```bash
# Ground truth in a JSONL manifest: {"id", "image_path", "target", "aspects": {"present": "yes", "grid_cell": "top_left", ...}}
./bin/dgem bench-vision -d my_images.jsonl -t templates/multimodal/vision_aspects_generic.json.tmpl \
  --vertex-url <ENDPOINT_ID> --gcp-auth --repeat 2 -o mine.json

# The same questions to Gemini, and dgem with the live cascade
./bin/dgem bench-vision -d my_images.jsonl -t templates/multimodal/vision_aspects_generic.json.tmpl --engine gemini -o mine_gemini.json
./bin/dgem bench-vision -d my_images.jsonl -t templates/multimodal/vision_aspects_generic.json.tmpl \
  --vertex-url <ENDPOINT_ID> --gcp-auth --only-scored --cascade-threshold 0.35 -o mine_cascade.json
```

- **Templates:** `templates/multimodal/vision_aspects.json.tmpl` (UI scenes) and `vision_aspects_generic.json.tmpl`
  (photos and screenshots).
- **Regression matrix:** the `vision` suite tracks this on every serving change ([regression matrix](../operate/regression-matrix.md)).
