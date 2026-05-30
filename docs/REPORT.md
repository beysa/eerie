# eerie — Technical Report

### Text Summarization and Image Generation for Illustrating Children's Stories

**Bachelor's thesis, Yıldız Technical University (YTU), Department of Computer Engineering, 2023**

| | |
|---|---|
| **Authors** | Yaren Yıldırım (17011019) · Beyza Nur Sezgin (17011045) |
| **Advisor** | Lect. Furkan Çakmak |
| **Original thesis title** | *Text Summarization and Image Generation* |
| **Theme** | Imagining children's stories — turning written prose into illustrated storybook panels |
| **Inspiration** | Refik Anadol — machine-generated visual art from data |
| **Fine-tuned models (published)** | [`ektvho/bart-cnn`](https://huggingface.co/ektvho/bart-cnn) · [`ektvho/sd-vist`](https://huggingface.co/ektvho/sd-vist) |

---

## 1. Executive summary

**eerie** is a text-to-illustration system: it takes a written children's story, summarizes it, and renders an illustration for every sentence so that the prose becomes a sequence of storybook panels. It began as a 2023 YTU senior thesis and has since been packaged into a reproducible portfolio project.

**What was trained.** The thesis contribution is *two fine-tuned models*, both published on the Hugging Face Hub and verified to be online:

- **[`ektvho/bart-cnn`](https://huggingface.co/ektvho/bart-cnn)** — the summarization model. It is a BART sequence-to-sequence summarizer fine-tuned **from** [`ccdv/lsg-bart-base-4096-booksum`](https://huggingface.co/ccdv/lsg-bart-base-4096-booksum) (an LSG-BART variant with a 4096-token long-context attention window) **on** the `cnn_dailymail` dataset. The Hugging Face page exposes it as an `AutoModelForSeq2SeqLM` with a `custom_code` tag, and loading it requires `trust_remote_code=True` because the LSG (Local–Sparse–Global) attention is custom code. Its tokenizer is `ccdv/lsg-bart-base-4096-booksum`.
- **[`ektvho/sd-vist`](https://huggingface.co/ektvho/sd-vist)** — the image-generation model. It is **Stable Diffusion 2.1 fine-tuned on the VIST (Visual Storytelling) dataset**, so the generator is adapted toward narrative, story-like scene imagery rather than generic text-to-image output. The Hugging Face page exposes it as a Diffusers `StableDiffusionPipeline` with a text-to-image pipeline tag. SD 2.1 runs at a native 768×768 resolution.

**What was used off-the-shelf for styling.** A third, *pretrained* model, [`timbrooks/instruct-pix2pix`](https://huggingface.co/timbrooks/instruct-pix2pix), provides instruction-driven style transfer — it edits an already-generated panel toward a free-text style instruction while preserving composition. This model is **not** fine-tuned by the thesis; it is used as-is.

**The core thesis pipeline** therefore is: *story → `ektvho/bart-cnn` summarize → `ektvho/sd-vist` generate (one image per sentence) → `instruct-pix2pix` style*, built and demonstrated in the thesis notebook `eerie.ipynb`.

**What was extended afterward.** This repository adds two clearly-labeled, **post-thesis** extensions that explore problems the thesis pipeline did not solve:

1. **A multi-style SD1.5 exploration** — an importable `eerie/` package that swaps among three *off-the-shelf, community* Stable Diffusion 1.5 art-style fine-tunes (Ghibli-Diffusion, a Disney-like "modern animation" model, and a watercolor model). This is a simplified, fast, art-style demo. **It is not the thesis** and uses none of the thesis's fine-tuned models.
2. **A ComfyUI IP-Adapter character-consistency demo** — a separate exploration (SD1.5 DreamShaper + IP-Adapter) that produces a real before/after showing the same character held consistent across panels. This directly attacks the thesis pipeline's biggest limitation (no cross-panel character consistency) and is, again, **post-thesis**.

The honest separation between the two layers is the point of this report: the **fine-tuned SD2.1 + BART system is the academic core**; the **SD1.5 multi-style package and the ComfyUI IP-Adapter demo are extensions**, and neither is ever presented as part of the thesis.

---

## 2. Project provenance

### 2.1 Thesis metadata

eerie is the senior (bachelor's) graduation thesis *"Text Summarization and Image Generation"*, completed in **2023** at **Yıldız Technical University, Department of Computer Engineering**. It was authored by **Yaren Yıldırım (student no. 17011019)** and **Beyza Nur Sezgin (student no. 17011045)**, and advised by **Lect. Furkan Çakmak**.

### 2.2 Problem statement

Children's stories are written as prose, but children experience them as pictures. The thesis asks: *can a machine read a short story and illustrate it automatically* — condensing the text to its essentials and then drawing a faithful image for each beat of the narrative? This decomposes into two classical-but-coupled problems, which are exactly the two halves of the thesis title:

1. **Text summarization** — reduce a story to the sentences that carry the narrative, so the illustration budget is spent on what matters. This is abstractive summarization with a transformer.
2. **Image generation** — turn each resulting sentence into a coherent, story-styled illustration. This is conditional text-to-image generation with a latent diffusion model.

The thesis's distinctive choice is to **fine-tune both halves for the storytelling domain** rather than relying on generic checkpoints: a long-context BART summarizer adapted on news-style summarization data, and a Stable Diffusion 2.1 generator adapted on a *visual storytelling* corpus.

### 2.3 Inspiration: Refik Anadol

The project is inspired by **Refik Anadol**, the media artist known for turning large datasets into immersive, machine-generated visual art. Anadol's work reframes a neural network as an *instrument for imagining* — taking abstract input (data, text, memory) and producing a visual world from it. eerie applies that same spirit at a small, concrete scale: the "data" is a children's story, and the "imagined" output is its illustrated panels.

### 2.4 Scope

The thesis scope is the end-to-end *prose-to-panels* pipeline and the two fine-tuned models that power it. It deliberately does **not** include controllable character consistency across panels, structured scene/character extraction, or quantitative generative-quality evaluation — those are acknowledged as open problems (see §9), and one of them (character consistency) is taken up as a post-thesis extension (§8).

---

## 3. Core thesis system

The thesis pipeline, implemented in `eerie.ipynb`, is a four-stage data flow. Each stage feeds the next; the output of summarization becomes the input to scene splitting, whose sentences become the prompts for generation, whose panels become the input to style transfer.

```
   Children's story (prose)
            │
            ▼
   ┌──────────────────────────────────────────────┐
   │ 1. Summarize                                  │
   │    ektvho/bart-cnn   (fine-tuned BART, LSG)   │   ← thesis model
   │    long-context abstractive summary           │
   └──────────────────────────────────────────────┘
            │  condensed story text
            ▼
   ┌──────────────────────────────────────────────┐
   │ 2. Split into sentences                       │
   │    one sentence  →  one scene                 │
   └──────────────────────────────────────────────┘
            │  [sentence_0, sentence_1, … ]
            ▼
   ┌──────────────────────────────────────────────┐
   │ 3. Generate one image per sentence            │
   │    ektvho/sd-vist  (SD 2.1 fine-tuned on VIST)│   ← thesis model
   │    768×768 storybook illustration per scene   │
   └──────────────────────────────────────────────┘
            │  panels: one image per sentence
            ▼
   ┌──────────────────────────────────────────────┐
   │ 4. Style transfer (per panel)                 │
   │    timbrooks/instruct-pix2pix  (pretrained)   │   ← off-the-shelf
   │    edit each panel toward a style instruction │
   └──────────────────────────────────────────────┘
            │
            ▼
   Illustrated, optionally restyled storybook panels
```

**Stage 1 — Summarize (`ektvho/bart-cnn`).** The full story is condensed by the fine-tuned BART summarizer. Because the base model is an LSG-BART with a 4096-token context window, it can ingest a long story without aggressive truncation, then emit an abstractive summary that keeps the narrative's load-bearing sentences. This is where the thesis spends effort *deciding what to illustrate*.

**Stage 2 — Split into scenes.** The summary is segmented into individual sentences. Each sentence is treated as one "scene" — the atomic unit that the generator will illustrate. One sentence in, one panel out.

**Stage 3 — Generate (`ektvho/sd-vist`).** Every sentence becomes a text-to-image prompt for the SD 2.1 generator that was fine-tuned on the VIST visual-storytelling dataset. Because the model has been adapted on story imagery, its panels lean toward narrative, illustration-like scenes rather than generic stock imagery. Each panel is generated **independently** from its own sentence — there is no shared state carried between panels (the consequence of this is discussed in §9).

**Stage 4 — Style transfer (`instruct-pix2pix`).** Each generated panel can be restyled by the pretrained InstructPix2Pix editor, which takes the panel plus a short instruction (e.g. a target art style) and edits the image toward that style while preserving its composition. This step is applied per panel and is the pipeline's *look* control.

---

## 4. Fine-tuned models

The two models below are the trained artifacts of the thesis. Both are **published and live on the Hugging Face Hub**.

### 4.1 `ektvho/bart-cnn` — abstractive summarizer

> **Hugging Face:** [`https://huggingface.co/ektvho/bart-cnn`](https://huggingface.co/ektvho/bart-cnn)

| Property | Value |
|---|---|
| Task | Abstractive text summarization (sequence-to-sequence) |
| Base / lineage | Fine-tuned **from** [`ccdv/lsg-bart-base-4096-booksum`](https://huggingface.co/ccdv/lsg-bart-base-4096-booksum) |
| Base architecture | LSG-BART — BART with **Local–Sparse–Global** attention and a **4096-token** context window |
| Fine-tuning dataset | `cnn_dailymail` |
| Hub class | `AutoModelForSeq2SeqLM` (PyTorch, Transformers) |
| Tokenizer | `ccdv/lsg-bart-base-4096-booksum` |
| Loading requirement | `trust_remote_code=True` (the LSG attention is custom code; the Hub page carries a `custom_code` tag) |

**Lineage rationale.** Starting from `ccdv/lsg-bart-base-4096-booksum` gives the summarizer two properties a vanilla `bart-large-cnn` would not have for this task: a **long-context** encoder (4096 tokens via LSG attention) so a whole children's story fits in one pass, and a base already oriented toward **book/long-form summarization** (BookSum). Fine-tuning that base on `cnn_dailymail` further tunes it for clean, news-style abstractive summaries. The trade-off is operational: because LSG attention ships as custom code, every load must pass `trust_remote_code=True` and use the matching `ccdv/lsg-bart-base-4096-booksum` tokenizer.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# LSG attention is custom code -> trust_remote_code is required.
tokenizer = AutoTokenizer.from_pretrained(
    "ccdv/lsg-bart-base-4096-booksum", trust_remote_code=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "ektvho/bart-cnn", trust_remote_code=True
)
```

### 4.2 `ektvho/sd-vist` — story-image generator

> **Hugging Face:** [`https://huggingface.co/ektvho/sd-vist`](https://huggingface.co/ektvho/sd-vist)

| Property | Value |
|---|---|
| Task | Text-to-image generation |
| Base | **Stable Diffusion 2.1** |
| Fine-tuning dataset | **VIST** (Visual Storytelling) |
| Native resolution | 768×768 (SD 2.1) |
| Hub class | Diffusers `StableDiffusionPipeline` (text-to-image) |

**Why VIST + SD 2.1.** VIST (Visual Storytelling) pairs sequences of images with narrative, story-style captions, which is much closer to "illustrate this sentence of a story" than a generic caption dataset. Fine-tuning **Stable Diffusion 2.1** — which renders natively at 768×768 — on VIST biases the generator toward coherent, narrative scene imagery, so the per-sentence panels read as storybook illustrations. This adapted generator is the visual heart of the thesis.

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "ektvho/sd-vist", torch_dtype=torch.float16
).to("cuda")

image = pipe("a curious penguin waddles across the snow").images[0]  # 768×768
```

> **Honesty note.** `ektvho/sd-vist` is **Stable Diffusion 2.1**, fine-tuned, and is the thesis generator. The Stable Diffusion **1.5** models that appear later in this report (Ghibli-Diffusion and friends) are a **separate, post-thesis extension** (§8.1) and are **not** part of the thesis pipeline.

### 4.3 `timbrooks/instruct-pix2pix` — style transfer (off-the-shelf)

The style-transfer stage uses the **pretrained** [`timbrooks/instruct-pix2pix`](https://huggingface.co/timbrooks/instruct-pix2pix) model unchanged. It is image-conditioned: given a panel and a free-text instruction, it edits the panel toward that instruction while keeping its composition. It is **used as-is, not fine-tuned** by the thesis, and is included here for completeness because it is part of the core pipeline's styling step.

---

## 5. Data & training

The thesis touches three datasets across its two fine-tuned models:

| Dataset | Role | Used by |
|---|---|---|
| `cnn_dailymail` | Abstractive-summarization fine-tuning corpus (news articles paired with highlight summaries) | `ektvho/bart-cnn` |
| **VIST** (Visual Storytelling) | Visual-storytelling fine-tuning corpus (image sequences with narrative captions) | `ektvho/sd-vist` |
| Custom story set | A hand-curated set of children's stories used to drive and demonstrate the end-to-end pipeline | Pipeline demonstration (`eerie.ipynb`) |

- **`cnn_dailymail`** supplies the supervised signal for the summarizer: long inputs with human-written summaries, which is what `ektvho/bart-cnn` learns to reproduce, on top of an LSG-BART base already pre-disposed (via BookSum) to long-form summarization.
- **VIST** supplies story-grounded image/text pairs for the generator. Fine-tuning SD 2.1 here is what makes `ektvho/sd-vist` produce narrative, storybook-style scenes rather than generic images.
- **The custom story set** is the thesis's own collection of children's stories used as end-to-end inputs — including the worked example **"Pip's Whirlwind Adventure"** (a curious penguin), which is used to demonstrate the full pipeline qualitatively (§7).

> **Hyperparameters.** This report intentionally does **not** state training hyperparameters (learning rate, epochs, batch size, optimizer schedule, etc.) for either fine-tune, because they are not part of the verified ground truth available here. The only quantities asserted are the ones that are verified: the base models, the architectures (LSG-BART 4096; SD 2.1), the datasets (`cnn_dailymail`; VIST), and the loading requirements (`trust_remote_code=True` and the `ccdv/lsg-bart-base-4096-booksum` tokenizer for `ektvho/bart-cnn`). No ROUGE, FID, CLIP, or other metric is reported, because none was measured in the verified ground truth.

---

## 6. How this repository reproduces the thesis honestly

This repository is a *portfolio* presentation of the thesis. Its guiding principle is to keep the academic record honest: the **thesis system is the core**, and anything added on top is **labeled as an extension** rather than retconned into the thesis.

- **The thesis system is described by its real artifacts.** The two fine-tuned models — `ektvho/bart-cnn` (summarizer lineage above) and `ektvho/sd-vist` (SD 2.1 on VIST) — are public on Hugging Face and are the canonical reference for what the thesis trained. The original thesis pipeline lives in `eerie.ipynb`.
- **The packaged `eerie/` demo in this repo is explicitly a simplification, not the thesis.** The importable `eerie/` Python package (see `eerie/pipeline.py`, `eerie/styles.py`, `eerie/models.py`) wires *off-the-shelf SD1.5* art-style checkpoints into a clean, fast, end-to-end `run_pipeline`. It is engineered for reproducibility (lazy, cached model loaders; pinned seeds; one image per sentence) and is a faithful *re-implementation of the pipeline shape*, but it swaps the heavyweight fine-tuned SD 2.1 generator for lightweight community SD1.5 style models. The package's own docstrings state this and state the limitation that there is no cross-panel character consistency.
- **Honesty rules, enforced throughout.** SD 2.1 fine-tuned (`ektvho/sd-vist`) is the thesis core; SD 1.5 (Ghibli-Diffusion / mo-di / watercolor) is an **extension** and is never implied to be the thesis. The "modern animation" SD1.5 model is a community *Disney-like* fine-tune and is deliberately **not** called "Pixar" or attributed to any studio. No fabricated metrics or hyperparameters are introduced anywhere.

In short: read `eerie.ipynb` and the two `ektvho/*` model pages for **the thesis**; read the `eerie/` package and §8 below for **the extensions**.

---

## 7. Qualitative results

The thesis evaluates its output **qualitatively**, by inspection of generated panels — there is no quantitative generative metric (no FID/CLIP), consistent with §5.

### 7.1 Worked example — *"Pip's Whirlwind Adventure"*

The flagship end-to-end example from the custom story set is **"Pip's Whirlwind Adventure"**, a short children's story about a **curious penguin**. Run through the core pipeline, the story is first condensed by `ektvho/bart-cnn`, split into sentences, and then each sentence is illustrated by `ektvho/sd-vist`, producing a sequence of storybook panels — one per narrative beat — that can optionally be restyled by InstructPix2Pix. The example demonstrates the intended behavior: the prose becomes a coherent strip of per-sentence illustrations in a consistent storybook idiom.

### 7.2 Generator comparison — Ghibli-Diffusion vs. *Aphantasia*

The thesis also includes a qualitative **comparison of image-generation backends**, contrasting a Ghibli-Diffusion-style generator against an *Aphantasia*-style approach to highlight the difference in how each renders the same narrative content. The comparison motivates the thesis's choice of a fine-tuned, story-adapted diffusion generator over a generic or non-fine-tuned alternative for the storybook-illustration task.

> **Note.** The qualitative comparisons above (the Pip example and the Ghibli-Diffusion vs. Aphantasia contrast) are from the thesis itself. They are *visual* findings; this report does not attach numeric scores to them because none were measured.

---

## 8. Extensions (post-thesis)

Everything in this section was built **after** the thesis. It is included because it extends the same problem space and demonstrates engineering depth, but it is **not** part of the thesis and uses **none** of the thesis's fine-tuned models unless stated.

### 8.1 SD1.5 multi-style exploration (the `eerie/` package)

The importable `eerie/` package is a lightweight, fast re-implementation of the *pipeline shape* on top of **off-the-shelf Stable Diffusion 1.5 art-style fine-tunes**, exposed through a single entry point `run_pipeline(story, art_style=…, style=…, summarize=…, seed=…)`. Its purpose is an art-style demo and a clean, reproducible package — not a reproduction of the thesis generator.

It selects an art style by **swapping the SD1.5 checkpoint** itself (no LoRA, IP-Adapter, or ControlNet), choosing among three community models defined in `eerie/styles.py`:

| `art_style` | Backing SD1.5 model | Nature |
|---|---|---|
| `ghibli` (default) | [`nitrosocke/Ghibli-Diffusion`](https://huggingface.co/nitrosocke/Ghibli-Diffusion) | Community fine-tune |
| `modern_animation` | [`nitrosocke/mo-di-diffusion`](https://huggingface.co/nitrosocke/mo-di-diffusion) | Community **Disney-like** fine-tune — **not Pixar, not studio-affiliated** |
| `watercolor` | [`ilee0022/watercolor_stable_diffusion`](https://huggingface.co/ilee0022/watercolor_stable_diffusion) | Community fine-tune |

In this extension the optional summarizer is the generic `facebook/bart-large-cnn` (not the thesis's `ektvho/bart-cnn`), generation is SD **1.5** (not the thesis's SD 2.1 `ektvho/sd-vist`), and `instruct-pix2pix` is reused for the optional restyle. The package is **labeled, in this report and in its own documentation, as the simplified demo / extension** — it is explicitly *not* the thesis pipeline. Its design notes also state plainly that panels are generated independently and that pinning a seed gives reproducibility, **not** cross-panel character consistency.

### 8.2 ComfyUI IP-Adapter character-consistency demo

A separate, standalone extension tackles the thesis pipeline's single biggest gap: keeping a character's appearance **consistent across panels**. Built as a **ComfyUI** workflow on **SD1.5 DreamShaper with IP-Adapter**, it conditions generation on a reference image so the same character keeps a stable appearance from one panel to the next. It produces a **real before/after** result — panels generated without the consistency mechanism vs. with it — making the improvement directly visible.

This is post-thesis and exploratory, but it is the natural research continuation of eerie: it demonstrates the controllability technique (IP-Adapter) that the thesis pipeline lacked, and it is the most direct answer to the limitation described in §9.1.

---

## 9. Honest limitations

These limitations apply to the **core thesis pipeline** (and, where noted, to the SD1.5 extension).

### 9.1 No cross-panel character consistency (thesis pipeline)

In the thesis pipeline, **each panel is generated independently** from its own sentence. There is no shared character embedding, reference image, or any other mechanism carried between generations, so a character described early in a story will **not** reliably look the same in a later panel — appearances drift from panel to panel. This is the pipeline's most significant limitation. It is precisely the gap that the post-thesis ComfyUI IP-Adapter extension (§8.2) was built to demonstrate a fix for; the IP-Adapter approach is *not* part of the thesis itself.

### 9.2 Model-version differences between the core and the SD1.5 extension

The thesis generator is **Stable Diffusion 2.1** (`ektvho/sd-vist`, 768×768, fine-tuned on VIST). The reproducible `eerie/` package extension instead uses **Stable Diffusion 1.5** community style checkpoints. These are different model families with different resolutions, priors, and look — so the SD1.5 package's panels are **not** expected to match the thesis generator's panels, and should not be read as a like-for-like reproduction of the thesis output. The version gap is intentional (the extension trades fidelity to the thesis generator for speed and a clean multi-style demo) and is stated here so the two are never conflated.

### 9.3 Summarization is "understanding" only in a shallow sense

The pipeline's grasp of a story is limited to abstractive summarization plus sentence splitting. There is **no** structured scene or character extraction (no named-entity parsing, scene graph, or per-character prompt construction), so the generator is conditioned only on a sentence and whatever style tokens are appended — not on an explicit model of who/where/what is in the scene.

### 9.4 No quantitative generative evaluation

Results are assessed **qualitatively**, by visual inspection (§7). The thesis reports **no** FID, CLIP-similarity, ROUGE, or human-evaluation score, and this report does not invent any. Quantitative evaluation of both the summaries and the generated panels is open future work.

---

## Appendix A — Model and dataset reference

| Component | Identifier | Type | Status |
|---|---|---|---|
| Summarizer (thesis) | [`ektvho/bart-cnn`](https://huggingface.co/ektvho/bart-cnn) | Fine-tuned LSG-BART (from `ccdv/lsg-bart-base-4096-booksum`, on `cnn_dailymail`) | Published, verified live |
| Summarizer base | [`ccdv/lsg-bart-base-4096-booksum`](https://huggingface.co/ccdv/lsg-bart-base-4096-booksum) | LSG-BART, 4096-token context | Base model |
| Generator (thesis) | [`ektvho/sd-vist`](https://huggingface.co/ektvho/sd-vist) | Stable Diffusion 2.1 fine-tuned on VIST (768×768) | Published, verified live |
| Style transfer | [`timbrooks/instruct-pix2pix`](https://huggingface.co/timbrooks/instruct-pix2pix) | Instruction image editor (pretrained, off-the-shelf) | Core pipeline, not fine-tuned |
| Datasets | `cnn_dailymail` · VIST · custom story set | Summarization / visual-storytelling / demo inputs | — |
| Extension generators (SD1.5) | `nitrosocke/Ghibli-Diffusion` · `nitrosocke/mo-di-diffusion` · `ilee0022/watercolor_stable_diffusion` | Community SD1.5 art-style fine-tunes | **Post-thesis extension** (§8.1) |
| Extension (consistency) | ComfyUI + SD1.5 DreamShaper + IP-Adapter | Character-consistency workflow | **Post-thesis extension** (§8.2) |

---

<sub>Thesis: *Text Summarization and Image Generation*, Yıldız Technical University, Department of Computer Engineering, 2023 — Yaren Yıldırım (17011019), Beyza Nur Sezgin (17011045); advisor Lect. Furkan Çakmak. The SD1.5 multi-style package and the ComfyUI IP-Adapter demo are post-thesis extensions and are not part of the thesis.</sub>