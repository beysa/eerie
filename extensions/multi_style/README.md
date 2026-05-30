# multi_style — community SD1.5 art-style explorer (extension)

This is a **post-thesis style exploration, NOT the thesis models.** The thesis core
(`eerie.run_pipeline`) generates images with the candidate's fine-tuned
[`ektvho/sd-vist`](https://huggingface.co/ektvho/sd-vist) (Stable Diffusion 2.1,
fine-tuned on VIST) after summarizing with the fine-tuned
[`ektvho/bart-cnn`](https://huggingface.co/ektvho/bart-cnn); this extension instead
swaps **community Stable Diffusion 1.5** fine-tunes — Ghibli
(`nitrosocke/Ghibli-Diffusion`), modern animation / Disney-like
(`nitrosocke/mo-di-diffusion`), and watercolor
(`ilee0022/watercolor_stable_diffusion`) — to render the same story in different
art styles via `run_multi_style(story, art_style=...)`. These are community
checkpoints, not affiliated with any studio and never the thesis model.
