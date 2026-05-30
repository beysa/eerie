"""Abstractive summarization with the thesis's fine-tuned BART.

This is the thesis's "Text Summarization" stage, run with the candidate's own
fine-tuned model rather than an off-the-shelf summarizer:

* ``ektvho/bart-cnn`` — BART fine-tuned *from* ``ccdv/lsg-bart-base-4096-booksum``
  (an LSG-BART with a 4096-token long context) on CNN/DailyMail. The LSG base
  gives it a long input window, so a whole children's story can be summarized in
  one pass rather than being truncated to the 1024 tokens a vanilla BART allows.

Because the LSG tokenizer ships custom attention code, the model and tokenizer
are loaded as a pair (with ``trust_remote_code=True``) in
:func:`eerie.models.get_summarizer`, and this module drives encode -> generate ->
decode directly. The decoding parameters mirror the notebook, with one
behavioural change kept from the package's first version: beam width defaults to
``4`` (the notebook used ``15``, which stalls for ~30-60s per call for no
meaningful quality gain here).

In the thesis flow this runs first: the story is summarized, then the summary is
split into one scene per sentence.
"""

import torch

from eerie.models import get_summarizer


def summarize(
    text: str,
    *,
    max_length: int = 450,
    min_length: int = 125,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 7,
) -> str:
    """Condense ``text`` into a shorter abstractive summary with ``ektvho/bart-cnn``.

    The text is encoded with the LSG long-context tokenizer (so long stories are
    not truncated), generated with beam search, and decoded back to a string.

    Args:
        text: The story prose to summarize.
        max_length: Maximum length of the summary in tokens.
        min_length: Minimum length of the summary in tokens.
        num_beams: Beam-search width. Defaults to ``4`` (the notebook used
            ``15``, which is far slower for no meaningful quality gain here).
        no_repeat_ngram_size: Block repeating n-grams of this size.

    Returns:
        The summary text produced by the fine-tuned ``ektvho/bart-cnn``.
    """
    model, tokenizer = get_summarizer()
    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
