"""Greedy decoding for dots.mocr: chunked prefill, then one token at a time.

Ported from mlx-vlm 0.7.4 (``mlx_vlm/generate/ar.py`` ``generate_step`` on its temperature-0 path,
and ``generate/common.py`` for the stream and the wired-memory limit), MIT, Copyright (c) 2025
Prince Canuma. What decides the numbers is kept exactly: the prompt is prefilled in chunks of
2048 tokens with its last token left for the first step, each step takes the argmax of the
log-probabilities (logits minus their logsumexp, in the model's dtype, so ties break as mlx-vlm's
do), and the next token is computed while the current one is handed out. Sampling, penalties,
prompt caches, speculative decoding and KV quantisation aren't ported.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING

from ._mlx import mx

if TYPE_CHECKING:
    from .model import DotsOCR

PREFILL_STEP = 2048              # mlx-vlm's DEFAULT_PREFILL_STEP_SIZE; it changes the numbers
CACHE_EVAL_INTERVAL = 50         # evaluate the KV cache this often, as mlx-vlm does (memory only)
CLEAR_CACHE_INTERVAL = 256       # and free mlx's buffer cache this often

_stream = None
_stream_lock = threading.Lock()
_wired_lock = threading.Lock()
_wired_users = 0
_wired_old = 0


def generation_stream():
    """The thread-local stream decoding runs on, made on first use."""
    global _stream
    with _stream_lock:
        if _stream is None:
            _stream = mx.new_thread_local_stream(mx.default_device())
        return _stream


@contextlib.contextmanager
def wired_limit(model: DotsOCR):
    """Keep the model's memory wired while any read runs (mlx-vlm's ``wired_limit``), and put
    the old limit back after the last one."""
    global _wired_users, _wired_old
    if not mx.metal.is_available():
        yield
        return
    with _wired_lock:
        if _wired_users == 0:
            _wired_old = mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        _wired_users += 1
    try:
        yield
    finally:
        mx.synchronize(generation_stream())
        with _wired_lock:
            _wired_users -= 1
            if _wired_users == 0:
                mx.set_wired_limit(_wired_old)


def greedy_tokens(model: DotsOCR, input_ids, pixel_values, grid: tuple[int, int, int],
                  max_tokens: int) -> Iterator[int]:
    """Yield up to ``max_tokens`` token ids, greedily. Stopping at an end token is the caller's
    job: close the generator to stop early."""
    lm = model.language_model.model
    cache = model.language_model.make_cache()
    stream = generation_stream()

    def step(inputs_embeds=None, token=None):
        with mx.stream(stream):
            if inputs_embeds is None:
                inputs_embeds = lm.embed_tokens(token[None])
            logits = lm.logits(lm.hidden(inputs_embeds, cache))[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return mx.argmax(logprobs, axis=-1)

    with mx.stream(stream):
        embeds = model.embed(input_ids, pixel_values, grid)
        while embeds.shape[1] > 1:                       # prefill all but the last prompt token
            n = min(PREFILL_STEP, embeds.shape[1] - 1)
            lm.hidden(embeds[:, :n], cache)
            mx.eval([c.state for c in cache])
            embeds = embeds[:, n:]
            mx.clear_cache()
    y = step(embeds)
    mx.async_eval(y)

    n = 0
    while True:
        if n != max_tokens:
            next_y = step(token=y)
            mx.async_eval(next_y)
        if n == 0:
            mx.eval(y)
        if n == max_tokens:
            break
        if (n + 1) % CACHE_EVAL_INTERVAL == 0:
            mx.eval([c.state for c in cache])
        yield y.item()
        if n % CLEAR_CACHE_INTERVAL == 0:
            mx.clear_cache()
        y = next_y
        n += 1
