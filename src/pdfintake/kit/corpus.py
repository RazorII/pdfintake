"""The kit's public corpus: texts pinned by revision and hash, and the fonts the bench pages use.

``packs/<id>/corpus.json`` (DESIGN §3.6) lists what a pack's bench reads:

* ``corpus_id`` (``public-v1`` unless the contributor names another);
* ``texts[]``: ``source``, ``lang``, ``title``, ``revid``, ``url``, ``sha256``, ``licence``;
* ``fonts[]``: ``family``, ``url`` (a notofonts file at a pinned commit or release), ``sha256``,
  ``bytes`` and ``licence`` (``OFL-1.1``); the bench renders with at least two per script;
* ``pages``: how many ``prose``, ``form``, ``microprint`` and ``mixed_script`` pages to render.

The texts are fetched at bench time by revision id and never committed, so the MIT repo doesn't
redistribute CC BY-SA text. A Wikipedia text is pinned through Wikidata: the item's article in the
pack's language, its current revision id, and the SHA-256 of that revision's raw wikitext. A
revision never changes, so a different hash later is drift (``IntegrityError``), never an update.

Allowed licences (``LICENCES``): public domain, CC0, CC BY, CC BY-SA, and government text the law
excludes from copyright. Anything else is a stop-and-ask point for the contributor (add-a-language.md).

A ``url`` text must come from an allowlisted host (``TEXT_HOST_RES``: Wikipedia, Wikisource,
Project Gutenberg, EUR-Lex), checked when it is pinned and every time ``corpus.json`` loads, so
``kit verify`` refuses any other host before CI fetches anything.

Every download goes through ``_http_get`` (https only, a size cap, no redirect off https), which the
tests replace; nothing here runs during ``pdfintake run``.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import IntegrityError, PackError, UsageError

CORPUS_FILE = "corpus.json"
DEFAULT_CORPUS_ID = "public-v1"
PAGE_KINDS: tuple[str, ...] = ("prose", "form", "microprint", "mixed_script")
DEFAULT_PAGES: Mapping[str, int] = {"prose": 4, "form": 2, "microprint": 1, "mixed_script": 1}
MAX_PAGES_PER_KIND = 50
LICENCES: tuple[str, ...] = ("public-domain", "CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "government-public")
WIKIPEDIA_LICENCE = "CC-BY-SA-4.0"
FONT_LICENCE = "OFL-1.1"
MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_FONT_BYTES = 10 * 1024 * 1024
MAX_API_BYTES = 2 * 1024 * 1024
TIMEOUT_S = 60.0
USER_AGENT = "pdfintake-kit/1.0 (https://github.com/RazorII/pdfintake)"

_ID_RE = re.compile(r"[a-z0-9-]{1,40}")
_LANG_RE = re.compile(r"[a-z]{2,3}")
_QID_RE = re.compile(r"Q[1-9][0-9]{0,11}")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_WIKI_HOST_RE = re.compile(r"[a-z]{2,3}(-[a-z]+)?\.wikipedia\.org")
# The hosts a ``url`` text may come from. ``kit verify --rebench`` fetches corpus texts in CI on a
# contributor's pull request, so a text host is allowlisted like a model or font host: a Wikimedia
# wiki (public-domain texts on Wikisource), Project Gutenberg, and EUR-Lex for EU law, which is
# excluded from copyright. Another host is a code change (decided 2026-11-05).
TEXT_HOST_RES: tuple[re.Pattern[str], ...] = (
    _WIKI_HOST_RE,
    re.compile(r"[a-z]{2,3}(-[a-z]+)?\.wikisource\.org"),
    re.compile(r"(www\.)?gutenberg\.org"),
    re.compile(r"eur-lex\.europa\.eu"),
)
_SOURCES = ("wikipedia", "url")


@dataclass(frozen=True)
class CorpusText:
    source: str          # "wikipedia" or "url"
    lang: str
    title: str
    revid: int           # 0 for a text that isn't a wiki revision
    url: str
    sha256: str          # of the raw bytes fetched from ``url``
    licence: str


@dataclass(frozen=True)
class Corpus:
    corpus_id: str
    texts: tuple[CorpusText, ...]
    fonts: tuple[dict[str, Any], ...]
    pages: Mapping[str, int]


# --------------------------------------------------------------------------- the network (tests replace it)

def _http_get(url: str, *, cap: int) -> bytes:
    """GET ``url`` over https and return the body, refusing more than ``cap`` bytes."""
    import ssl
    import urllib.request

    if urllib.parse.urlsplit(url).scheme != "https":
        raise PackError("kit_download", f"only https downloads are allowed: {url}")
    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx.load_verify_locations(cafile=certifi.where())
    except ImportError:
        pass

    text_host = text_host_allowed(url)

    class _HttpsOnly(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
            if urllib.parse.urlsplit(newurl).scheme != "https":
                raise PackError("kit_download", f"{url} redirects off https")
            if text_host and not text_host_allowed(newurl):     # a corpus text stays on the allowlist
                raise PackError("kit_download", f"{url} redirects off the corpus text hosts")
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx), _HttpsOnly())
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(req, timeout=TIMEOUT_S) as resp:
            body = resp.read(cap + 1)
    except PackError:
        raise
    except Exception as exc:
        status = getattr(exc, "code", None) if isinstance(getattr(exc, "code", None), int) else None
        raise PackError("kit_download", f"couldn't download {url}: {type(exc).__name__}: {exc}", url=url,
                        status=status) from None
    if len(body) > cap:
        raise PackError("kit_download", f"{url} is larger than {cap} bytes", url=url)
    return body


def _get_json(url: str) -> Any:
    try:
        return json.loads(_http_get(url, cap=MAX_API_BYTES).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise PackError("kit_download", f"{url} didn't answer with JSON", url=url) from None


# --------------------------------------------------------------------------- pinning

def wikipedia_raw_url(lang: str, revid: int) -> str:
    return f"https://{lang}.wikipedia.org/w/index.php?oldid={int(revid)}&action=raw"


def pin_wikipedia(lang: str, items: Sequence[str]) -> list[CorpusText]:
    """Pin the ``lang`` Wikipedia article of each Wikidata item (``Q283``): its current revision id
    and the SHA-256 of that revision's raw wikitext. UsageError for a malformed id; PackError when an
    item has no article in that language or an answer can't be read."""
    if not _LANG_RE.fullmatch(lang or ""):
        raise UsageError("kit_usage", f"{lang!r} isn't a Wikipedia language code (two or three letters)")
    out: list[CorpusText] = []
    for item in items:
        if not _QID_RE.fullmatch(item or ""):
            raise UsageError("kit_usage", f"{item!r} isn't a Wikidata item id such as Q283")
        q = urllib.parse.urlencode({"action": "wbgetentities", "ids": item, "props": "sitelinks",
                                    "sitefilter": f"{lang}wiki", "format": "json"})
        data = _get_json(f"https://www.wikidata.org/w/api.php?{q}")
        try:
            title = data["entities"][item]["sitelinks"][f"{lang}wiki"]["title"]
        except (KeyError, TypeError):
            raise PackError("kit_no_article", f"Wikidata item {item} has no article on {lang}.wikipedia.org",
                            item=item, lang=lang) from None
        if not isinstance(title, str) or not title or len(title) > 250:
            raise PackError("kit_no_article", f"Wikidata item {item} gave an unusable title", item=item)
        q = urllib.parse.urlencode({"action": "query", "prop": "revisions", "titles": title, "rvprop": "ids",
                                    "format": "json", "formatversion": "2"})
        data = _get_json(f"https://{lang}.wikipedia.org/w/api.php?{q}")
        try:
            revid = int(data["query"]["pages"][0]["revisions"][0]["revid"])
        except (KeyError, IndexError, TypeError, ValueError):
            raise PackError("kit_no_article", f"{lang}.wikipedia.org has no revision for item {item}",
                            item=item, lang=lang) from None
        url = wikipedia_raw_url(lang, revid)
        raw = _http_get(url, cap=MAX_TEXT_BYTES)
        out.append(CorpusText("wikipedia", lang, title, revid, url, hashlib.sha256(raw).hexdigest(), WIKIPEDIA_LICENCE))
    return out


def text_host_allowed(url: str) -> bool:
    """Whether ``url`` is https on a host ``TEXT_HOST_RES`` allows (no user info, no port)."""
    try:
        u = urllib.parse.urlsplit(url)
        port = u.port
    except ValueError:
        return False
    if u.scheme != "https" or u.username is not None or u.password is not None or port not in (None, 443):
        return False
    host = (u.hostname or "").lower()
    return any(r.fullmatch(host) for r in TEXT_HOST_RES)


def pin_url(url: str, *, lang: str, title: str, licence: str) -> CorpusText:
    """Pin a plain-text file under a licence ``LICENCES`` allows, from a host ``TEXT_HOST_RES``
    allows (UsageError for anything else)."""
    if licence not in LICENCES:
        raise UsageError("kit_licence", f"licence {licence!r} isn't one the kit allows ({', '.join(LICENCES)}); "
                                        "a text under another licence can't be used")
    if urllib.parse.urlsplit(url).scheme != "https":
        raise UsageError("kit_usage", "a corpus URL must use https")
    if not text_host_allowed(url):
        raise UsageError("kit_text_host", f"{url} isn't on a host the kit fetches corpus texts from "
                                          "(Wikipedia, Wikisource, Project Gutenberg, EUR-Lex)")
    raw = _http_get(url, cap=MAX_TEXT_BYTES)
    return CorpusText("url", lang, title, 0, url, hashlib.sha256(raw).hexdigest(), licence)


def pin_font(url: str, *, family: str | None = None) -> dict[str, Any]:
    """A corpus font entry for ``url``: a notofonts file at a pinned commit or release (the manifest's
    font allowlist), with its size and SHA-256."""
    from ..langpacks.manifest import source_problems

    problems = source_problems("font", url)
    if problems:
        raise UsageError("kit_font_source", f"{url} isn't an allowed font source: {'; '.join(problems)}")
    raw = _http_get(url, cap=MAX_FONT_BYTES)
    name = urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]
    fam = family or re.sub(r"[-_](Regular|Bold|Italic)?\.(ttf|otf)$", "", name, flags=re.I) or name
    return {"family": fam, "url": url, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "licence": FONT_LICENCE}


# --------------------------------------------------------------------------- fetching

def _cache_file(cache: Path | None, sha256: str, suffix: str) -> Path | None:
    return None if cache is None else Path(cache) / f"{sha256}{suffix}"


def _fetch_pinned(url: str, sha256: str, cap: int, cache: Path | None, suffix: str, what: str) -> bytes:
    path = _cache_file(cache, sha256, suffix)
    if path is not None and path.is_file():
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() == sha256:
            return raw
    raw = _http_get(url, cap=cap)
    got = hashlib.sha256(raw).hexdigest()
    if got != sha256:
        raise IntegrityError("corpus_drift", f"{what} at {url} has SHA-256 {got}; corpus.json pins {sha256}. "
                                             "Pin it again with `pdfintake kit corpus` only if the change is expected.",
                             url=url, expected=sha256, actual=got)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)
    return raw


def fetch_text(text: CorpusText, *, cache: Path | None = None) -> str:
    """The text's plain words: downloaded (or read from ``cache``), checked against its SHA-256
    (IntegrityError on drift), then stripped of wiki markup for a Wikipedia revision."""
    raw = _fetch_pinned(text.url, text.sha256, MAX_TEXT_BYTES, cache, ".txt", f"{text.title!r}")
    decoded = raw.decode("utf-8", errors="replace")
    plain = wikitext_to_plain(decoded, text.lang) if text.source == "wikipedia" else decoded
    return unicodedata.normalize("NFC", plain)


def fetch_font(font: Mapping[str, Any], dest: Path) -> Path:
    """The corpus font's file in ``dest`` (downloaded when missing or wrong), hash-checked."""
    name = urllib.parse.urlsplit(str(font["url"])).path.rsplit("/", 1)[-1]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,80}\.(ttf|otf)", name):
        raise PackError("kit_font", f"the corpus font {font['url']} isn't a .ttf or .otf file")
    path = Path(dest) / name
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == font["sha256"]:
        return path
    raw = _fetch_pinned(str(font["url"]), str(font["sha256"]), MAX_FONT_BYTES, None, "", f"font {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- wikitext to plain text

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF = re.compile(r"<ref\b[^>/]*/>|<ref\b[^>]*>.*?</ref\s*>", re.S | re.I)
_BLOCK_TAGS = re.compile(r"<(math|code|pre|gallery|timeline|score|syntaxhighlight|source|table)\b[^>]*>.*?</\1\s*>",
                         re.S | re.I)
_TAG = re.compile(r"</?[A-Za-z][^<>]*>")
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_TABLE = re.compile(r"\{\|[^{}]*?\|\}", re.S)
_LINK = re.compile(r"\[\[([^\[\]|]*)(?:\|([^\[\]]*))?\]\]")
_EXT_LINK = re.compile(r"\[(?:https?:)?//[^\s\]]+(?:\s+([^\]]*))?\]")
_HEADING = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.M)
_LIST_MARK = re.compile(r"^[*#:;]+\s*", re.M)
_ENTITIES = {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&ndash;": "–", "&mdash;": "—"}

# Templates that are part of the sentence: the space or dash between a number and its unit, numbers,
# units, dates and centuries, no-break wrappers, chemical formulas and inline quotations. Each is
# rendered from its own arguments, close to what the wiki shows (checked against es, fr, en, hu, it
# and pt Wikipedia's parser on 2026-11-06): numbers as written, without the wiki's locale grouping,
# and a conversion template's value and unit without the converted value. A unit's power is written
# after the unit as plain digits, the way the wiki's own text reads with its superscript tags removed
# ("m3", "mol−1", like "km2" in a conversion code or m/s<sup>2</sup> in prose). Every other template
# is dropped: references, maintenance tags, infoboxes, language tags, and anything that needs
# typography plain text can't hold: a power of ten (1e30, e9km3, e=, which flattened would read as a
# different number such as "1030") and stacked fractions. Names match case-folded, "_" read as a space.
_EQUALS = "\ue000"      # {{=}} inside another template's argument; "=" itself would split it as name=value
_FIXED_TEMPLATES: Mapping[str, str] = {
    "esd": " ", "nbsp": " ", "sp": " ", "space": " ",
    "snd": " – ", "spaced ndash": " – ", "spaced en dash": " – ",
    "ndash": "–", "endash": "–", "en dash": "–", "mdash": "—", "emdash": "—", "em dash": "—",
    "'": "'", "okina": "ʻ", "etc.": ", etc.", ", etc.": ", etc.", "av jc": "av. J.-C.", "co2": "CO2",
    "=": _EQUALS,
}
_JOINED_TEMPLATES = frozenset({
    "nowrap", "nobr", "lnobr", "nobreak", "tutto attaccato",            # no-break wrappers
    "formatnum", "fmtn", "tagol", "nombre",                              # numbers
    "tmp",                                                               # a temperature and its unit
    "date", "date-",                                                     # dates
})
_FORMULA_TEMPLATES = frozenset({"chem", "chem2", "fchim", "formule chimique"})
_CONVERT_TEMPLATES = frozenset({"convert", "cvt", "converti"})
_UNIT_TEMPLATES = frozenset({"unité", "unidad"})
_UNIT_CODES: Mapping[str, str] = {"C": "°C", "F": "°F"}
_RANGE_WORDS = frozenset({"-", "–", "to", "and", "or", "by", "x", "×", "+", "to(-)", "and(-)"})
_VAL_RANGE_WORDS = frozenset({"-", "–", "to", "and", "or", "by", "x", "×"})
_TIGHT_UNITS = frozenset({"%", "°", "′", "″", "‰"})  # written against the number: "5%", "0.527°" (but "100 °C")
# The words each wiki's {{val}} (and it.wikipedia's {{M}}) shows between a range's ends, where they differ from the
# argument; checked on en, it, pt and vi Wikipedia on 2026-11-06. Any other wiki shows the argument.
_VAL_WORDS: Mapping[str, Mapping[str, str]] = {
    "en": {"-": "–", "x": "×"},
    "it": {"and": "e", "to": "fino a", "or": "o", "by": "su", "x": "×"},
    "pt": {"-": "–", "x": "×", "to": "a", "and": "e"},
    "vi": {"-": "–", "x": "×", "to": "đến"},
}
_UNITE_RANGE_KEYS = ("–", "-", "à", "et", "ou", "±", "×")
# A power of ten: "3,83e26", "1.5E9", a lone "e30", or an e-prefixed unit such as "e9km3".
_EXPONENT = re.compile(r"(?:\d|^|\s)e[-+−]?\d", re.I)
_POWER = re.compile(r"[-+−]?\d+")
_SIGNED_POWER = re.compile(r"[-−]\d+")
_UNIT_POWER = re.compile(r"([^\W\d_]+)-(\d+)")   # "mol-1" in a one-argument {{unité}}


def _template_args(inner: str) -> tuple[str, list[str], dict[str, str]]:
    """A template's name (case-folded) and its positional and named arguments. ``|`` inside a
    ``[[link|label]]`` doesn't split; a ``{{formatnum:N}}``-style name carries its first argument."""
    parts: list[str] = []
    cur: list[str] = []
    depth = i = 0
    while i < len(inner):
        two = inner[i:i + 2]
        if two in ("[[", "]]"):
            depth += 1 if two == "[[" else -1 if depth else 0
            cur.append(two)
            i += 2
            continue
        if inner[i] == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(inner[i])
        i += 1
    parts.append("".join(cur))
    head = parts[0].strip()
    positional: list[str] = []
    if ":" in head and head.split(":", 1)[0].strip().casefold() in ("formatnum",):
        head, first = head.split(":", 1)
        positional.append(first.strip())
    name = " ".join(head.replace("_", " ").split()).casefold()
    named: dict[str, str] = {}
    for p in parts[1:]:
        key, eq, value = p.partition("=")
        key = key.strip()
        if eq and key and not any(ch in key for ch in "[]<>{}"):
            if key.isdigit() and 0 < int(key) <= 20:
                while len(positional) < int(key):
                    positional.append("")
                positional[int(key) - 1] = value.strip()
            else:
                named[key.casefold()] = value.strip()
        else:
            positional.append(p.strip())
    return name, positional, named


def _power(p: str, sign: str) -> str:
    return p.replace("-", sign).replace("−", sign) if sign else p


def _render_unit(name: str, pos: list[str], named: dict[str, str]) -> str:
    """{{unité|value|unit|power|unit|power…}} (fr) and {{unidad|…}} (es): the value, a range end held in
    a named argument ({{unité|0|–=11|km}}: "0–11 km"), then each unit with its power written after it
    ({{unité|60|m|3}}: "60 m3"). fr also reads a whole quantity in one argument ({{unité|150000 kg m -3}}:
    "150000 kg m−3"). Nothing at all when a power of ten is involved."""
    if "e" in named or any(_EXPONENT.search(p) for p in pos):
        return ""
    sign = "−" if name == "unité" else ""
    value = pos[0] if pos else ""
    for key in _UNITE_RANGE_KEYS:
        if named.get(key):
            sep = "–" if key in ("–", "-") else f" {key} "
            value = f"{value}{sep}{named[key]}"
            break
    units: list[str] = []
    for i in range(1, len(pos), 2):
        unit = pos[i]
        power = pos[i + 1] if i + 1 < len(pos) else ""
        if power and _POWER.fullmatch(power):
            unit += _power(power, sign)
        elif power:
            unit = f"{unit} {power}".strip()
        if unit:
            units.append(unit)
    text = " ".join(p for p in (value, *units) if p)
    if name == "unité":                              # the one-argument form: "kg m -3", "J mol-1"
        out: list[str] = []
        for tok in text.split():
            if out and _SIGNED_POWER.fullmatch(tok) and out[-1] not in _UNITE_RANGE_KEYS \
                    and any(ch.isalpha() for ch in out[-1]) and not any(ch.isdigit() for ch in out[-1]):
                out[-1] += _power(tok, sign)
            else:
                out.append(_UNIT_POWER.sub(lambda u: f"{u.group(1)}−{u.group(2)}", tok)
                           if any(ch.isalpha() for ch in tok) else tok)
        text = " ".join(out)
    return text


def _render_val(filled: list[str], named: dict[str, str], lang: str) -> str:
    """{{val}} (en, it, pt, vi) and {{M}} (it): a value, or a range of two with the wiki's word between
    them, and its unit in u= or ul=. "%" and "°" sit against each number ("5%–30%"), any other unit
    once, after a space ("1–2 km"); en repeats it around "×" ("1 m × 2 m")."""
    if not filled or "e" in named or any(_EXPONENT.search(p) for p in filled):
        return ""
    unit = named.get("u") or named.get("ul") or ""
    tight = unit in _TIGHT_UNITS

    def num(v: str) -> str:
        return v + unit if tight else v

    if len(filled) >= 3 and filled[1].casefold() in _VAL_RANGE_WORDS:
        a, b = filled[0], filled[2]
        word = _VAL_WORDS.get(lang, {}).get(filled[1].casefold(), filled[1])
        if word in ("-", "–"):
            body = f"{num(a)}{word}{num(b)}"
        elif word == "×" and lang == "en" and unit and not tight:
            return f"{a} {unit} × {b} {unit}"
        else:
            body = f"{num(a)} {word} {num(b)}"
    else:
        body = num(filled[0])
    return body if tight or not unit else f"{body} {unit}"


def _render_template(m: re.Match[str], lang: str = "") -> str:
    """The text a template stands for in the sentence (``_FIXED_TEMPLATES`` and the sets beside it),
    or "" for every other template. ``lang`` is the wiki's language code, for the words {{val}} shows."""
    name, pos, named = _template_args(m.group(0)[2:-2])
    if name in _FIXED_TEMPLATES:
        return _FIXED_TEMPLATES[name]
    filled = [p for p in pos if p]
    if name in _JOINED_TEMPLATES:
        return " ".join(filled)
    if name in _FORMULA_TEMPLATES:
        return "".join(filled)
    if name in _UNIT_TEMPLATES:
        return _render_unit(name, pos, named)
    if name in _CONVERT_TEMPLATES:                   # value [range word value] unit; the conversion isn't shown
        if not pos or not pos[0]:
            return ""
        n = 4 if len(filled) >= 4 and filled[1].casefold() in _RANGE_WORDS else 2
        vals = filled[:n]
        if any(_EXPONENT.search(v) for v in vals):  # 3,83e26 J, e9km3: a power of ten
            return ""
        vals[-1] = _UNIT_CODES.get(vals[-1], vals[-1])
        return " ".join(vals)
    if name == "szám":                               # hu: value, unit, and a third word ("170 millió tonnát")
        vals = filled[:3]
        if any(_EXPONENT.search(v) for v in vals):
            return ""
        if len(vals) > 1:
            vals[1] = _UNIT_CODES.get(vals[1], vals[1])
        return " ".join(vals)
    if name in ("val", "m"):
        return _render_val(filled, named, "it" if name == "m" else lang)
    if not filled:
        return ""
    first = filled[0]
    if name == "circa":
        return f"c. {first}" + (f" – c. {filled[1]}" if len(filled) > 1 else "")
    if name == "siglo":                              # {{siglo|XXI||s}}: "siglo XXI"; a 2nd "a": before Christ
        flag = pos[2] if len(pos) > 2 else ""       # "s": "siglo XX", "S": "Siglo XX"
        word = f"{flag}iglo " if flag.casefold() == "s" else ""
        return word + first + (" a. C." if len(pos) > 1 and pos[1] == "a" else "")
    if name in ("séc", "-séc"):
        return f"século {first}" + (" a.C." if name == "-séc" else "")
    if name in ("s-", "-s-"):
        return f"{first}e siècle" + (" av. J.-C." if name == "-s-" else "")
    if name == "ac":
        return f"{first} a.C."
    if name == "citation":
        return f"« {first} »"
    if name == "incise":
        return f"— {first} —"
    return ""


def _strip_nested(pattern: re.Pattern[str], text: str, repl: str | Callable[[re.Match[str]], str] = "") -> str:
    for _ in range(50):
        new = pattern.sub(repl, text)
        if new == text:
            return new
        text = new
    return text


def wikitext_to_plain(raw: str, lang: str = "") -> str:
    """Wiki markup removed, deterministically: comments, references, tables, files and categories go;
    templates go too, innermost first, except those that are part of the sentence (a number's unit,
    a date, a no-break wrapper: ``_render_template``), which leave their text; links keep their label
    (innermost first, so a file caption's links go with the file); headings and list items keep their
    words. One paragraph per line, spaces collapsed. ``lang`` is the wiki's language code."""
    t = raw.replace("\r\n", "\n")
    t = _COMMENT.sub("", t)
    t = _REF.sub("", t)
    t = _BLOCK_TAGS.sub("", t)
    t = _strip_nested(_TEMPLATE, t, lambda m: _render_template(m, lang))
    t = _strip_nested(_TABLE, t)

    def link(m: re.Match[str]) -> str:
        target, label = m.group(1), m.group(2)
        bare = target.lstrip(":")
        if ":" in bare and not target.startswith(":"):
            return ""                                    # File:, Category:, interwiki
        return label if label is not None else bare

    for _ in range(50):
        new = _LINK.sub(link, t)
        if new == t:
            break
        t = new
    t = _EXT_LINK.sub(lambda m: m.group(1) or "", t)
    t = _TAG.sub("", t)
    t = t.replace("'''", "").replace("''", "")
    t = _HEADING.sub(lambda m: m.group(1), t)
    t = _LIST_MARK.sub("", t)
    for k, v in _ENTITIES.items():
        t = t.replace(k, v)
    lines = []
    for line in t.split("\n"):
        line = " ".join(line.split())
        if line and not line.startswith(("|", "!", "{", "}")) and any(ch.isalpha() for ch in line):
            lines.append(line)
    return "\n".join(lines).replace(_EQUALS, "=")


# --------------------------------------------------------------------------- corpus.json

def _problems(data: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(data, dict):
        return ["$: must be an object"]
    extra = sorted(set(data) - {"corpus_id", "texts", "fonts", "pages"})
    if extra:
        out.append("$: unknown fields " + ", ".join(extra))
    if not (isinstance(data.get("corpus_id"), str) and _ID_RE.fullmatch(data["corpus_id"])):
        out.append("$.corpus_id: lower-case letters, digits and hyphens, at most 40")
    texts = data.get("texts")
    if not isinstance(texts, list):
        out.append("$.texts: must be a list")
        texts = []
    keys = {"source", "lang", "title", "revid", "url", "sha256", "licence"}
    for i, t in enumerate(texts):
        p = f"$.texts[{i}]"
        if not isinstance(t, dict) or set(t) != keys:
            out.append(f"{p}: needs exactly {', '.join(sorted(keys))}")
            continue
        if t["source"] not in _SOURCES:
            out.append(f"{p}.source: one of {', '.join(_SOURCES)}")
        if not (isinstance(t["lang"], str) and _LANG_RE.fullmatch(t["lang"])):
            out.append(f"{p}.lang: a two- or three-letter code")
        if not (isinstance(t["title"], str) and 0 < len(t["title"]) <= 250):
            out.append(f"{p}.title: a non-empty string")
        if isinstance(t["revid"], bool) or not isinstance(t["revid"], int) or t["revid"] < 0:
            out.append(f"{p}.revid: a whole number")
        if not (isinstance(t["sha256"], str) and _SHA_RE.fullmatch(t["sha256"])):
            out.append(f"{p}.sha256: 64 hex characters")
        if t["licence"] not in LICENCES:
            out.append(f"{p}.licence: one of {', '.join(LICENCES)}")
        url = t["url"]
        if not isinstance(url, str) or urllib.parse.urlsplit(url).scheme != "https":
            out.append(f"{p}.url: an https URL")
        elif t["source"] == "wikipedia":
            u = urllib.parse.urlsplit(url)
            if (not _WIKI_HOST_RE.fullmatch(u.hostname or "") or isinstance(t["revid"], bool)
                    or not isinstance(t["revid"], int) or url != wikipedia_raw_url(str(t["lang"]), t["revid"])):
                out.append(f"{p}.url: a Wikipedia text is the raw revision URL "
                           f"(https://<lang>.wikipedia.org/w/index.php?oldid=<revid>&action=raw)")
            if t["licence"] != WIKIPEDIA_LICENCE:
                out.append(f"{p}.licence: Wikipedia text is {WIKIPEDIA_LICENCE}")
        elif not text_host_allowed(url):
            out.append(f"{p}.url: not a host the kit fetches corpus texts from "
                       "(Wikipedia, Wikisource, Project Gutenberg, EUR-Lex)")
    fonts = data.get("fonts")
    if not isinstance(fonts, list):
        out.append("$.fonts: must be a list")
        fonts = []
    from ..langpacks.manifest import source_problems

    fkeys = {"family", "url", "sha256", "bytes", "licence"}
    for i, f in enumerate(fonts):
        p = f"$.fonts[{i}]"
        if not isinstance(f, dict) or set(f) != fkeys:
            out.append(f"{p}: needs exactly {', '.join(sorted(fkeys))}")
            continue
        if not (isinstance(f["family"], str) and re.fullmatch(r"[A-Za-z0-9 _.+-]{1,60}", f["family"])):
            out.append(f"{p}.family: letters, digits, spaces and _.+- only")
        out += [f"{p}.url: {why}" for why in source_problems("font", f["url"])]
        if not (isinstance(f["sha256"], str) and _SHA_RE.fullmatch(f["sha256"])):
            out.append(f"{p}.sha256: 64 hex characters")
        if isinstance(f["bytes"], bool) or not isinstance(f["bytes"], int) or not 0 < f["bytes"] <= MAX_FONT_BYTES:
            out.append(f"{p}.bytes: a size from 1 to {MAX_FONT_BYTES}")
        if f["licence"] != FONT_LICENCE:
            out.append(f"{p}.licence: {FONT_LICENCE}")
    pages = data.get("pages")
    if not isinstance(pages, dict) or set(pages) - set(PAGE_KINDS):
        out.append(f"$.pages: an object with counts for {', '.join(PAGE_KINDS)}")
    else:
        for k, v in pages.items():
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= MAX_PAGES_PER_KIND:
                out.append(f"$.pages.{k}: a whole number from 0 to {MAX_PAGES_PER_KIND}")
    return out


def from_json(data: Any) -> Corpus:
    problems = _problems(data)
    if problems:
        raise PackError("invalid_corpus", f"corpus.json is invalid: {'; '.join(problems)}", problems=problems)
    texts = tuple(CorpusText(t["source"], t["lang"], t["title"], int(t["revid"]), t["url"], t["sha256"], t["licence"])
                  for t in data["texts"])
    pages = {k: int(data["pages"].get(k, 0)) for k in PAGE_KINDS}
    return Corpus(data["corpus_id"], texts, tuple(dict(f) for f in data["fonts"]), pages)


def to_json(c: Corpus) -> dict[str, Any]:
    return {"corpus_id": c.corpus_id,
            "texts": [{"source": t.source, "lang": t.lang, "title": t.title, "revid": t.revid, "url": t.url,
                       "sha256": t.sha256, "licence": t.licence} for t in c.texts],
            "fonts": [dict(f) for f in c.fonts],
            "pages": {k: int(c.pages.get(k, 0)) for k in PAGE_KINDS}}


def load(path: Path) -> Corpus:
    """``corpus.json``, strictly (PackError listing every problem)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PackError("corpus_missing", f"{path} is missing; start it with `pdfintake kit init`") from None
    except (OSError, ValueError) as exc:
        raise PackError("invalid_corpus", f"{path} isn't readable JSON: {exc}") from None
    return from_json(data)


def save(c: Corpus, path: Path) -> None:
    to_json(c)                                           # shape check by construction
    from_json(to_json(c))
    Path(path).write_text(json.dumps(to_json(c), indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                          newline="\n")


def empty(corpus_id: str = DEFAULT_CORPUS_ID) -> Corpus:
    return Corpus(corpus_id, (), (), dict(DEFAULT_PAGES))


def corpus_digest(c: Corpus) -> str:
    """The SHA-256 of the corpus's canonical JSON: what ``bench.corpus`` names as ``<id>@<digest>``."""
    payload = json.dumps(to_json(c), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def corpus_ref(c: Corpus) -> str:
    return f"{c.corpus_id}@{corpus_digest(c)}"
