#!/usr/bin/env python3
"""
clean_for_tts.py

Preprocesses a story text file to make it read better by neural TTS engines
(like Chatterbox), which often mispronounce ALL-CAPS text (headers, titles,
chapter names) because the model rarely sees shouty-caps in normal training
data.

What it does:
  1. Detects lines that are "shouty" (mostly uppercase letters) — typically
     chapter headers, titles, or all-caps captions.
  2. Converts those lines to normal sentence case, so the TTS engine reads
     them the same way it reads regular prose (fixing pronunciation issues
     like "ITHACA" -> "ith-AH-kah" instead of "ITH-uh-kuh").
  3. Normalizes dash-style pauses everywhere in the text (not just on shouty
     lines): both em dashes ("—") and plain-text double-hyphens ("--", a
     common ebook/plaintext substitute for an em dash) get converted to
     clean sentence breaks so the model doesn't stumble over them.
  4. Collapses spaced-out ellipses (". . .") into a single "..." token, so
     the model treats a trailing-off thought as one pause instead of reading
     each period as a separate sentence end (which can cause stuttering).
  5. Strips plain-text scene-break markers (e.g. "= = = = = =", "* * *",
     "---") that would otherwise get read aloud literally (e.g. "equals
     equals equals...").
  6. Collapses runs of multiple spaces/tabs down to a single space. Text
     copied from justified PDFs or "look inside" previews often has extra
     spaces stuffed between words to stretch each line to a fixed width;
     left as-is, that can translate into unwanted extra micro-pauses
     between words when read by the TTS engine.
  7. Applies a user-editable pronunciation-override dictionary (see
     PRONUNCIATION_OVERRIDES below) for invented/uncommon words the TTS
     engine tends to misread (e.g. Dune's "Gesserit", pronounced with a
     soft J, which Chatterbox may otherwise read with a hard G). Matches
     are whole-word and case-preserving, so "GESSERIT", "Gesserit", and
     "gesserit" all get correctly re-spelled without changing your actual
     source text's capitalization style.

Usage:
    python clean_for_tts.py some_folder
    python clean_for_tts.py some_folder --no-dash-split
    python clean_for_tts.py some_folder --caps-threshold 0.7
    python clean_for_tts.py some_folder --suffix _tts

All .txt files directly inside the given folder are cleaned. Each cleaned
file is written back into that same folder, alongside the originals, named
"<original_stem>_clean.txt" (the suffix can be changed with --suffix).
Files that already end in the clean suffix (e.g. from a previous run) are
skipped so re-running the script doesn't clean already-cleaned files.
"""

import argparse
import re
import sys
from pathlib import Path

# Common short acronyms/abbreviations you probably want to KEEP in caps
# even inside an otherwise-normalized line (extend as needed).
PRESERVE_CAPS = {
    "I", "OK", "TV", "US", "USA", "UK", "AI", "CEO", "FBI", "CIA", "NASA",
    "DNA", "PhD", "Mr", "Mrs", "Dr", "Jr", "Sr", 
}

# ==== EDIT THIS TO FIX MISPRONOUNCED WORDS ====
# Maps a word as it's actually spelled in your source text -> a respelling
# that nudges the TTS engine toward the correct pronunciation. Matching is
# whole-word and case-insensitive; the replacement's capitalization is
# automatically matched to however the original word was cased in the text
# (Title, ALLCAPS, or lowercase), so you only need to write each entry once
# in plain lowercase-or-Title form below.
#
# Force-exact-casing marker: add a trailing "!" to a replacement to make it
# always render exactly as written, ignoring the source word's casing. This
# matters for made-up respellings of real acronyms: "CHOAM" is genuinely an
# acronym (all-caps is correct for it), but its respelling "Chohm" is not an
# acronym at all, so the normal "match source casing" rule would turn it
# back into "CHOHM" -- right back to the original mispronunciation. Writing
# "Chohm!" instead pins the output to "Chohm" every time.
#
# Example: Dune's "Bene Gesserit" has a soft-J sound in "Gesserit" that
# Chatterbox otherwise reads with a hard G, like the "g" in "get".
PRONUNCIATION_OVERRIDES = {
    "Bene": "Benny",
    "Gesserit": "Jezzerit",
    "Muad'Dib": "Moo'ah'Deeb",
    "CHOAM": "Chohm!",
    "Fedaykin": "Feddyekin",
    "Feyd": "Fade",
    "Rautha": "Rawtha",
    "Feyd-Rautha": "Faydrawtha",
    "Fremen": "Fraymen",
    "Kwisatz": "Kweezahtch",
    "Haderach": "Hadurack",
    "Thufir": "Thoofurr",
    "Hawat": "Ha'Wahtt!",
    "Usul": "Oo'sull",
    "Sardaukar": "Sarduhcar",
    "Tleilaxu": "Tlayloxoo",
    "Harkonnen": "HarConan!",
    "Harkonnens": "HarConans!",
    "Atreides": "Atraydeez",
    "Yueh": "Youway!",
    "Piter": "Peeta"
}
# ===============================================


def apply_pronunciation_overrides(text: str, overrides: dict) -> str:
    """
    Whole-word, case-preserving find/replace for known TTS mispronunciations.
    Preserves the matched word's original casing style:
      - all uppercase   -> replacement in all uppercase
      - capitalized      -> replacement capitalized
      - anything else    -> replacement exactly as written in the dict

    Exception: if a replacement value ends with "!", that's a "force exact
    casing" marker, not a literal character to insert. The "!" is stripped
    and the replacement is written out exactly as spelled in the dict,
    regardless of how the matched source word was cased. This is for
    acronym-like words such as "CHOAM" -> "Chohm!", where "Chohm" is a
    made-up respelling (not a real acronym), so letting the all-caps source
    word force an all-caps "CHOHM" replacement would just recreate the
    original mispronunciation.
    """
    if not overrides:
        return text

    # Longest keys first, so multi-word overrides aren't shadowed by a
    # shorter overlapping entry.
    sorted_keys = sorted(overrides.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b",
        re.IGNORECASE,
    )

    # Case-insensitive lookup back to the canonical dict key/value, plus a
    # per-key flag recording whether the value had a trailing "!" (force
    # exact casing) so we strip it once here instead of re-checking it on
    # every match inside replace().
    lower_map = {}
    force_exact = {}
    for k, v in overrides.items():
        key = k.lower()
        if v.endswith("!"):
            force_exact[key] = True
            v = v[:-1]
        else:
            force_exact[key] = False
        lower_map[key] = v

    def replace(match: re.Match) -> str:
        matched = match.group(0)
        key = matched.lower()
        replacement = lower_map[key]
        if force_exact[key]:
            return replacement
        if matched.isupper():
            return replacement.upper()
        if matched[0].isupper():
            return replacement[0].upper() + replacement[1:].lower()
        return replacement.lower()

    return pattern.sub(replace, text)


def letter_upper_ratio(line: str) -> float:
    """Fraction of alphabetic characters in the line that are uppercase."""
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def is_shouty_line(line: str, threshold: float) -> bool:
    """
    Heuristic: a line counts as 'shouty' (needs normalizing) if it has a
    reasonable number of letters and most of them are uppercase.
    Short all-caps lines (like a single word 'STOP') are still caught,
    but we require at least a few letters to avoid false positives on
    stray single initials or punctuation-only lines.
    """
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 3:
        return False
    return letter_upper_ratio(line) >= threshold


def sentence_case(line: str) -> str:
    """
    Convert a shouty line to normal sentence case:
      - lowercase everything
      - capitalize the first letter of the line
      - capitalize the first letter after sentence-ending punctuation (. ! ?)
      - re-capitalize small set of preserved acronyms/titles
      - fix possessives/contractions mangled by naive title-casing (not an
        issue here since we use sentence case, not title case)
    """
    lowered = line.lower()

    # Capitalize first alphabetic character of the whole line.
    chars = list(lowered)
    for i, c in enumerate(chars):
        if c.isalpha():
            chars[i] = c.upper()
            break
    result = "".join(chars)

    # Capitalize first letter after ., !, ? followed by space(s).
    def cap_after_punct(match: re.Match) -> str:
        return match.group(1) + match.group(2).upper()

    result = re.sub(r"([.!?]\s+)([a-z])", cap_after_punct, result)

    # Restore preserved acronyms/abbreviations (case-insensitive match on
    # whole words only).
    for word in PRESERVE_CAPS:
        pattern = r"\b" + re.escape(word.lower()) + r"\b"
        result = re.sub(pattern, word, result, flags=re.IGNORECASE)

    return result


def split_dashes(line: str) -> str:
    """
    Replace dash-style pauses with periods to break run-on lines into
    separate sentences, giving the TTS engine clearer phrasing. Handles
    both real em dashes ("—") and the plain-text double-hyphen convention
    ("--") used by most plaintext ebooks in place of an em dash.
    Collapses any resulting double spacing/punctuation.
    """
    # Interrupted/trailing-off dialogue: a double-hyphen immediately before
    # a closing quote, bracket, or the end of the line (e.g. '"Your
    # Reverence, I --"') reads more naturally as a trailing-off ellipsis
    # than a hard sentence break. Protect it with the same placeholder used
    # for real ellipses so the dedup step below doesn't crush it back down.
    line = re.sub(r"--(?=[\"'\)\]]|$)", ELLIPSIS_PLACEHOLDER, line)

    # Real em dash, with or without surrounding spaces.
    line = re.sub(r"\s*—\s*", ". ", line)

    # A double-hyphen at the very start of a line is usually a word-wrap
    # artifact from plain-text formatting (the sentence continues from the
    # previous line, e.g. "Arrakis\n-- Dune -- Desert Planet." is really
    # one sentence: "Arrakis -- Dune -- Desert Planet."). Treat it as a
    # normal sentence break, same as a mid-line dash.
    line = re.sub(r"^--\s+", ". ", line)

    # ASCII double-hyphen used as an em-dash substitute elsewhere in the
    # line (needs word-boundary-ish spacing on at least one side so we
    # don't mangle real hyphenated words).
    line = re.sub(r"\s+--\s+", ". ", line)
    line = re.sub(r"(?<=\w)--\s+", ". ", line)
    line = re.sub(r"\s+--(?=\w)", ". ", line)

    # Avoid doubled punctuation like ".." created by the substitutions
    # above (this must NOT touch the ellipsis placeholder, which is why
    # trailing-off dashes were protected with it earlier in this function).
    line = re.sub(r"\.{2,}", ".", line)

    # Ensure a capital letter follows each new sentence break we created.
    def cap_after_period(match: re.Match) -> str:
        return match.group(1) + match.group(2).upper()

    line = re.sub(r"(\.\s+)([a-z])", cap_after_period, line)
    return line



# Private-use-area marker with NO alphabetic characters in it at all. This
# matters: is_shouty_line() counts letters to decide if a line needs
# recasing, so an earlier version of this placeholder (which spelled out
# "ELLIPSIS") could itself tip short lines over the caps threshold and get
# lowercased before it was restored, corrupting the placeholder.
ELLIPSIS_PLACEHOLDER = "\uE000\uE000\uE000"


def collapse_ellipses(text: str) -> str:
    """
    Collapse spaced-out ellipses (". . .", ".  .  .", etc.) into a single
    protected placeholder (later restored to "..."). TTS engines tend to
    treat each period as a distinct sentence-ending, so
    "floating . . . focusing" can come out as a series of stuttering pauses
    instead of one smooth trailing-off. Using a placeholder (instead of
    writing "..." directly here) keeps it safe from the punctuation-dedup
    step in split_dashes(), which would otherwise crush "..." back down to
    a single ".".
    """
    return re.sub(r"\.(\s*\.){2,}", ELLIPSIS_PLACEHOLDER, text)


def restore_ellipses(text: str) -> str:
    """Swap the protected ellipsis placeholder back to a real '...'."""
    return text.replace(ELLIPSIS_PLACEHOLDER, "...")


def normalize_number_ranges(text: str) -> str:
    """
    Convert a single hyphen directly between two numbers (e.g. page ranges,
    date ranges, "10,082-10,191") into " to ", so the TTS engine reads it as
    a spoken range instead of stumbling over the bare hyphen (which can come
    out as "dash", "minus", or a confused pause). Commas inside numbers
    (thousands separators) are allowed on either side.

    Deliberately narrow: requires a digit immediately before AND after the
    hyphen, so it won't touch hyphenated words ("well-known"), a leading
    negative number ("-5 degrees"), or an em-dash/double-hyphen range (those
    are handled separately by split_dashes() and won't match here, since the
    character right after the first hyphen in "--" is another hyphen, not a
    digit).
    """
    return re.sub(r"(\d[\d,]*)-(\d[\d,]*)", r"\1 to \2", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse runs of 2+ spaces or tabs down to a single space. Text copied
    from justified PDFs or "look inside" previews often has extra spaces
    stuffed between words to stretch each line to a fixed width (e.g.
    "A   beginning   is   the   time"), which can translate into unwanted
    extra micro-pauses between words when read by a TTS engine. This only
    touches horizontal whitespace (spaces/tabs), not newlines, so line
    structure and paragraph breaks are preserved.
    """
    return re.sub(r"[ \t]{2,}", " ", text)


def strip_scene_breaks(text: str) -> str:
    """
    Remove plain-text scene-break markers on their own line (e.g.
    "= = = = = =", "* * *", "- - -", "---", "***") that would otherwise be
    read aloud literally by the TTS engine. Replaced with a blank line so
    paragraph spacing is preserved.
    """
    def is_scene_break(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        # Line made up of only one repeated symbol (and optional spaces),
        # at least 3 symbol characters total, no letters/digits at all.
        if re.fullmatch(r"[=*_~-](\s*[=*_~-]){2,}", stripped):
            return True
        return False

    out_lines = []
    for line in text.splitlines():
        if is_scene_break(line):
            out_lines.append("")
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def clean_text(text: str, caps_threshold: float, do_split_dashes: bool) -> str:
    # Whitespace-normalization and line-independent passes first.
    text = normalize_whitespace(text)
    text = normalize_number_ranges(text)
    text = strip_scene_breaks(text)
    text = collapse_ellipses(text)
    text = apply_pronunciation_overrides(text, PRONUNCIATION_OVERRIDES)

    out_lines = []
    for line in text.splitlines():
        if is_shouty_line(line, caps_threshold):
            line = sentence_case(line)
        if do_split_dashes:
            line = split_dashes(line)
        out_lines.append(line)
    text = "\n".join(out_lines)

    # Restore ellipses last, after the dedup step in split_dashes() has
    # already run (that step is what the placeholder was protecting against).
    text = restore_ellipses(text)
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Normalize ALL-CAPS lines in every .txt file in a folder, "
                     "for cleaner TTS narration."
    )
    parser.add_argument(
        "input_dir", type=Path,
        help="Path to a folder containing .txt files to clean"
    )
    parser.add_argument(
        "--suffix", type=str, default="_clean",
        help="Suffix inserted before the extension of each cleaned file "
             "(default: _clean, e.g. part_001_clean.txt)"
    )
    parser.add_argument(
        "--caps-threshold", type=float, default=0.6,
        help="Fraction (0-1) of uppercase letters required to treat a line "
             "as shouty/needing normalization (default: 0.6)"
    )
    parser.add_argument(
        "--no-dash-split", action="store_true",
        help="Don't split em-dash-separated header lines into separate sentences"
    )
    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"❌ Input folder not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    txt_files = sorted(
        p for p in args.input_dir.glob("*.txt")
        if p.is_file() and not p.stem.endswith(args.suffix)
    )

    if not txt_files:
        print(f"⚠️  No .txt files found in: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    cleaned_count = 0
    for input_file in txt_files:
        output_path = input_file.with_name(
            input_file.stem + args.suffix + input_file.suffix
        )

        text = input_file.read_text(encoding="utf-8")
        cleaned = clean_text(
            text,
            caps_threshold=args.caps_threshold,
            do_split_dashes=not args.no_dash_split,
        )
        output_path.write_text(cleaned, encoding="utf-8")
        print(f"✅ Cleaned: {input_file.name} -> {output_path.name}")
        cleaned_count += 1

    print(f"\nDone. {cleaned_count} file(s) cleaned in: {args.input_dir}")


if __name__ == "__main__":
    main()
