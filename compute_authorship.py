"""
compute_authorship.py

Reproduces the narrative authorship indicators reported in
"LLM-Storyteller: Child-Guided Narrative Co-Creation via Real-Time Adaptive
Prompting and Visualization with an Embodied Robot" (Alali & Ezzini).

It reads the 12 session logs released with the paper (session_*.xlsx) and
prints, per session and overall:

  Lexical Narrative Ownership (LNO)
      The share of unique content words in the child's finished narrative that
      first appeared in a child turn rather than in a robot turn. Section 4.4
      of the paper. This is a lexical-origin measure, not a semantic measure of
      story ownership.

  Question compliance
      The share of robot turns containing at least one question mark and no
      more than two sentence-like segments, where segments are obtained by
      splitting on periods, exclamation marks, and question marks. Section 4.4
      of the paper. The check is a surface one. It does not verify that a turn
      carries exactly one semantic question, and it does not inspect what an
      extra segment says.

Neither indicator establishes who decided the events or the direction of the
story. Section 4.4 of the paper states the limits of both.

Expected output (Table 7 of the paper):
      mean LNO 0.87 +/- 0.06, range 0.76 to 0.96
      child-origin content words 272 / 313
      question compliance 58 / 72, with 8 of the 14 exceptions at or after the
      wrap-up turn and 6 occurring earlier

Usage:
    pip install openpyxl
    python compute_authorship.py path/to/sessions/

Session IDs S1 to S12 in the paper are ordered by mean response latency,
ascending. This script prints the session ID from the log file instead, so the
mapping is explicit and nothing depends on the ordering.
"""

import glob
import os
import re
import statistics as st
import sys

import openpyxl

# Function words and discourse markers removed before attribution. The list is
# fixed and applied identically to both speakers; no per-session tuning.
STOP_WORDS = set("""
a an the and or but so if then than that this these those there here it its
i you he she they we me him her them my your his their our
is am are was were be been being do does did doing have has had having
of in on at to for with from by about into over under again very just too also as
not no yes yeah ok okay what when where who whom which why how
can could will would shall should may might must
one two three four five some any all each every both few more most other another such only own same
up down out off once because while during before after above below between through
let go going get got say said says tell told make made
um uh hmm like know err uhh umm right indeed actually well now oh wow great good nice really
story stories telling thing things something someone lot bit way sure think
""".split())

# Words whose final 's' is part of the stem, so the naive stemmer must skip them.
IRREGULAR = {"this", "its", "yes", "was", "has", "does", "goes"}

# From this turn onward the system prompt appends a wrap-up instruction, so a
# closing statement rather than a question is the intended behaviour. Exceptions
# at or after this turn are reported separately from exceptions before it.
WRAPUP_TURN = 4


def normalise(word: str) -> str:
    """Lowercase, strip non-alphabetic characters, remove a plural 's'."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if word in IRREGULAR:
        return word
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def content_words(text: str) -> list:
    """Content words of an utterance, in order of appearance."""
    if not text:
        return []
    out = []
    for token in re.split(r"\s+", text):
        w = normalise(token)
        if w and w not in STOP_WORDS and len(w) > 2:
            out.append(w)
    return out


def read_turns(path: str) -> list:
    """Turn-by-turn records from the 'Turns' sheet of a session log."""
    sheet = openpyxl.load_workbook(path)["Turns"]
    rows = list(sheet.iter_rows(values_only=True))
    header = rows[0]
    return [dict(zip(header, r)) for r in rows[1:] if r[0] not in (None, "")]


def is_question_compliant(robot_turn: str) -> bool:
    """
    True if the robot turn contains at least one question mark and splits into
    no more than two sentence-like segments.

    The check is a surface one. It does not verify that the turn carries exactly
    one semantic question, and it does not inspect the content of an additional
    segment.
    """
    if not robot_turn or "?" not in robot_turn:
        return False
    sentences = [s for s in re.split(r"[.!?]+", robot_turn) if s.strip()]
    return len(sentences) <= 2


def score_session(path: str) -> dict:
    """LNO and question compliance for one session."""
    turns = read_turns(path)

    seen = set()          # every content word uttered so far, either speaker
    child_first = set()   # content words whose first utterance was the child's
    narrative = []        # all content words in the child's turns
    compliant = 0         # robot turns meeting the question-compliance criterion
    wrapup_exceptions = 0 # non-compliant turns at turn >= 4 (wrap-up phase)
    early_exceptions = 0  # non-compliant turns before the wrap-up trigger

    for turn in turns:
        child_text = turn.get("Child Text") or ""
        robot_text = turn.get("Robot Response") or ""
        turn_number = int(turn.get("Turn #", 0))

        # Child speaks first in each turn, so attribute the child's words first.
        words = content_words(child_text)
        narrative.extend(words)
        for w in words:
            if w not in seen:
                seen.add(w)
                child_first.add(w)

        if is_question_compliant(robot_text):
            compliant += 1
        elif turn_number >= WRAPUP_TURN:
            wrapup_exceptions += 1
        else:
            early_exceptions += 1

        # Robot words seen for the first time here belong to the robot.
        for w in content_words(robot_text):
            if w not in seen:
                seen.add(w)

    unique = set(narrative)
    child_origin = len([w for w in unique if w in child_first])
    lno = child_origin / len(unique) if unique else 0.0

    return {
        "session": os.path.basename(path).replace("session_", "").replace(".xlsx", ""),
        "unique": len(unique),
        "child_origin": child_origin,
        "lno": lno,
        "compliant": compliant,
        "wrapup_exceptions": wrapup_exceptions,
        "early_exceptions": early_exceptions,
        "turns": len(turns),
    }


def main(folder: str) -> None:
    paths = sorted(glob.glob(os.path.join(folder, "session_*.xlsx")))
    if not paths:
        sys.exit(f"No session_*.xlsx files found in {folder!r}")

    results = [score_session(p) for p in paths]

    print(f"{'session':>10} {'unique':>7} {'child-first':>12} {'LNO':>6} {'compliance':>12}")
    print("-" * 54)
    for r in results:
        print(f"{r['session']:>10} {r['unique']:>7} {r['child_origin']:>12} "
              f"{r['lno']:>6.2f} {r['compliant']:>7}/{r['turns']}")

    lnos = [r["lno"] for r in results]
    turns = sum(r["turns"] for r in results)
    compliant = sum(r["compliant"] for r in results)
    wrapup = sum(r["wrapup_exceptions"] for r in results)
    early = sum(r["early_exceptions"] for r in results)

    print("-" * 54)
    print(f"sessions                       : {len(results)}")
    print(f"turns                          : {turns}")
    print(f"mean LNO                       : {st.mean(lnos):.3f} +/- {st.stdev(lnos):.3f}")
    print(f"LNO range                      : {min(lnos):.3f} to {max(lnos):.3f}")
    print(f"child-origin content words     : {sum(r['child_origin'] for r in results)}"
          f" / {sum(r['unique'] for r in results)}")
    print(f"question compliance            : {compliant} / {turns}")
    print(f"  exceptions at/after turn {WRAPUP_TURN}   : {wrapup}   (wrap-up phase, closing is intended)")
    print(f"  exceptions before turn {WRAPUP_TURN}     : {early}   (outside the wrap-up phase)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
