"""Audit the repository's prose against the house style, and print numbers instead of impressions.

The paper had its own audit from the start; the README, the experiment write-ups, the script headers and
the strings inside the plotting scripts did not, which is why the "X rather than Y" construction stood at
200 instances across tracked files while the manuscript sat at a curated ten, and why the last
full-sentence titles in the project survived inside figures. Every check here exists because the rule was
broken without that being visible on reading.

    python tools/check_prose.py              audit
    python tools/check_prose.py -v           list every instance
    python tools/check_prose.py --fix-rather rewrite the construction where it is mechanical, then audit

The rewrite is conservative by design. An earlier attempt replaced all 200 at once and produced real
damage: "variance in place of bias", "answered structurally in place of empirically", and a systematic
space before an inserted comma. Both faults are fixed here, and "in place of" is now offered only where
a determiner follows, but the pass still declines anything it cannot classify and reports it for a human.

Exit status is 0 if nothing is flagged and 1 otherwise.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERBOSE = "-v" in sys.argv
FIX = "--fix-rather" in sys.argv

RED, GRN, YLW, OFF = "\033[31m", "\033[32m", "\033[33m", "\033[0m"

# per-file caps: the construction is right where it weighs two readings and no substitute does the work
CAP_MD, CAP_SCRIPT = 2, 2

DET = {"a", "an", "the", "its", "their", "our", "one", "two", "three", "any", "some", "this",
       "that", "these", "those", "no", "all", "most", "each", "every", "another", "100"}
PREP = {"of", "to", "on", "in", "by", "with", "from", "as", "at", "for", "about", "against",
        "between", "over", "under", "into", "through", "across", "per", "here", "there"}
NOT_GERUND = {"something", "nothing", "anything", "everything", "during", "string", "thing"}

RATHER = re.compile(r"(,)?(\s*)\brather\s+than\b(\s+)(\S+)")
_rota = {}


def pick(nxt, had_comma):
    """Which plainer form is grammatical before `nxt`, rotated so one form does not take over."""
    w = nxt.strip(".,;:)\"'*_`").lower()
    if w.endswith("ing") and w not in NOT_GERUND:
        k, forms = "gerund", ["instead of"]
    elif w in DET:
        k, forms = "det", [", not", "and not", "in place of"]
    elif w in PREP or w.endswith("ly") or (w.endswith("ed") and len(w) > 3):
        k, forms = "closed", ["and not", ", not"]
    elif re.fullmatch(r"[a-z][a-z-]+", w):
        k, forms = "noun", [", not", "and not"]
    else:
        return None                      # capitalised, numeric or marked-up: leave it alone
    _rota[k] = _rota.get(k, 0) + 1
    form = forms[(_rota[k] - 1) % len(forms)]
    if had_comma and form == ", not":    # a comma is already there, so do not add a second
        form = "and not"
    return form


def rewrite(text):
    edits, skipped = [], []
    for m in RATHER.finditer(text):
        comma, pre, gap, nxt = m.group(1), m.group(2), m.group(3), m.group(4)
        form = pick(nxt, bool(comma))
        if form is None:
            skipped.append(nxt)
            continue
        # the span replaced runs from the start of the match to just before `nxt`, so the preceding
        # whitespace is consumed and an inserted comma cannot end up detached from its word
        if form == ", not":
            rep = ", not "
        else:
            rep = (comma or "") + (pre if pre else " ") + form + " "
        edits.append((m.start(), m.end() - len(nxt), rep))
    for a, b, rep in reversed(edits):
        text = text[:a] + rep + text[b:]
    return text, len(edits), skipped


files = [f for f in subprocess.run(["git", "ls-files", "*.md", "*.py", "*.jl", "*.sh"], cwd=str(ROOT),
                                   capture_output=True, text=True, encoding="utf-8").stdout.split()
         if "check_prose" not in f]

if FIX:
    print("\nREWRITING 'rather than' WHERE IT IS MECHANICAL")
    done = left = 0
    for rel in files:
        p = ROOT / rel
        try:
            s = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if not re.search(r"\brather\b", s, re.I):
            continue
        new, n, skip = rewrite(s)
        done += n
        left += len(skip)
        if new != s:
            p.write_text(new, encoding="utf-8", newline="\n")
            print("  %-50s %d rewritten, %d left" % (rel, n, len(skip)))
    print("  %d rewritten, %d left for a human" % (done, left))

print("\nPER-FILE 'rather' COUNT  (cap %d a file)" % CAP_MD)
over = []
for rel in files:
    p = ROOT / rel
    try:
        s = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        continue
    n = len(re.findall(r"\brather\b", s, re.I))
    cap = CAP_MD if rel.endswith(".md") else CAP_SCRIPT
    if n > cap:
        over.append((rel, n))
tot = sum(len(re.findall(r"\brather\b", (ROOT / f).read_text(encoding="utf-8", errors="replace"), re.I))
          for f in files)
ok = not over
print("  [%s] total %d across %d files; %d files over cap"
      % ((GRN + " ok " + OFF) if ok else (RED + "FLAG" + OFF), tot, len(files), len(over)))
for rel, n in sorted(over, key=lambda x: -x[1])[:12]:
    print("        %-50s %d" % (rel, n))

print("\nMARKDOWN HEADINGS  (plain noun phrases)")
# The last clause bans the two-halves title, "X, and Y". It is the contrastive pair's twin: a title
# that promises a name and delivers an apposition, and it stood at 44 instances project-wide. Where
# the halves are a tight noun pair the comma simply goes; where the second half was a pointer, the
# replacement names the thing instead.
BAD = re.compile(r"\b(is|are|was|were|does|do|did|has|have|will|can|cannot|must|gets?|makes?)\b"
                 r"|^(What|Why|How|Whether)\b|,\s*(Not|not)\s|,\s+and\s", re.I)
bad_heads = []
for rel in [f for f in files if f.endswith(".md")]:
    fenced = False
    for ln in (ROOT / rel).read_text(encoding="utf-8", errors="replace").split("\n"):
        if ln.lstrip().startswith("```"):
            fenced = not fenced          # a '#' inside a fence is a code comment, not a heading
            continue
        if fenced:
            continue
        if ln.startswith("#"):
            h = ln.lstrip("# ").strip()
            if h and BAD.search(h):
                bad_heads.append((rel, h))
print("  [%s] %d headings that are not noun phrases"
      % ((GRN + " ok " + OFF) if not bad_heads else (RED + "FLAG" + OFF), len(bad_heads)))
if bad_heads and (VERBOSE or len(bad_heads) <= 12):
    for rel, h in bad_heads[:14]:
        print("        %-34s %s" % (rel.split("/")[-1], h[:70]))

print("\nBRITISH SPELLING AND BANNED REGISTER")
AMER = (r"\bartifact|\bbehavior\b|normaliz|analyz|recogniz|optimiz|generaliz|emphasiz|summariz"
        r"|utiliz|paralleliz|binariz|regulariz|\bfavor\b|\bmodeling\b|\blabeled\b")
HEDGE = r"we believe|arguably|it is worth noting|it should be noted|\bnovel\b|we are the first"
a = h = 0
for rel in files:
    low = (ROOT / rel).read_text(encoding="utf-8", errors="replace").lower()
    a += len(re.findall(AMER, low))
    h += len(re.findall(HEDGE, low))
print("  [%s] American spellings %d" % ((GRN + " ok " + OFF) if not a else (YLW + "warn" + OFF), a))
print("  [%s] hedging and marketing %d" % ((GRN + " ok " + OFF) if not h else (RED + "FLAG" + OFF), h))

print("\nFIGURE HEADINGS  (plain noun phrases)")
# A heading slot has to be found mechanically here, because a figure carries no '#' to key on: an
# all-caps label of two or more words, a set_title or suptitle argument, a string set in semibold or at
# display size, or a string sharing a tuple with an all-caps label, which is how a subtitle sits under
# its head. A string ending in a full stop is prose inside the panel and is exempt. That last signal is
# the one that separates the two reliably; font size alone does not, since a figure sets body text in
# semibold for emphasis.
CAPS = re.compile(r"^[A-Z0-9 ,.'$&()%+–—-]+$")
VERBISH = re.compile(r"\b(?:is|are|was|were|does|do|did|has|have|had|will|can|cannot|must|should"
                     r"|uses|gives|gave|takes|took|reaches|lands|retired|survives|equals|moves"
                     r"|needs|holds|leads|trails|recovers|separates|contributes|sits)\b", re.I)


def heading_slots(src):
    lines = src.split("\n")
    found = {}

    def add(node):
        t = node.value
        if t.rstrip().endswith((".", ",")) or len(t.split()) < 2:
            return
        if t.startswith(("$", "→")) or "\\" in t:   # maths, an arrow bullet, a TeX fragment
            return
        found.setdefault((node.lineno, t), True)

    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Tuple):
            es = [e for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if any(CAPS.fullmatch(e.value) and e.value.strip() for e in es):
                for e in es:
                    add(e)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            ln = lines[node.lineno - 1]
            stmt = "\n".join(lines[max(0, node.lineno - 3):node.lineno + 2])
            m = re.search(r"fontsize=([\d.]+)", stmt)
            if ((CAPS.fullmatch(node.value) and len(node.value.split()) >= 2)
                    or ".set_title(" in ln or ".suptitle(" in ln
                    or 'weight="semibold"' in stmt or (m and float(m.group(1)) >= 12)):
                add(node)
    return sorted(found)


fig_heads = []
for rel in [f for f in files if "figures/" in f and f.endswith(".py")]:
    for ln, t in heading_slots((ROOT / rel).read_text(encoding="utf-8")):
        if (VERBISH.search(t) or re.match(r"\s*(what|why|how|whether)\b", t, re.I)
                or re.search(r",\s*not\s", t, re.I) or re.search(r",\s+and\s", t, re.I)):
            fig_heads.append((rel, ln, t))
print("  [%s] %d figure headings that are not noun phrases"
      % ((GRN + " ok " + OFF) if not fig_heads else (RED + "FLAG" + OFF), len(fig_heads)))
for rel, ln, t in fig_heads[:14]:
    print("        %-30s %4d  %s" % (rel.split("/")[-1], ln, t.replace("\n", " / ")[:60]))

bad = bool(over or bad_heads or h or fig_heads)
print("\n" + ((GRN + "all checks pass" + OFF) if not bad else (RED + "flagged" + OFF)))
sys.exit(1 if bad else 0)
