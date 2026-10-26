#!/usr/bin/env python3
"""
Static checks for the manuscript (no TeX installation required).

1. Balanced braces in main.tex, supplementary.tex and generated/*.tex
2. Every citation key exists in references.bib
3. Every \\ref has a matching \\label (main text, supplement, generated tables;
   S- prefixed references resolve against the supplement)
4. Every \\R{key} used in the text is defined in generated/results_macros.tex
   (numbers in the text come only from the result files)
5. No hand-written tabular environment in main.tex or supplementary.tex outside
   the static tables S1--S2 (all result tables are generated)
6. Word counts per Emerald rules: abstract (incl. keywords and paper type) <= 250;
   article total (abstract, body, table text, references, back matter) <= 9000,
   with an alternative count charging 280 words per table/figure.
"""
import glob
import os
import re
import subprocess
import sys

PAPER = "Paper"
MAIN = os.path.join(PAPER, "main.tex")
SUPP = os.path.join(PAPER, "supplementary.tex")
BIB = os.path.join(PAPER, "references.bib")
GEN = os.path.join(PAPER, "generated")
STATIC_SUPP_TABLES = {"tab:supp_related_work", "tab:supp_hyperparams"}
ABSTRACT_LIMIT, ARTICLE_LIMIT, WORDS_PER_DISPLAY = 250, 9000, 280


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def strip_comments(text):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.split("\n"))


def expand_inputs(text, base=PAPER):
    def repl(mo):
        p = os.path.join(base, mo.group(1))
        if not p.endswith(".tex"):
            p += ".tex"
        return strip_comments(read(p)) if os.path.exists(p) else ""
    return re.sub(r"\\input\{([^}]+)\}", repl, text)


def plain_words(tex_fragment):
    """Word count of a LaTeX fragment via pandoc's plain-text writer."""
    out = subprocess.run(["pandoc", "-f", "latex", "-t", "plain", "--wrap=none"], input=tex_fragment,
                         capture_output=True, text=True)
    return len(out.stdout.split())


def macros():
    path = os.path.join(GEN, "results_macros.tex")
    if not os.path.exists(path):
        return {}
    return dict(re.findall(r"\\csname rv@([^\\]+)\\endcsname\{(.*)\}$", read(path), flags=re.M))


def substitute_macros(text, values):
    return re.sub(r"\\R\{([^}]+)\}", lambda mo: values.get(mo.group(1), "0"), text)


def main():
    ok = True
    gen_files = sorted(glob.glob(os.path.join(GEN, "*.tex")))

    for path in [MAIN, SUPP] + gen_files:
        clean = strip_comments(read(path))
        if clean.count("{") != clean.count("}"):
            print(f"[ERROR] unbalanced braces in {path}")
            ok = False
    print(f"[OK] brace balance checked in {2 + len(gen_files)} files" if ok else "")

    bib = set(k.strip() for k in re.findall(r"@\w+\s*\{\s*([^,]+),", read(BIB)))
    main_full = expand_inputs(strip_comments(read(MAIN)))
    supp_full = expand_inputs(strip_comments(read(SUPP)))
    cites = set()
    for c in re.findall(r"\\cite(?:asnoun)?\{([^}]+)\}", main_full + supp_full):
        cites.update(k.strip() for k in c.split(","))
    missing = cites - bib
    if missing:
        print(f"[ERROR] citations missing from references.bib: {sorted(missing)}")
        ok = False
    else:
        print(f"[OK] {len(cites)} cited keys resolved")

    main_labels = set(re.findall(r"\\label\{([^}]+)\}", main_full))
    supp_labels = set(re.findall(r"\\label\{([^}]+)\}", supp_full))
    unresolved = set()
    for name, text, labels in (("main", main_full, main_labels), ("supplement", supp_full, supp_labels)):
        for ref in re.findall(r"\\ref\{([^}]+)\}", text):
            target = supp_labels if ref.startswith("S-") else labels
            key = ref[2:] if ref.startswith("S-") else ref
            if key not in target:
                unresolved.add((name, ref))
    for name, ref in sorted(unresolved):
        print(f"[ERROR] unresolved \\ref{{{ref}}} in {name}")
    if unresolved:
        ok = False
    else:
        print("[OK] all cross-references resolve")

    values = macros()
    used = set(re.findall(r"\\R\{([^}]+)\}", main_full + supp_full))
    undefined = sorted(used - set(values))
    if undefined:
        print(f"[ERROR] {len(undefined)} \\R keys not generated: {undefined[:20]}")
        ok = False
    else:
        print(f"[OK] {len(used)} \\R keys defined ({len(values)} generated)")

    for path in (MAIN, SUPP):
        own = strip_comments(read(path))
        for mo in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", own, flags=re.S):
            label = re.search(r"\\label\{([^}]+)\}", mo.group(0))
            if not label or label.group(1) not in STATIC_SUPP_TABLES:
                print(f"[ERROR] hand-written table in {path}: {label.group(1) if label else '(no label)'}")
                ok = False
    print("[OK] result tables are generated")

    # Word counts (main article only; the supplement is a separate file)
    text = substitute_macros(main_full, values)
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, flags=re.S).group(1)
    n_abstract = plain_words(abstract)
    body = re.search(r"\\end\{abstract\}(.*?)\\bibliographystyle", text, flags=re.S).group(1)
    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", body, flags=re.S)
    figures = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", body, flags=re.S)
    body_wo = body
    for t in tables + figures:
        body_wo = body_wo.replace(t, "")
    n_body = plain_words(body_wo)
    n_tables = sum(plain_words(t) for t in tables)
    n_figcap = sum(plain_words(f) for f in figures)
    cited = set()
    for c in re.findall(r"\\cite(?:asnoun)?\{([^}]+)\}", body):
        cited.update(k.strip() for k in c.split(","))
    bibtext = read(BIB)
    n_refs = 0
    for key in cited:
        mo = re.search(r"@\w+\s*\{\s*" + re.escape(key) + r",(.*?)\n\}", bibtext, flags=re.S)
        if mo:
            fields = re.findall(r"(author|title|booktitle|journal|volume|pages|year|howpublished|institution)\s*=\s*\{(.*)\}", mo.group(1))
            n_refs += sum(len(re.sub(r"[{}\\]", "", v).split()) for _, v in fields)
    total_text = n_abstract + n_body + n_tables + n_figcap + n_refs
    total_display = n_abstract + n_body + n_refs + WORDS_PER_DISPLAY * (len(tables) + len(figures))
    print(f"[INFO] abstract {n_abstract} words (limit {ABSTRACT_LIMIT})")
    print(f"[INFO] body {n_body}; tables {len(tables)} ({n_tables} words); figures {len(figures)}; references {len(cited)} ({n_refs} words)")
    print(f"[INFO] total, table text counted: {total_text}; total, 280 words per table/figure: {total_display} (limit {ARTICLE_LIMIT})")
    if n_abstract > ABSTRACT_LIMIT:
        print("[ERROR] abstract too long")
        ok = False
    if max(total_text, total_display) > ARTICLE_LIMIT:
        print("[ERROR] article exceeds the word limit under at least one counting rule")
        ok = False

    print("\n>>> VALIDATION PASSED <<<" if ok else "\n>>> VALIDATION FAILED <<<")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
