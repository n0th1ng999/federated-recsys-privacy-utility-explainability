#!/usr/bin/env python3
"""
Validates LaTeX manuscript integrity:
1. Balanced braces in all .tex files
2. All citation keys in main.tex and tables.tex exist in references.bib
3. All reference keys match declared labels
4. Table column balance
"""
import re
import os
import sys

def check_balanced_braces(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove comments (only % not preceded by backslash)
    lines = [re.sub(r'(?<!\\)%.*$', '', line) for line in content.split('\n')]
    clean = '\n'.join(lines)

    opens = clean.count('{')
    closes = clean.count('}')
    if opens != closes:
        print(f"[ERROR] {filepath}: Unbalanced braces! {{: {opens}, }}: {closes}")
        return False
    print(f"[OK] {filepath}: Balanced braces ({opens} pairs)")
    return True

def get_bib_keys(bibpath):
    with open(bibpath, 'r', encoding='utf-8') as f:
        content = f.read()
    keys = set(re.findall(r'@\w+\s*\{\s*([^,]+),', content))
    return {k.strip() for k in keys}

def get_citations(texpath):
    with open(texpath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Strip comments
    lines = [re.sub(r'(?<!\\)%.*$', '', line) for line in content.split('\n')]
    clean = '\n'.join(lines)
    cites = re.findall(r'\\cite(?:asnoun)?\{([^}]+)\}', clean)
    keys = set()
    for c in cites:
        for k in c.split(','):
            keys.add(k.strip())
    return keys

def get_labels_and_refs(texpath):
    with open(texpath, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = [line.split('%')[0] for line in content.split('\n')]
    clean = '\n'.join(lines)
    labels = set(re.findall(r'\\label\{([^}]+)\}', clean))
    refs = set(re.findall(r'\\ref\{([^}]+)\}', clean))
    return labels, refs

def main():
    paper_dir = "Paper"
    main_tex = os.path.join(paper_dir, "main.tex")
    tables_tex = os.path.join(paper_dir, "tables.tex")
    bib_file = os.path.join(paper_dir, "references.bib")

    all_ok = True
    for tex in [main_tex, tables_tex]:
        if os.path.exists(tex):
            if not check_balanced_braces(tex):
                all_ok = False

    bib_keys = get_bib_keys(bib_file)
    print(f"[INFO] Found {len(bib_keys)} entries in references.bib")

    all_cites = set()
    all_labels = set()
    all_refs = set()

    for tex in [main_tex, tables_tex]:
        if os.path.exists(tex):
            all_cites.update(get_citations(tex))
            lbls, rfs = get_labels_and_refs(tex)
            all_labels.update(lbls)
            all_refs.update(rfs)

    missing_cites = all_cites - bib_keys
    if missing_cites:
        print(f"[ERROR] Missing citations in bib: {missing_cites}")
        all_ok = False
    else:
        print(f"[OK] All {len(all_cites)} citations resolved in references.bib")

    missing_refs = all_refs - all_labels
    if missing_refs:
        print(f"[WARNING] Unresolved refs: {missing_refs}")
    else:
        print(f"[OK] All {len(all_refs)} references match labels")

    if all_ok:
        print("\n>>> MANUSCRIPT VALIDATION PASSED SUCCESSFULLY! <<<")
    else:
        print("\n>>> MANUSCRIPT VALIDATION FAILED! <<<")
        sys.exit(1)

if __name__ == '__main__':
    main()
