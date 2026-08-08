"""Generate synthetic raw->canonical product pairs for electrical/plumbing supply.

Writes data/pairs.csv with columns: raw_description, canonical_description.
Deterministic under --seed. Noise applied: abbreviations/synonyms, unit format
changes, token reordering, typos, casing, separators, vendor SKU prefixes.

DONE WITH AI.
"""

import argparse
import csv
import itertools
import pathlib
import random

# canonical token -> raw variants seen in distributor POs/quotes
V = {
    "THHN": ["thhn", "THHN/THWN-2"],
    "XHHW": ["xhhw", "XHHW-2"],
    "Copper": ["CU", "Cu", "COPPER"],
    "Aluminum": ["AL", "ALUM", "aluminium"],
    "Wire": ["wire", "bldg wire", "building wire", "cable"],
    "Stranded": ["STR", "STRD"],
    "Solid": ["SOL"],
    "Black": ["BLK"],
    "White": ["WHT"],
    "Red": ["RD"],
    "Green": ["GRN"],
    "Blue": ["BLU"],
    "PVC": ["pvc", "P.V.C."],
    "Schedule 40": ["SCH40", "SCH 40", "sch-40"],
    "Schedule 80": ["SCH80", "SCH 80"],
    "Pipe": ["pipe", "tubing"],
    "EMT Conduit": ["EMT", "emt conduit", "E.M.T."],
    "Circuit Breaker": ["breaker", "CB", "circ brkr"],
    "1-Pole": ["1P", "SP", "single pole"],
    "2-Pole": ["2P", "DP", "double pole"],
    "Brass": ["BRS"],
    "Ball Valve": ["BV", "ball vlv"],
    "Threaded": ["THD", "NPT", "IPS"],
    "Sweat": ["SWT", "solder", "C x C"],
    "Elbow 90 Degree": ["90 ell", "90° elbow", "90 deg ell", "L90"],
    "Elbow 45 Degree": ["45 ell", "45° elbow", "45 deg ell"],
    "Tee": ["tee", "T"],
    "Coupling": ["cplg", "coup."],
    "Square D QO": ["SQD QO", "SQ D QO", "SquareD QO"],
    "Eaton BR": ["EATON BR", "Cutler-Hammer BR", "C-H BR"],
    "Siemens QP": ["SIE QP", "siemens qp"],
    "12 AWG": ["#12", "12ga", "12 GA", "AWG 12"],
    "14 AWG": ["#14", "14ga", "14 GA", "AWG 14"],
    "10 AWG": ["#10", "10ga", "10 GA", "AWG 10"],
    "8 AWG": ["#8", "8ga", "AWG 8"],
    "6 AWG": ["#6", "6ga", "AWG 6"],
    "1/2 in": ['1/2"', "0.5 in", "1/2-inch", "half inch"],
    "3/4 in": ['3/4"', "0.75 in", "3/4-inch"],
    "1 in": ['1"', "1-inch", "1.0 in"],
    "1-1/4 in": ['1-1/4"', "1.25 in", '1 1/4"'],
    "2 in": ['2"', "2-inch", "2.0 in"],
    "10 ft": ["10'", "10FT", "10 feet"],
    "20 ft": ["20'", "20FT"],
    "50 ft": ["50'", "50FT"],
    "100 ft": ["100'", "100FT", "100 feet"],
    "250 ft": ["250'", "250FT"],
    "500 ft": ["500'", "500FT", "500 feet"],
    "15 A": ["15A", "15 amp", "15 AMP"],
    "20 A": ["20A", "20 amp", "20 AMP"],
    "30 A": ["30A", "30 amp"],
    "50 A": ["50A", "50 amp"],
}

# category grammars: each field is a list of canonical tokens; canonical = one pick per field
GRAMMARS = [
    (
        124,
        [
            ["THHN", "XHHW"],
            ["Copper", "Aluminum"],
            ["Wire"],
            ["14 AWG", "12 AWG", "10 AWG", "8 AWG", "6 AWG"],
            ["Stranded", "Solid"],
            ["Black", "White", "Red", "Green", "Blue"],
            ["50 ft", "100 ft", "250 ft", "500 ft"],
        ],
    ),
    (
        30,
        [
            ["PVC"],
            ["Schedule 40", "Schedule 80"],
            ["Pipe"],
            ["1/2 in", "3/4 in", "1 in", "1-1/4 in", "2 in"],
            ["10 ft", "20 ft"],
        ],
    ),
    (
        35,
        [
            ["Copper"],
            ["Elbow 90 Degree", "Elbow 45 Degree", "Tee", "Coupling"],
            ["1/2 in", "3/4 in", "1 in", "1-1/4 in", "2 in"],
        ],
    ),
    (10, [["EMT Conduit"], ["1/2 in", "3/4 in", "1 in", "2 in"], ["10 ft"]]),
    (
        40,
        [
            ["Square D QO", "Eaton BR", "Siemens QP"],
            ["15 A", "20 A", "30 A", "50 A"],
            ["1-Pole", "2-Pole"],
            ["Circuit Breaker"],
        ],
    ),
    (25, [["Brass"], ["Ball Valve"], ["1/2 in", "3/4 in", "1 in", "2 in"], ["Threaded", "Sweat"]]),
]


def typo(s: str, rng: random.Random) -> str:
    i = rng.randrange(len(s) - 1)
    op = rng.choice(["swap", "drop", "dupe"])
    if op == "swap":
        return s[:i] + s[i + 1] + s[i] + s[i + 2 :]
    if op == "drop":
        return s[:i] + s[i + 1 :]
    return s[:i] + s[i] + s[i:]


def variate(tokens: list[str], rng: random.Random) -> str:
    toks = [rng.choice([t] + V.get(t, [])) if rng.random() < 0.7 else t for t in tokens]
    if rng.random() < 0.35:  # reorder: move one field elsewhere
        t = toks.pop(rng.randrange(len(toks)))
        toks.insert(rng.randrange(len(toks) + 1), t)
    raw = rng.choice(
        ["{}", "{}", "{}", "{} ea", f"ITM#{rng.randrange(9999):04d} {{}}".replace("{{}}", "{}")]
    ).format(rng.choice([" ", " ", " ", ", ", " - "]).join(toks))
    if rng.random() < 0.4:
        raw = typo(raw, rng)
    case = rng.choice([str, str, str.upper, str.lower, str.title])
    return case(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", type=int, default=10, help="raw rows per canonical")
    ap.add_argument("--out", default="data/pairs.csv")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    canonicals = []
    for n, fields in GRAMMARS:
        pool = list(itertools.product(*fields))
        combos = rng.sample(pool, min(n, len(pool)))
        canonicals += [list(c) for c in combos]

    rows, seen = [], set()
    for tokens in canonicals:
        canon = " ".join(tokens)
        while len({r for r, c in rows if c == canon}) < args.variants:
            raw = variate(tokens, rng)
            if raw not in seen:
                seen.add(raw)
                rows.append((raw, canon))
    rng.shuffle(rows)

    out = pathlib.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["raw_description", "canonical_description"])
        w.writerows(rows)
    print(f"{len(rows)} rows, {len(canonicals)} canonicals -> {out}")


if __name__ == "__main__":
    main()
