# Practice plan — Canals live coding (Wed Aug 13, noon)

## The single most important insight

Your interviewer is **Matias Weber, Matching Team Lead**. Canals' product literally ships
"Part Number Conversion: maps products across different manufacturer catalogs." The challenge
in this repo's README (raw → canonical product description matching) **is his team's day job**.
This is not a LeetCode interview — it's "build a small, honest version of our real system,
live, and show me how you work."

## What they're actually grading (from their values pages)

- **"Just do it" / YAGNI**: a working baseline in 30 minutes beats a half-built transformer at 90.
  Say "I'll start with the dumbest thing that works and iterate" — that sentence *is* their culture.
- **"Assuming the burden" / no sloppy work**: tests, a clean eval metric, good commit messages.
  They explicitly call out good commit messages and code comments as valued.
- **Directness / high-feedback**: think out loud, state trade-offs, and when Matias nudges you,
  engage with it immediately and visibly enjoy the feedback. This is a culture screen too.
- **End-to-end ownership**: data → train → eval → serve → test. Touch every layer, even shallowly.

## The 90-minute target architecture (know this cold)

```
prep-canals/
  data/            # pairs.csv: raw_description, canonical_description
  matcher/
    normalize.py   # lowercase, strip units/punct, expand abbreviations
    model.py       # TfidfVectorizer(char_wb 2-4) + NearestNeighbors(cosine)  OR rapidfuzz
    train.py       # fit, evaluate, joblib.dump to artifacts/
    evaluate.py    # accuracy@1, recall@k on held-out split
  api/
    main.py        # FastAPI: POST /match (single), POST /match/batch
  tests/
    test_normalize.py, test_model.py, test_api.py   # pytest + fastapi TestClient
  Dockerfile
```

Milestone order under time pressure (each is a commit, each leaves you demoable):
1. **min 0–10**: repo scaffold, load/inspect data, hold-out split
2. **min 10–30**: normalize + baseline matcher (exact-after-normalize, then TF-IDF char n-gram kNN)
3. **min 30–45**: evaluation harness — accuracy@1 / recall@5, print a small error table
4. **min 45–65**: FastAPI serving, single + batch endpoints, pydantic models
5. **min 65–80**: pytest coverage of normalize + API; Dockerfile
6. **min 80–90**: iterate on errors OR discuss production (retraining, monitoring, latency, fallback to fuzzy)

If the interview instead extends your take-home: same muscles apply — practice *modifying* your
own code fast (add batch endpoint, add a metric, swap the model).

## Skills to have in muscle memory (no lookups)

- `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))` + `NearestNeighbors(metric="cosine")`
  (and the rapidfuzz alternative: `process.extractOne(q, choices, scorer=fuzz.token_set_ratio)`)
- FastAPI from blank file: app, pydantic BaseModel, POST endpoints, `TestClient`, `uvicorn api.main:app`
- pytest: fixtures, `@pytest.mark.parametrize`, running a subset with `-k`
- `train_test_split`, `joblib.dump/load`, basic pandas (read_csv, value_counts, sample)
- A 10-line Dockerfile for a Python API, from memory
- git: small frequent commits with real messages, done without thinking

## The schedule (≈1–1.5 h/day, 3 full dress rehearsals)

**Thu Aug 7 — Environment + scaffold drill (60–90 min)**
Interview is confirmed to be in your own IDE. Set up the exact machine you'll use: font/theme,
no notifications, terminal split, Python env with fastapi/uvicorn/scikit-learn/rapidfuzz/pytest/
pandas preinstalled and verified.
Workflow to drill (no Jupyter, no migration step): `# %%` cells in .py files + interactive window
for EDA/driving, `%autoreload 2` so module edits hot-reload, pytest bound to one keystroke.
`explore.py` is the only scratch file; real code goes straight into modules and is driven from cells.
Debugging rule: a wrong match becomes a parametrized failing test, not REPL fiddling.
Drill: from empty dir to "scaffold + venv + first commit" three times. Target: under 5 minutes.
Still to ask the recruiter: AI-allowed or AI-off?

**Fri Aug 8 — Matching core (90 min)**
Generate a synthetic dataset (plumbing/electrical SKUs with abbreviations, typos, unit variants —
have Claude generate 500 pairs *today*, then never touch AI again for the drills).
From memory: normalize + TF-IDF kNN + eval harness. Don't look up syntax; when you're forced to,
write the incantation on a flashcard. Review flashcards daily after this.

**Sat Aug 9 — DRESS REHEARSAL 1 (90 min, timed, talking out loud)**
Full build from empty repo per the milestone list. Record your screen + voice.
After: 15 min review of the recording. Note every stall, every syntax lookup, every silent stretch.

**Sun Aug 10 — Serving + tests day (60 min)**
FastAPI single+batch endpoints and pytest suite from memory, twice. Dockerfile once.
Fix whatever stalled in rehearsal 1.

**Mon Aug 11 — DRESS REHEARSAL 2 (90 min, timed)**
Same task, but inject a twist at minute 45 — pick from the detours list in NOTES.md (learned
matching from feedback, attribute extraction, non-unique/dirty catalog, hand-rolled trigrams)
or a scope twist ("confidence threshold + abstain", "add a manufacturer field", "top-k results").
This trains the mid-interview pivot, which is where live coding is won or lost.

**Tue Aug 12 — Light day: story + flashcards (45 min, no heavy coding)**
Run flashcards. Rehearse your opening 3 minutes out loud: restate problem, propose milestone plan,
name your baseline-first trade-off. Prepare 3 good questions for Matias about the matching team.
Prepare the "production talk" answers: retraining cadence, drift, human-in-the-loop for low
confidence, latency of batch vs single. Sleep well.

**Wed Aug 13 — Morning: 20-min warm-up only**
One tiny scaffold drill + one FastAPI endpoint to warm the fingers. No new material. Go get it.

## Demo craft rules (practice these in every rehearsal)

1. Narrate intent before typing, not after: "I'm writing the eval first so every later change is measurable."
2. Never silently stall >20s: say what you're weighing.
3. When stuck on syntax: write a TODO placeholder, keep moving, come back.
4. Commit at every milestone with a real message — it's a visible artifact of their stated values.
5. Ask one scoping question up front ("do you care more about model quality or the serving path?") —
   then respect the answer.

## Reality check on your constraints

Kids + freelance + other interviews: this plan needs ~7 hours total across 7 days, front-loaded on
the weekend. If a day dies, sacrifice Sun/Tue, never a rehearsal. Two rehearsals minimum; the third
(Aug 12 could absorb one if Sat slips) is a bonus. And every hour transfers to the other interviews —
this same repo is your live-coding warm-up asset for the rest of the season.
