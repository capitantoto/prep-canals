# Notes — trivia, snippets, detours (crucial stuff lives in PLAN.md)

Running notebook of everything discussed while prepping. Reviewed-later material, not the plan.

## Matching theory

### TF-IDF on char n-grams (the baseline)
Split text into overlapping character chunks instead of words: `"SCH40"` → `SCH`, `CH4`, `H40`.
Typos/abbreviations/format drift degrade the score gracefully instead of zeroing it (word-level
TF-IDF sees `SCH40` vs `SCH 40` as disjoint vocab). IDF down-weights ubiquitous chunks (` wi`,
`ire`) and boosts discriminative ones (`QO `, `XHH`).

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))  # char_wb = pad words with spaces
X = vec.fit_transform(catalog)  # index the canonicals
nn = NearestNeighbors(metric="cosine").fit(X)
dist, idx = nn.kneighbors(vec.transform(queries), n_neighbors=5)
```

### pg_trgm = same family, minus IDF
`similarity(a,b)` = shared trigrams / total distinct trigrams (set overlap, fixed n=3,
word-boundary padded, no weighting). Its superpower is indexability:

```sql
CREATE EXTENSION pg_trgm;
CREATE INDEX ON catalog USING gin (description gin_trgm_ops);
SELECT * FROM catalog WHERE description % 'thhn cu 12ga str blk 500'''
ORDER BY similarity(description, 'thhn cu 12ga str blk 500''') DESC LIMIT 50;
```

Production pattern for Canals stack (they run Postgres + Elasticsearch): **pg_trgm as indexed
candidate generator (recall), rerank with TF-IDF/attributes/cross-encoder (precision)** —
classic blocking → ranking. Elasticsearch equivalent: ngram analyzer + BM25 (BM25 does include
IDF-style weighting).

### Learned matching landscape (this IS partly a train/test problem)
- **Unsupervised retrieval** (the 90-min baseline): normalize → index → kNN. Pairs used for
  *evaluation only*.
- **Fellegi–Sunter** probabilistic record linkage: libraries `splink`, `dedupe`.
- **Pairwise classifier** over similarity features (fuzzy scores, attribute agreements) trained
  on match/non-match pairs.
- **Bi-encoder fine-tuned with contrastive loss** on (raw, canonical) pairs — sentence-transformers,
  ~30 lines; genuinely *learns* "CU means copper", generalizes to unseen raws; embed catalog + ANN.
- **Cross-encoder reranker** for scoring top candidates (see the *Ditto* paper — BERT entity matching).

**Eval subtlety worth saying out loud:** two held-out regimes — (a) unseen raw variants of known
canonicals ("seen product, new phrasing"), (b) entirely held-out canonicals (cold start). A model
can ace (a) and fail (b).

### Entity/attribute extraction (maybe the strongest card)
Numeric attributes are near in string space, disjoint in meaning: `1/2 in` vs `3/4 in`, `12 AWG`
vs `14 AWG`, the dataset's `#122` trap. Pure string similarity returns confidently wrong SKUs;
in procurement wrong-but-confident is worse than abstain. Architecture: extract structured attrs
(material, type, gauge, dimension, length, color) — regex live, LLM at scale (what Canals' Parsing
team does) — then **hard-match / heavily penalize numeric mismatches, fuzzy-match the remainder**.
A 15-line gauge/size regex extractor used as a match filter is a big differentiator over TF-IDF-only.

## Detours from the practice architecture (rehearsal-2 twist candidates / discussion topics)

1. **Learned matching from user feedback.** Canals mentions learning from clients' interactions
   with matching picks. Every accepted/corrected match is a labeled pair → retraining data.
   Discussion: feedback loop design (log pick + shown candidates, not just pick), position bias,
   incremental vs periodic retraining, per-customer vs global models (same raw string can map
   differently per client's catalog!), drift monitoring.
2. **NLP attribute extraction, then match on semantics.** Extract material/dimensions/type into a
   structured record; match structured-to-structured. Relates to (3): attributes define an
   equivalence class, not necessarily one SKU.
3. **Catalog non-uniqueness / dirty catalog.** Same physical product may have multiple valid SKUs
   (different suppliers), duplicate/near-duplicate catalog rows, or the catalog itself is dirty.
   Then the target isn't "the one canonical" but a *cluster* of equivalent items → dedupe the
   catalog first (same matching tech pointed at itself), return any/all SKUs in the cluster,
   accuracy metric must count any cluster member as correct. Business rules pick the SKU within
   a cluster (preferred supplier, price, stock).
4. **Hand-rolled trigram extractor as drop-in for TfidfVectorizer.** Nice "I understand the
   internals" demo:

   ```python
   def trigrams(s: str) -> list[str]:
       s = f"  {s.lower()} "  # pg_trgm-style padding
       return [s[i : i + 3] for i in range(len(s) - 2)]


   vec = TfidfVectorizer(analyzer=trigrams)  # custom analyzer, rest unchanged
   # or go full DIY: Jaccard over set(trigrams(a)) & set(trigrams(b)) == pg_trgm's similarity()
   ```

### Inspecting sparse TF-IDF without densifying
CSR rows carry `.indices` (nonzero columns) + `.data` (values); feature names index natively:

```python
names = tv.get_feature_names_out()
row = raw_tf[i]  # 1×V CSR slice
pd.Series(row.data, index=names[row.indices]).sort_values(ascending=False)
# sklearn built-in (names only, no weights): tv.inverse_transform(X)
# whole matrix: coo = X.tocoo(); zip(coo.row, names[coo.col], coo.data)
# CSR internals: row i = indices[indptr[i]:indptr[i+1]] — the "how is it stored" answer
```

Same idea on the score matrix: `sims = raw_tf @ canon_tf.T` is sparse too — `sims[i].indices/.data`
IS the top-k candidates payload. (NB: scipy `cdist` defaults to euclidean; ranks same as cosine
only because sklearn TF-IDF is L2-normalized. Prefer the sparse matmul.)

### Project-level `check` command (aliases don't cross process boundaries)
dotenv/direnv export env vars only — never aliases/functions. Options: `bin/check` script +
`PATH_add bin` in `.envrc` (direnv-native, works on cd); or the conventional **`Makefile` with a
`check:` target** — `make check` = ruff check + format --check + pytest. Interview repo: Makefile.

## What tests to write — the decision procedure
One question per test: **"what concrete failure would this catch, and do I care?"** No answer → delete.
1. **Regression cases from error analysis** — every miss class actually found (glued `250ft`,
   decimal destruction, `aluminium`, ITM# fake-awg) = one parametrize row. Strongest rationale:
   the failure demonstrably happened. Live move when a new bug appears: failing case first → fix →
   green. Tests-as-debugging-log.
2. **Promises, not mechanisms** (Matcher): exact string matches itself; variant resolves to right
   SKU; results sorted & bounded; save/load faithful. Never pin scores/vocab sizes — if a
   legitimate improvement would break the test, it tests the mechanism.
3. **One test per seam**: normalize (parametrize), Matcher (behavioral fixtures), API (one
   TestClient round-trip + one 422). Integration catches "pieces don't fit" — one is enough.
4. **Don't test the framework** (pydantic validation, sklearn cosine, FastAPI routing) — say so
   out loud; it's a YAGNI statement and it funds the tests that matter.
5. **Time-boxed recipe**: bug cases + one promise per public method + one round-trip ≈ 10 tests,
   ~15 min. Priority under pressure: regression cases first.
Framing for Canals (no manual QA): "regression tests for every bug I hit, behavioral tests for
every promise the interface makes, one integration test per seam, nothing that tests the framework."

### pytest/FastAPI incantations (flashcards)
- `@pytest.mark.parametrize("raw, expected", [(..., ...)])` — each tuple = independent test; `-k` selects.
- `@pytest.fixture(scope="module")` — name becomes an argument tests request; `tmp_path` built-in for save/load.
- `TestClient(app)` runs in-process, no server: `client.post("/match", json={...})`, `resp.json()`, assert 200/422.
- FastAPI minimal: pydantic models declare shapes; `@app.post("/match")`; module-level artifact load;
  batch-first (single = list of one). Run: `uvicorn api:app --reload` → demo from `/docs` (Swagger).

## Library trivia

**fuzzywuzzy → thefuzz → rapidfuzz lineage:** fuzzywuzzy (SeatGeek, 2011) was slow in pure Python
and GPL-tainted via `python-Levenshtein`. thefuzz (2021) is the *same SeatGeek project renamed*,
now a thin compatibility shim delegating everything to rapidfuzz. rapidfuzz (Max Bachmann,
independent) is a C++ MIT-licensed reimplementation that outgrew the original: more scorers,
`distance` module, `score_cutoff` early-exit, `process.cdist` many-to-many matrix.
**Use rapidfuzz directly; skip the wrapper.**

```python
from rapidfuzz import fuzz, process

best = process.extractOne(query, catalog, scorer=fuzz.token_set_ratio, score_cutoff=70)
matrix = process.cdist(raw_batch, catalog, scorer=fuzz.token_set_ratio)  # batch endpoint in 3 lines
```

## Public datasets
- [Magellan/DeepMatcher benchmarks](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) —
  Amazon–Google, Abt–Buy, Walmart–Amazon labeled product pairs; the academic standard. Skim Abt-Buy once.
- [WDC Product Data Corpus](http://webdatacommons.org/largescaleproductcorpus/) — millions of offer
  pairs clustered by identifier; closest public thing to the task at scale.
- Nothing public for electrical/plumbing distributor descriptions. ETIM / UNSPSC are the industry's
  classification taxonomies (mention as the standardization layer, not usable as matching pairs).

## Interview logistics & etiquette
- Own IDE confirmed. **Working assumption: AI-off** (recruiter question still outstanding).
- 2026 consensus: ask upfront "AI-allowed or AI-off?"; hidden assistance is the one universal no;
  if allowed — narrate prompts, boilerplate only, conspicuously review the output.
- Screen-share prep: ~16pt font, notifications off, clean prompt, test a screen-share with yourself.

## Docs access during the interview (AI-off)
1. **IDE-native first:** `obj?` / `obj??` in the interactive window; hover / signature help
   (Cmd+Shift+Space inside call parens) via Pylance. No context switch — strongest look.
2. **Dash (macOS):** offline docsets — install scikit-learn, pandas, FastAPI, Python stdlib,
   rapidfuzz if available. Fuzzy search to exact page in ~2s. Alternatives: Zeal (Linux), devdocs.io (browser, offline).
3. **Fallback:** bookmarks folder `interview` with TfidfVectorizer / NearestNeighbors /
   rapidfuzz process / FastAPI testing pages. Avoid raw web search in AI-off interviews —
   AI answer-overviews on a shared screen are an awkward gray zone.
Etiquette: doc lookups are normal & expected; narrate them ("never remember if char_wb pads —
two seconds") instead of silent tab-drifting. Practice the `?` reflex during Thursday env session.

## VS Code layout & data viewing
- Interactive window can't dock in the Ctrl+` panel (editors ≠ views). Instead: drag its tab to
  the bottom edge of the editor area → full-width bottom group under the .py file. `Cmd+B` hides
  sidebar. `Cmd+K Cmd+M` (or double-click group tab bar) maximizes code group without killing
  kernel. Optional: `"jupyter.interactiveWindow.viewColumn": "secondGroup"`.
- Data Wrangler from the interactive window's ipykernel: toolbar → **Variables** view → each
  DataFrame row has an "Open in Data Wrangler" icon (attaches to live kernel var, same as
  notebooks). Note: output-renderer launch path is globally disabled in user settings
  (`dataWrangler.outputRenderer.enabled: false`) — Variables route unaffected. Variables view is
  also good demo narration ("here's my catalog frame…").

## Interactive window survival rules — FINAL DIAGNOSIS (Aug 11)
Two overlapping causes for the "kernel hangs":
1. **vscode-jupyter #17271**: with **ipykernel 7.x**, the Jupyter *Variables view* panel's silent
   `_VSCODE_getVariable` request deadlocks the execution queue → first cell runs, everything after
   hangs pending. Trigger = variables panel open. "Fixed in 7.2.0" per maintainer; reporter says
   it persists. **Workaround: pin `ipykernel<7` (uv add) + keep the Variables panel closed while
   executing** — open on demand for inspection/Data Wrangler, close after.
2. Machine overload: Dash's generate-docset-from-URL crawler leaked ~400 Safari WebContent
   renderers → load avg 92 → everything starved. Fix: cancel crawl, `killall
   com.apple.WebKit.WebContent`, reboot. Ritual: reboot day before, `uptime` < ~10 before sharing screen.
(Also: rebuilt venv must stay on pinned 3.11 — `uv sync -p 3.11`; a stray rebuild landed on 3.14
with a broken ancient ipykernel 6.3.0.)

## Interactive window survival rules (learned the hard way, Aug 10)
The kernel almost never dies on its own — VS Code's *renderer* chokes; logs showed every kernel
exit was a requested restart (Exit Code null = SIGTERM, not crash/OOM).
- **"Restart Kernel" does NOT clear rendered output.** Webview DOM bloat survives it. Hang fix:
  close the Interactive tab and reopen (disposes webview). Nuke: Developer: Reload Window.
- **Giant single-line outputs freeze the renderer** (bare `dict`/`list` reprs, e.g. `vocabulary_`).
  Line-limit truncation doesn't apply to one mega-line. Pandas/numpy reprs self-truncate → safe.
- Reflex: separate compute from display (`v = big_thing` then `len(v)`), view via `pd.Series(v).head(20)`
  / slices. Ask len before showing any unfamiliar object.
- Per-row debug prints never survive past the cell where they earned their keep (12k prints = 1 min hang).
- f-string format specs: integers are `:d` (or no spec) — `:i` is a ValueError; `%i` is printf-only.

## Tooling fixes made this week
- **zsh vi-mode trap:** `export EDITOR=vim` makes zsh silently switch to vi keybindings.
  Fixed with `bindkey -e` at end of `~/.zshrc` (keeps vim for git commit etc.).
- **Ruff in VS Code:** on-save *code actions* removed entirely (it was `source.fixAll.ruff`
  auto-deleting not-yet-used imports — F401). `formatOnSave` kept: the formatter
  (`ruff format`, Black-style) only touches whitespace/quotes/wrapping, can't delete code.
  Manual vocabulary: `ruff check .`, `ruff check --fix .`, `ruff format .`, `ruff format --diff .`.
- Pre-commit hooks: skipped — solo repo, and a hook mutating/blocking commits mid-interview is a
  gremlin risk. Instead: `alias check='ruff check . && ruff format --check . && pytest -q'`,
  run before each milestone commit (say it out loud — good demo theater).
- Dev loop: `# %%` cells in .py files + interactive window, `%autoreload 2`, pytest on one
  keystroke. `explore.py` is the only scratch file. Wrong match → parametrized failing test,
  not REPL fiddling.
