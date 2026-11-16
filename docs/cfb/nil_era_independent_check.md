# NIL / portal era — independent check of the 2026-11-16 memo

**Question:** Does NIL + free transfers change how we should audit CFB (patterns vs model inputs)?  
**This note:** Independent recompute + code review. Not a pricing change.

Sources used here (no Odds API credits; **no CFBD key in this VM**):

- SportsDataverse public releases: `cfb_team_talent`, `cfb_ratings` (`adj_net`), `espn_cfb_betting` + `cfb_schedules` / `espn_cfb_schedules`, `cfb_returning_production`
- Engine docs: `cfb_engine/data/returning.py`, `portal.py`, `scripts/cfb/stability_study.py`
- External: 247Sports talent-composite parity writeup; Yurko/Benz NESSIS 2025 portal–NIL volatility slides

CFBD `/talent`, `/ratings/sp`, and `/lines` (the memo’s primary stack) could not be re-pulled here (`CFBD_API_KEY` absent; public CFBD calls 401). Where numbers disagree, treat the memo’s CFBD figures as the better primary measurement and this note as a second instrument.

---

## Verdict

| Memo claim | Independent read |
|---|---|
| Talent has spread out / top programs less dominant | **Mostly agree on direction**; **disagree on “2024–25 = trough”** on SportsDataverse 247 composites |
| Teams less like last year; recruiting predicts less | **Agree on direction** (weaker YoY / talent→rating links post-2021) |
| Closing line as accurate as ever | **Agree** on 2021–23 vs 2024–25 (only eras with usable ESPN closes here) |
| Big-brand favs drifted toward a coin flip, not a fade | **Agree** — move is real-ish, not proven edge |
| Pattern tests → prefer 2021–25; CLV/close-EV still valid pre-NIL | **Agree strongly** |
| Returning / stability fits on mixed 2014–2025 overweight pre-NIL carryover | **Agree** — and engine already mostly disabled those terms |
| Add era split to close-test / refit returning on 2021+ walk-forward | **Agree — do it (measurement only)** |
| Drop `CFBD_API_KEY` for `_2` on Mac | Plausible; this VM has **no** CFBD key at all |

---

## 1. Talent dispersion

Memo (CFBD 247 talent): top-10 share **12.4–13.4% → 10.8–11.1%**, SD **179–198 → 149–154**, drop concentrated in 2024–25.

This check (SportsDataverse `talent_composite`, top-130 by rank):

| Year | Top-10 share | SD |
|---|---|---|
| 2015 | 12.2% | 228 |
| 2019 | 11.9% | 183 |
| 2021 | **11.3%** | **148** |
| 2023 | 11.4% | 156 |
| 2024 | 11.6% | 170 |
| 2025 | 11.8% | 186 |
| 2026 | 11.9% | 197 |

**Opinion:** Parity / compression is real (247’s own “more teams above 900” reporting matches the story). On *this* composite, the trough is **~2021–22**, then dispersion **widens again** through 2026 — not a clean “2024–25 most equal” series. Possible reasons: CFBD talent vs SDV composite scale/coverage; FCS bleed; blue-chip definition drift. I would not treat “2024–25 is the most compressed season” as settled until both series are plotted on the same CFBD pull.

---

## 2. Year-to-year predictability

Memo: SP+ YoY **0.83 → 0.76**; roster talent→SP+ **0.75 → 0.66**.

This check (`adj_net` YoY / talent→same-year `adj_net`, pooled):

| Window | YoY adj_net | Talent → adj_net |
|---|---|---|
| 2015–19 | 0.74 | 0.71 |
| 2021–25 | **0.69** | **0.66** |

Same direction, smaller SP+ gap (different rating). Returning-production SD in the SDV file also falls (2015 ~0.21 → 2025 ~0.15) while means drop — rosters look less “sticky” in the production table too.

**Nuance (Yurko/Benz 2025):** NIL-era **league-wide** between-season innovation variance can be *lower*, while **high-transfer teams** get *higher* innovation variance. So “less like last year” is partly **heterogeneous by portal volume**, not a uniform AR(1) break. Era-splitting averages is still right; a transfer-count interaction is the sharper model fix later.

---

## 3. Closing line

ESPN closes are thin before ~2023 in the public sportsdataverse betting release, so **2013–19 miss SD was not recomputed** here. On eras that *are* populated (FBS–FBS, regular season):

| Era | n | miss SD | 14+ fav cover | Power-6 fav cover | 21+ share | Weeks 1–4 SD | Week 9+ SD |
|---|---|---|---|---|---|---|---|
| 2021–23 (mostly 2023 lines) | 723 | 15.06 | 51.0% ± 3.4 | 60.5% ± 5.4 (81) | 13.4% | 13.06 | 15.85 |
| 2024–25 | 1515 | **15.16** | 47.9% ± 2.4 | **46.4% ± 4.2 (138)** | 13.7% | 15.52 | 15.14 |

**Agree:** close did not degrade into 2024–25.  
**Agree:** power-brand favorites drifted toward ~50% (here even through the coin flip); SE still too wide to sell a fade.  
**Soft agree on September:** 2024–25 early miss SD 15.52 vs late 15.14 — same sign as the memo (16.1 vs 15.1), smaller gap; still a hint.  
**External corroboration:** walk-forward models still lose to the close in 2021–25 public trackers (close winner rates ~71–77%).

---

## 4. What it means for *this* repo (code review)

Already encoded, and consistent with the memo:

| Module | State | Era handling |
|---|---|---|
| `returning.py` | Fitted **2014–2025**, **+2.5 pts/unit**, ships **OFF** (51.96% ATS, −0.8% ROI) | Mixed-era fit; memo’s concern is exactly why leaving it off is correct until a 2021+ refit wins walk-forward |
| `portal.py` | **2021–2025 only**, graded not priced; top-decile **+3.1 pts in 2024–25 vs +0.1 in 2021–23** | Already era-aware; still unproven |
| `stability_study.py` / VSiN haircut | Haircut **defaults off** — more shrink → worse MAE | Shrink/stability were pre-NIL intuitions; measurement already vetoed them |
| Market blend / close tests | CLV / close-EV are the bar | Memo is right: old closes already priced that year’s roster — **CLV validity does not require an era split** |

**What I would still do (same as the memo):**

1. **Era-split the close-test / market-weight study** (pre-2021 vs 2021+, drop 2020): set `w` from post-portal holdout first.  
2. **Refit returning production on 2021+ only**, walk-forward vs current OFF / vs 2014–2025 coefficient — expect the coefficient to shrink or die.  
3. Keep portal top-decile as a **2024–25+ graded band**, not a price.  
4. Optional later: transfer-count × YoY innovation term (Yurko), not a flat “NIL dummy” on every team.  
5. Fix Mac/env **`CFBD_API_KEY` → working key (`_2`)** so pulls don’t silently rot.

**Where I push back slightly:**

- Don’t rewrite history as “the close got easier to beat after NIL.” Evidence says the opposite.  
- Don’t promote “fade Alabama/Georgia/etc.” — 49% is not 45%.  
- Don’t treat 2013–19 pattern wins as dead *and* 2024-only portal tails as live without the same 150-game / 2 SE / both-halves bar; the memo already says this — keep it strict.  
- Talent “most compressed in 2024–25” needs a CFBD-vs-SDV reconciliation before it drives a feature.

---

## 5. Bottom line

Same conclusion as the memo on the decision that matters: **NIL reshapes model inputs and pattern out-of-sample windows; it does not create a new closing-line inefficiency.** Prefer 2021–25 for betting-pattern validation; keep full-history closes for CLV; refit returning/stability/market weight on post-portal seasons walk-forward; leave unproven portal tails graded-only.

CBB aside (Torvik already post-portal) stands.
