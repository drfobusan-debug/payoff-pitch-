# CFB Engine — Betting Strategy Masterplan

**Source audit:** `PayoffPitch_CFB_Weekly_Audit_2026-11-14` (week Nov 8–14; ledger through that week, 44 slates, 475 buys).  
**Operating identity:** automated AI bettor / handicapper with an **$8,000** bankroll. Goal order is (1) **keep the bankroll**, (2) **grow it**.  
**Rule of evidence (already coded):** judge on **CLV / EV at the close**, not ROI. Profit ahead of what the close pays is labeled variance by `cfb_engine/audit/weekly.py`. Act only when `probation.py` says SHIP / PROMOTE / SHUT (n≥100, >1 SE, both halves agree) unless noted as a pre-registered tighten.

Labels: **[MEASURED]** = number from this audit; **[CODE]** = already in the engine; **[PLAN]** = proposed change.

---

## 0. One-line verdict

The ledger is **+26.9u / +5.8% ROI [MEASURED]** but the average buy is still worth **−2.5¢ at the close** against **−3.0¢ juice at bet time [MEASURED]**. The close moved toward the bets (+0.5¢, beat close 53%, +0.25 pp) — not enough to pay the juice. **This week's +7.2u is variance.** An $8k bot that bets the whole card is gambling on luck; the same bot that stakes only close-confirmed ATS dog slices is trying to compound a real edge.

---

## 0b. AI bettor operating manual ($8,000)

### Mission

Survive first. The audit says the **average** buy still loses to the close after juice. That means flat-betting every Strong/Moderate row is how an automated book dies a slow death while the weekly email looks green. Growth comes only from the segments where price and outcome agree.

### Bankroll math

| Parameter | Value | Why |
|---|---|---|
| Bankroll \(B\) | **$8,000** | Given |
| Base unit | **1% = $80** | Standard preservation unit; resize only at ±20% bankroll marks |
| Half unit | **$40** | Secondary ATS dogs |
| Max single stake | **1.5u = $120** | ~⅛ Kelly on the soft ATS-15+ win-rate estimate; never full Kelly |
| Max slate risk | **5% = $400** | One bad Saturday (like Nov 1–7, −22.6u ledger-scale) cannot cripple $8k |
| Max live bets / Saturday | **6** | Forces ranking; card may list more — bot only funds the top Kelly-ordered dogs that clear gates |
| Correlated cap | **3** ATS dogs that share conference or clearly move together | Same-slate correlation is why full Kelly is ruin |

**Kelly note.** At ~−110 with a hoped-for ~64% cover rate (ATS 15+ sample), raw Kelly is ~20%+ of bankroll per bet. That number assumes the sample *is* the true edge. It is not proven (needs ~66). The bot therefore uses **flat 1% with a 1.5u ceiling**, not model Kelly dollars. `cfb_engine.market.ordering` already ranks the card on Kelly **[CODE]** — use that for *order*, not for *size*.

### What the bot bets (funded)

| Priority | Selection | Stake | Gate |
|---|---|---|---|
| 1 | **ATS dog, line ≥ +14.5** | 1.0u ($80); 1.5u if Strong *and* drift ≥ +2 pp toward us | edge ≥ 3%, no adverse drift ≥ 2 pp, not graded-only |
| 2 | **ATS dog, \|line\| outside 4–7.5** | 0.5u ($40) | same gates; favorites ATS default skip |
| 3 | **Moneyline** | 0.5u only | sharp gate pass, price ≤ +200, edge ≥ 3%; skip if slate already at $400 risk |
| — | Totals / EV≤3% / mid 4–7.5 spreads / alt lines / parlays | **$0** | price & grade only |

### What the bot never does

- Chase a hot week (Nov 8–14 +14.4% is variance **[MEASURED]**).
- “Fill the card” when few rows clear — sitting at $0 risk is a valid Saturday.
- Raise unit size after a winning week; only after **B ≥ $9,600** (+20%), reset unit to 1% of new bankroll.
- Bet into adverse pre-bet drift (gate already live **[CODE]**).
- Recycle losers by buying/selling points (spread alts −6% to −12% **[MEASURED]**).

### Kill switches (capital preservation)

| Trigger | Bot action |
|---|---|
| Bankroll **−15% from peak** (first trip: $6,800) | Primary ATS 15+ only, 0.5u, max slate 3% |
| Bankroll **−20% from start** ($6,400) | Pause live stakes one full week; still run card + audit |
| Last **50 buys** close-EV **< −4¢** | Freeze new market types; only Priority-1 |
| Probation **SHUT** on a market | That market → graded-only same day |
| Pass rows **winning beyond noise** | Raise `min_edge` another point; do not add bets |
| One slate loss **> 8%** of bankroll | Next slate max risk 3% |

### Expected pace (honesty check)

- ~1–4 funded bets most Saturdays under these gates (not 50).
- At ~+10% ROI on a *selected* book (beat-close cohort), 3 × $80 / week ≈ $24/week expected — **~$1.2k over a season if the edge is real**, with large path variance.
- A Nov 1–7 style wipeout at old volume would have been ~$1.8k on $80 units; the caps exist so that week costs **≤ $400**.

### Decision loop (every slate)

1. Run engine card; take buys already past live gates.  
2. Drop totals, EV≤3%, \|ATS\|∈[4,7.5], ATS favorites, ML failures.  
3. Sort remainder by Kelly conviction **[CODE]**.  
4. Fund Priority-1 then 2 then 3 until $400 slate cap.  
5. Log stake tier next to `predictions_*.json` (manual or future `stake` field **[PLAN]**).  
6. Next morning: audit CLV first; ignore the unit P/L for strategy changes.

---

## 1. What counts as a "real edge bet"

| Test | Pass means | Fail means |
|---|---|---|
| EV at the close ≥ juice paid (or clearly less negative) | Price taken was good vs final market | Lucky outcomes on bad prices |
| Beat-close rate meaningfully > 50% with volume | Market moved to the side | Coin-flip CLV |
| Segment ROI 95% range excludes 0, or near it with CLV support | Beyond noise | Hot week / selection luck |
| Pass rows do *not* win | Filter is selecting | Gate deleting winners |

**This audit's own call on the whole book:** profit runs ahead of close-justified EV → **likely variance [MEASURED]**. So "winning segments" below are ranked by **close confirmation first**, ROI second.

---

## 2. Non-variance winners (the real edge bets)

Ordered by how much the close agrees, not by units.

### 2.1 Primary edge — **ATS large dogs (spread 15+)**

| Metric | Value |
|---|---|
| Record | 37–21–1 |
| Units / ROI | **+13.5u / +23.3%** |
| Beat close | **58%** |
| Needs ~N for ROI proof | ~66 (closest ATS band to proven) |
| Audit label | Strongest ledger segment |

**Why it is edge, not variance:** best beat-close rate among large ATS bands; unbet dogs in this zone lose on the candidate study that registered `ats_upgrade_dog_14.5_plus` **[CODE]** (`cfb_engine/market/atsline.py`). The upgrade already accrues in probation; do not hand-promote — wait for PROMOTE, then make Strong-by-default or raise stake rank.

### 2.2 Definitional edge — **buys that beat the close**

| Metric | Value |
|---|---|
| Record | 145–101–3 (n=249) |
| Units / ROI | **+24.7u / +10.0%** |
| Contrast | Matched/lost close: +3.2u / +1.5% on 225 |

Almost all lasting profit sits in the beat-close half. Matched/lost-to-close profitability is **pure variance [MEASURED]** — outcomes without price confirmation. Strategy work should maximize *probability of beating the close*, not maximize weekly units.

### 2.3 Secondary edge — **ATS underdogs (all sizes)**

| Metric | Value |
|---|---|
| Record | 84–64–6 |
| Units / ROI | **+15.3u / +10.4%** |
| Beat close | 53% |
| Needs ~N | ~338 |

Favorites ATS: 51–47–2, **+1.1% ROI**, beat close 49% — noise. Road-dog / home-dog upgrades are already candidate-graded **[CODE]**; keep grading, do not ship from this audit alone.

### 2.4 Selection edge — **the Pass filter**

Pass rows (not bet): 1,685–1,748–42, **−202.4u / −5.9% ROI**, **losing beyond noise [MEASURED]**. The buy bar is correctly refusing a losing population. Do not loosen `min_edge` / `min_ev` to "get more action."

### 2.5 Near-edge / conditional — **EV 5%+ bands**

| Band | Record | ROI | Beat close |
|---|---|---|---|
| EV 5–8% | 65–47–2 | +9.9% | 48% |
| EV over 8% | 71–57–1 | +11.3% | 51% |
| EV up to 3% | 73–54–2 | **−0.9%** | 58% |
| EV 3–5% | 56–44–3 | +2.5% | 52% |

Thin EV clears juice at bet time more often than it clears the *close* (EV≤3% even beats the close 58% but still loses money — juice). **[PLAN]** Raise the effective buy floor toward ~3–5% EV *or* raise `min_edge` so the ≤3% band stops shipping. Prefer edge (price-independent) over EV floor — see `tiers.py` lesson **[CODE]**.

### 2.6 Already correctly sidelined — **totals**

Totals: 38–40, **−4.6u / −5.9%**, Over and Under both −5.9%. Already `graded_only` for `game_total` **[CODE]** (`gradedonly.py`). Keep until `LIFT` on `graded_only_game_total`. Alternate-total *bought* rungs look soft-positive but underpowered (SE ~12–19%); do not reopen the main total from alt ladder noise.

### 2.7 Explicitly *not* real edge this week

- Week Nov 8–14: 31–19–1, +7.2u, +14.4% — audit: **underpowered + profit ahead of close = variance**.
- Week ML 15–6 / Strong buys 22–12–1 — small-n, week-local.
- Oct 25–31 "winning beyond noise" vs Nov 1–7 "losing beyond noise" — weekly ROI is not a strategy signal.

---

## 3. Where the book leaks (cut or probation)

| Leak | Ledger | Close / note | Action |
|---|---|---|---|
| **ATS spread 4–7.5** | 28–34–5, **−7.5u (−12.1%)** | Beat close 52%; weakest ATS band | Candidate refuse already covers mid-fav −7…−3.5 and mid-dog +7.5…+14 **[CODE]**. **[PLAN]** Register full abs-line 4–7.5 as a candidate screen; ship only on SHIP. |
| **ATS favorites** | +1.1% ROI | Beat close 49% | Prefer dogs; no hard ban until probation. |
| **EV ≤ 3%** | −0.9% ROI | Beats close often but does not pay | Raise floor **[PLAN]**. |
| **Totals** | −5.9% | Graded only | Keep shut. |
| **Spread alt buy/sell** | Bought −6.1%, sold −11.7% (by game SE still huge) | Main number not improved by shopping points | Do not auto-buy alts; keep graded-only ladder. |
| **Model Brier** | ML/spread/total all lose to market | Close also beats or ties model | Raise `CFBE_MARKET_BLEND` / enable `CFBE_MARKET_ANCHOR` study **[PLAN]**. |
| **Moderate buys** | +3.5% vs Strong +7.4% | Both ~52–53% beat close | Optional `CFBE_STRONG_ONLY` A/B after EV floor raise. |

---

## 4. Strategy refinement — masterplan phases

Principles carried from existing CFB design (do not reopen without evidence):

1. Rank on **edge**, not EV (`tiers.py`).
2. Judge on **CLV**, not weekly ROI (`weekly.py`, `clv.py`).
3. Screens stamp `pass_gate` and grade refusals (`probation.py`).
4. Measured-null features stay OFF (efficiency blend 0, returning 0, wind 0, VSiN HFA off, mark bumps off).

### Phase A — Immediate (config / no model rewrite)

Pre-registered, reversible, quote this audit in the commit message.

| # | Change | Lever | Rationale |
|---|---|---|---|
| A1 | **Raise buy quality** | `CFBE_MIN_EDGE` 0.02 → **0.03** (or per-market `CFBE_MIN_EDGE_GAME_ATS=0.03`, ML leave 0.02 pending sharp gate) | Kill EV≤3% population that loses after juice **[MEASURED]**. |
| A2 | **Keep totals graded-only** | no change to `CFBE_GRADED_ONLY_MARKETS` | Confirmed −5.9% **[MEASURED]**. |
| A3 | **Do not chase this week's ML heater** | no ML reopen / no Strong-only from week ROI | Week CLV does not clear juice. |
| A4 | **Keep adverse drift veto** | `CFBE_DRIFT_GATE` on | Already CFB-ledger justified **[CODE]**; continue grading. |
| A5 | **Keep ML +200 long-dog band + sharp gate** | `priceband` / `mlsharp` | Still the right priors while ML Brier loses to market. |

### Phase B — Candidate screens / upgrades (code, act only on probation)

| # | Register or tighten | Module | Ship when |
|---|---|---|---|
| B1 | **`ats_refuse_abs_4_to_7.5`** — refuse ATS buys with \|line\| ∈ [4, 7.5] | extend `atsline.py` + `CANDIDATE_SCREENS` | SHIP |
| B2 | **Promote big dogs** — `ats_upgrade_dog_14.5_plus` already registered | `atsline.BIG_DOG_RULE` | PROMOTE → Strong bump or stake weight |
| B3 | Road/home dog upgrades | already registered | PROMOTE only |
| B4 | **`ev_refuse_under_3pct`** as candidate if A1 not taken as live | new candidate | SHIP |
| B5 | Mid-fav / mid-dog candidates already live | `MID_FAV_RULE`, `MID_DOG_RULE` | leave; may be subsumed by B1 |
| B6 | Totals line-agree / YPP / unit rules | already graded-only rules | PROMOTE before any totals buy |

### Phase C — Forecast quality (model, to earn CLV)

The market wins Brier on ML (0.170 vs 0.176), spread (0.249 vs 0.253), total (0.250 vs 0.254) **[MEASURED]**. Selection can harvest CLV for a while; long-run edge needs a better probability.

| # | Work | Success metric |
|---|---|---|
| C1 | Sweep `CFBE_MARKET_BLEND` (0.35 → 0.45–0.55) on holdout Brier *and* forward CLV | Model Brier ≤ market on ≥2 markets; buy close-EV ≥ −juice |
| C2 | Trial `CFBE_MARKET_ANCHOR` > 0 on the EV screen probability (not the sim means) | Beat-close % on buys ↑ without Pass rows flipping to winners |
| C3 | Re-fit `margin_sd` / `total_sd` / ML market-SD curve against 2026 closes | Dog ML lean not systematically long |
| C4 | Leave efficiency / returning / weather / mark bumps OFF until residual vs close is positive | Partial-r / CLV study, not ATS win% |

### Phase D — Staking & card presentation ($8k bot)

| # | Change | Note |
|---|---|---|
| D1 | Order by **Kelly**, size by **§0b ladder** (not raw Kelly $) | Ranking **[CODE]**; sizing is bankroll policy |
| D2 | Persist `stake_units` / `stake_usd` on funded rows | Audit can separate “card buy” vs “bot funded” |
| D3 | Enforce slate cap $400 and correlated-dog cap 3 | Soft card note first; hard gate in a thin wrapper script **[PLAN]** |
| D4 | No alternate-line auto-execution | Alts stay graded; main number only |
| D5 | Weekly article: bankroll mark-to-market + kill-switch state | Beside CLV summary |

### Phase E — Stop-losses (probation-driven)

| Trigger | Action |
|---|---|
| Market SHUT (buys losing, n≥100, both halves) | Set market graded-only with verdict in commit |
| Screen LIFT (refusals winning) | Disable gate, keep measuring |
| Candidate SHIP | Enable gate with `pass_gate` name = candidate name |
| Upgrade PROMOTE | Apply Strong bump for that rule only |
| Another "losing beyond noise" week *and* close-EV worse than juice | Freeze Phase A loosens; no new buys from new features |

---

## 5. Target portfolio (what the $8k bot funds)

After Phases A–B, **funded** bets only:

1. **ATS dogs ≥ +14.5** — 1.0–1.5u (§0b).  
2. **ATS other dogs** outside the 4–7.5 dead zone — 0.5u.  
3. **Moneyline** — 0.5u, sharp + price band, only if slate room remains.  
4. **Totals / ATS favorites / thin EV / mid spreads** — $0 (still priced for the ledger).

That portfolio is the intersection of (beat-close profit) ∩ (strongest ledger segments) ∩ (Pass rows correctly losing) ∩ (**bankroll caps**).

---

## 6. Success scoreboard (pre-register)

Do not declare the strategy "refined" on ROI. Track weekly:

| Metric | Baseline (this audit) | Goal (next ~100–150 buys) |
|---|---|---|
| Buy EV at close | −2.5¢ | ≥ −1.0¢ and ≥ juice improvement of +1.0¢ |
| Beat close % | 53% | ≥ 55% |
| ATS 15+ share of ATS buys | ~23% (59/256) | Rising without Pass-row ROI flipping positive |
| EV≤3% share of buys | ~27% (129/475) | Near 0 after A1 |
| Model vs market Brier (ML/ATS/total) | market wins 3/3 | model wins ≥1, gap shrinks on others |
| Pass rows ROI | −5.9% beyond noise | Stay ≤ −3% (filter still working) |
| Probation actionable | (run `cfb-engine probation`) | Only ship/lift from verdicts |

---

## 7. Explicit non-goals

- Do not retune from Nov 8–14's +14.4% or the ML 15–6.  
- Do not reopen totals from one week of model Brier wins (week model beat market; ledger market still wins).  
- Do not buy spread alts because bought rungs are "less bad" than sold.  
- Do not enable returning production / efficiency blend / weather from descriptive cover rates.  
- Do not lower `min_edge` to fill the card.

---

## 8. Implementation checklist (for the next coding PR)

1. Config: A1 min-edge raise (+ tests that ≤3% EV / thin edge → Pass).  
2. `atsline.py`: add `abs_mid_band(4, 7.5)` helper + candidate screen B1; wire into `CANDIDATE_SCREENS`.  
3. **Bankroll bot stub:** given `B=8000`, emit a `stakes_<date>.json` from card buys applying §0b ladder + caps (dry-run default).  
4. Weekly report: Phase B candidate statuses + bankroll mark / kill-switch state.  
5. Script or notebook: blend/anchor sweep C1–C2 offline (no live Odds spend).  
6. After each Saturday audit: probation actionable lines in the weekly email (already supported).  
7. Only then: if `ats_upgrade_dog_14.5_plus` → PROMOTE, raise that slice to 1.5u default.

---

## 9. Audit snapshot (numbers cited above)

**Week Nov 8–14:** 51 buys, 31–19–1, +7.2u, +14.4%; close EV −3.1¢ vs juice −3.8¢; beat close 51%.  
**Ledger:** 475 buys, 265–202–8, +26.9u, +5.8%; close EV −2.5¢ vs juice −3.0¢; beat close 53%, CLV +0.25 pp.  
**Strongest / weakest (ledger):** ATS 15+ +23.3%; ATS 4–7.5 −12.1%.  
**Only week called winning beyond noise:** Oct 25–31. **Losing beyond noise:** Nov 1–7.  
**Pass (ledger):** −5.9%, losing beyond noise.

*Model audit plan, not investment advice. All live changes go through probation or a dated config commit that cites this document.*
