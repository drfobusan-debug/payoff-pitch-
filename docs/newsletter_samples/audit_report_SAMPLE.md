# PayoffPitch Engine — Audit Report

### Daily · slate graded 2026-07-23

---

## Executive summary

Across every graded market, the side the model favored won **54.2%** of the time, for a **+1.9% ROI** on the whole book — profitable overall. Its sharpest work is in Pitcher outs, Pitcher strikeouts, Batter singles. The losses concentrate in First-5 run line, Batter H+R+RBI combo, First-5 total, and the pattern below is consistent: the engine is a solid handicapper whose leaks are a too-loose bet-selection filter and a slightly cold offensive model — both fixable without touching what already works.

---

## Core metrics

| Scope | n | PPV (pick win%) | NPV | ROI |
|---|---|---|---|---|
| **Whole engine** (favored side) | 201 | **54.2%** | 0.72 | **+1.9%** |
| Strong buys | 12 | 8.3% | — | -83.2% |
| Moderate buys | 2 | 50.0% | — | +3.0% |

---

## Market scorecard

Every graded market rated on PPV / NPV / ROI, sorted highest-to-lowest return. **🟢 Play** = profitable and above breakeven; **🟡 Neutral** = no usable edge yet (or the model correctly abstains) — wait for more data; **🔴 Fade** = losing money, do not bet until fixed. *Min p to Play* is the minimum model probability a selection must clear before the engine fires a bet in that market.

| Market | PPV | NPV | ROI | Min p to Play | Verdict |
|---|---|---|---|---|---|
| Pitcher outs | 0.80 | 0.47 | +52.8% | 0.58 | 🟢 Play |
| First-5 moneyline | 0.75 | 0.78 | +43.2% | 0.62 | 🟡 Neutral |
| Pitcher strikeouts | 0.73 | 1.00 | +38.9% | 0.58 | 🟢 Play |
| Batter singles | 0.64 | 0.54 | +21.2% | 0.58 | 🟢 Play |
| Batter hits | 0.61 | 0.78 | +12.5% | 0.58 | 🟢 Play |
| Game moneyline | 0.60 | 0.60 | +5.4% | 0.58 | 🟢 Play |
| Game run line | 0.50 | 0.50 | -3.2% | 0.62 | 🟡 Neutral |
| Pitcher hits allowed | 0.50 | 0.79 | -4.5% | 0.62 | 🟡 Neutral |
| Game total | 0.50 | 0.50 | -4.5% | 0.62 | 🟡 Neutral |
| First-5 run line | 0.40 | 0.40 | -23.6% | avoid | 🔴 Fade |
| Batter H+R+RBI combo | 0.35 | 0.62 | -33.1% | avoid | 🔴 Fade |
| First-5 total | 0.30 | 0.30 | -42.7% | avoid | 🔴 Fade |
| Batter runs | 0.00 | 0.62 | -100.0% | 0.62 | 🟡 Neutral |
| Pitcher walks | 0.00 | 0.58 | -100.0% | 0.62 | 🟡 Neutral |
| Batter doubles | — | 0.87 | ~0.0% | — | 🟡 Neutral |
| Batter home runs | — | 0.90 | ~0.0% | — | 🟡 Neutral |
| Batter RBI | — | 0.73 | ~0.0% | — | 🟡 Neutral |
| Batter total bases | — | 0.77 | ~0.0% | — | 🟡 Neutral |
| Pitcher earned runs | — | 0.85 | ~0.0% | — | 🟡 Neutral |

**Definitions —** **PPV (Positive Predictive Value):** of the sides the model *backs*, the share that actually win. **NPV (Negative Predictive Value):** of the sides the model *fades*, the share that actually lose. *(Diagnostic accuracy terms — NPV here is **not** the financial 'Net Present Value'.)*

---

## Most common errors

- **Coin-flips dressed as strong bets.** Strong/Moderate buys won only 14.3% (n=14) — below the 52.4% breakeven. Plus-money prices are promoting near-toss-ups to 'buys' on EV alone.
- **Selling starters short.** Faded 'outs' overs still won 53.3% (n=15) — efficient starters are pitching deeper than the model expects.
- **Money-losing pockets.** First-5 run line, Batter H+R+RBI combo, First-5 total are currently below breakeven and should be sat out until the fixes above ship.

---

## Recommendations

*Each action is mapped to the goal it serves:* **↑ PPV · ↑ NPV · ✕ eliminate false positives · ↓ reduce false negatives.**

> **1 — Add a conviction floor to bet selection — require model probability ≥ ~0.58 *in addition to* positive EV before tagging any play a Moderate/Strong buy.**
> → **✕ eliminate false positives · ↑ PPV**

> **2 — Loosen starter-length limits for efficient arms — raise the outs projection for low-pitch-per-batter starters.**
> → **↓ reduce false negatives · ↑ NPV**

> **3 — Gate or drop the red markets (First-5 run line, Batter H+R+RBI combo, First-5 total) — do not bet them until their PPV recovers.**
> → **✕ eliminate false positives · ↑ PPV**

> **4 — Shrink thin edges toward the market — when the market prices a side ≥ ~.57 but the model is under .50, blend toward the market instead of fully fading it.**
> → **↓ reduce false negatives · ↑ NPV**

> **5 — Leave the green markets alone — Pitcher outs, Pitcher strikeouts, Batter singles, Batter hits, Game moneyline are the model's strengths; keep them and size up where conviction is real.**
> → **protects existing PPV**

### What to play and fade right now

Based on this audit, until the fixes above ship *(verdicts match the scorecard)*:

- **🟢 Play:** Pitcher outs, Pitcher strikeouts, Batter singles, Batter hits, Game moneyline — each only when the selection clears the **0.58** conviction floor.
- **🔴 Fade / avoid:** First-5 run line, Batter H+R+RBI combo, First-5 total — bleeding money right now.
- **🟡 Neutral (wait for data):** First-5 moneyline, Game run line, Pitcher hits allowed, Game total, Batter runs, Pitcher walks, Batter doubles, Batter home runs, Batter RBI, Batter total bases, Pitcher earned runs — coin-flips or markets the model abstains from; require a higher **0.62** bar before betting.

*Sample note: this report covers 1 graded day — treat per-market figures as directional; the accumulating ledger firms them up over time.*