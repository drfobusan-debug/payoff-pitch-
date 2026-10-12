# CBB engine — Year-1 scope sheet (2026-27)

One page for builders. `master_plan.md` is the reasoning; this is what ships. Frozen before the
first slate (~Nov 3); changes only at the fixed 4th-Monday review by the rules in
`cbb/params/preregistered.yaml`.

## Ships (live, can be bought)

| Piece | Spec |
|---|---|
| Capture | Odds API `basketball_ncaab`, DK + BetMGM execution, all-book median excl. execution books; tip-scaled per-event cadence 60/30/15 min; **featured markets every 10 min inside T-3h (3 credits/pass)**; October futures once; written daily credit budget |
| Market-only channel | soft vs sharp/consensus, stale-line flag, limit-adjusted CLV, fill quality, `max_per_book_per_day` |
| v0 distribution | consensus **devigged median** (margin, total) · SD of **`final − close` residuals** per market-quality bucket (bucket defined from pre-close information only) · key-number push + parity masses fitted **separately**, smoother for the rest · margin-conditional foul bump and OT as **redistributions only** · halves = correlated pair, 1H share + correlation from residuals conditional on the close |
| Anchor solve | possessions fixed (Torvik tempo), two efficiency levels solved so stage 1 + stage 2 + OT reproduce the median; sim median within **0.1 pt** after stage 2 / OT or `anchor_miss` refusal; CRN; seed = `(event_id, model_version)`; path count s.t. SE < ⅓ tier edge, **capped**, cap hit ⇒ `se_cap` refusal |
| Price band | live: totals juice band per bucket; dog-odds cap applies to research rows only |
| Blend | location space: `w_m`, `w_t` per quality bucket, hierarchical, **prior at 0**, leaves 0 only past MDE; missing Torvik ⇒ no blend; ESPN rating report-only; stale flag never in `p_model`; rating mean = **Torvik `super_sked` projection** (admitted only after the as-of leakage check) |
| `p_model` | one distribution object; coherence tests every pass (monotone alt lines, team totals sum, 1H + 2H = game, P(win) ≥ P(cover) for favourites); edge > `k` × sampling spread, `k = 1.0` |
| Markets bought | `status=live`: game totals only; team totals / 1H / alts `shadow`; sides / ML `research` |
| Availability | tri-state overrides; points-lost-per-player, high-majors only; Beta priors per status × tier (daily conjugate update); shift applied **after the solve, only the un-moved consensus residual**; box-vs-market race report-only until its event count + MDE; `AMBIGUOUS` ⇒ variance multiplier; `pending` = no consensus move > ½ pt / 5¢ across two featured passes; `starter_unknown` strict below high-major |
| Gates | thin market, starter_unknown, pending, drift (measured), price band per quality, `anchor_miss`, `se_cap`, `halves_sum` / `team_totals_sum` / `monotone_alt` / `win_vs_cover`; every refusal ledgered and graded |
| Row grain | `(event_id, market, side, line, book, pass_id)`; `status` live/shadow/research; `pass_kind` board/featured/close; `bet_of_record` = card pass; close passes archive-only; `stale_flag` own column; fields per plan §4.a table |
| Cadence | 60/30/15 board by time-to-tip · 10-min featured inside T-3h · 5-min watch ESPN/RotoWire only, no credits |
| 03:00 report | DuckDB views `v_clv v_shape v_rung v_gate v_persist v_prior` over the row grain; PDF renders them; page-one lights first, W-L/ROI/PPV beneath with base rate + break-even; main-line Brier a coherence column |
| Ledger + audit | write-once decision snapshot per row (provenance / decision / context / execution); `cbb-engine placed` entry + reconciliation; ESPN official finals/halves; append-only corrections; consensus + execution CLV; slate-day cluster bootstrap; post-game tags (`tags.yaml`), PIT + surprise score; luck-adjusted P&L, edge-bin calibration, tier monotonicity, component attribution; page-one scorecard; time-travel test; hash-chained ledger; 4th-Monday review; kill switch = futility (CLV CI upper < +0.5¢ at min n); drawdown pause in SDs + CLV deterioration |
| Storage | git: ledger/params/manifests/audit only; quote archive: per-pass Parquet shards, daily manifest hash in git, encrypted off-site (rclone crypt); DuckDB query layer; chain head anchored weekly (OpenTimestamps) |
| Market-only verdict | hypothesis #1 judged on fill-verified rows or with a fill haircut; opportunity-persistence light; pre-launch ~10 dense historical slates (on go-ahead) for CLV SD and stale persistence |
| Futures prior | five shadow candidates (all-zero, 0.15 / 0.30 / 0.45, mid-halved-for-brand), scored at week 5, all-zero may win; **no futures weight on a live price before week 5** |
| Featured-pass value | 10-min pass CLV vs 15-min subset CLV on the same games (audit cut 10); fails MDE by 4th Monday ⇒ back to 15 min, reserve released |
| Delivery | plain card 11:30 / 16:30 + email; audit md+PDF 03:00 |

## Pinned-mean rules (frozen)
| Rule | Setting |
|---|---|
| Width | `final − close` residual SD per pre-close-defined quality bucket; key-number push/parity masses fitted separately |
| Solve | possessions fixed (Torvik tempo), two efficiencies solved; stage 2 / OT redistribute only; sim median within 0.1 pt or `anchor_miss` refusal |
| Paths | cap in `preregistered.yaml`; SE still > ⅓ edge at cap ⇒ `se_cap` refusal |
| Blend | `w_m`, `w_t` prior at 0 per bucket; leaves 0 only past MDE; missing Torvik ⇒ no blend |
| Availability | shift applied after the solve, only the un-moved residual; high-majors only |
| Week-5 prior | five candidates incl. all-zero futures weight |

## Shadow (priced, ledgered as `shadow`, graded, never bought)

| Piece | Leaves shadow when |
|---|---|
| Team totals, 1H totals, alt lines (≤ ±1 SD) | ≥ 600 derivative closes archived **and** the derivative-consistency study (book leg + outcomes leg) passes its pre-registered rule at a fixed review |
| Inner rungs above v0: margin–total correlation, full stage-2 foul sub-model, diffusion score model | beats the rung below by its published log-score / CRPS MDE, date-clustered CI, Holm/FDR across rungs |
| Situational terms (one ridge/horseshoe model on close residuals) | term's MDE ≤ plausible effect **and** survives on pace/variance first |
| Sides / ML | outer-ladder rung beats `close alone` on consensus CLV ≥ 0.5¢ and Brier ≥ 0.0010 |
| ML rungs (§3.8): logistic fill-probability + recalibration; GBM/GAM shape targets; close-movement forecaster | shadow beside a linear baseline; outputs to distribution object; never a bet selector |
| Shadow priors (5 variants incl. all-zero) | scored at week 5 on `v_prior`; all-zero stays unless a positive weight beats it by its MDE; written once to `prior_weights.json` |
| ESPN box-score fallback rating vs Torvik | report only |

## Deferred (Phase 1b — not built in year 1 unless `w > 0` is worth testing)

`prior.py` (portal / NIL / coach change), own Kalman vs joint-ridge ratings, player posteriors,
minutes model, both-lineups-rebuilt mixture, 1H carry from the rotation model, worksheets,
totals sheet, MP3, long PDF, props.

## Frozen vs adaptive (see §4.z)

Frozen: structure, rung status, hyperparameters, `w_m`/`w_t`, `k`, tiers, gates, MDEs,
consensus definition. Adaptive (declared rule, stamped as-of): Beta status priors, `AMBIGUOUS`
dispersion refit, data inputs, credit allocation.

## Phase 0b go/no-go (before ~Nov 3)

credit budget written + 80-game Saturday simulated · gate-funnel sim run · D7 / D11 / D12 /
`super_sked` / Pinnacle-eu verified or fallback active · both emails sent from the dry run ·
write-once + time-travel green · `preregistered.yaml` committed with MDEs (from the pre-launch CLV-SD sample) and `k` · `tags.yaml` + `registry.yaml` committed · this sheet
matches the code. **Any fail ⇒ opening week is research mode.**
