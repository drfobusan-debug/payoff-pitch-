# The Regression Report: Which Arms Are Lying to You Tonight

### PayoffPitch · July 26, 2026 · 15 games, 30 starters, read through the V2 engine

Every box score tells a story. The problem is that half of them are fiction. A pitcher's ERA, his strikeout total, the three hits he "allowed" — those are outcomes, and outcomes are noisy. What we actually want is the *truth underneath*: how nasty was the stuff, how sharp was the command, how hard was the contact. Those stabilize fast and lie a lot less. When the results and the truth disagree, that gap has a name — **regression** — and it's the single most bettable edge in baseball, because the market prices the box score and the box score is often lying.

So here's the whole slate, run through the V2 engine's regression model, translated out of math and into English. Three questions for every arm: **Are the strikeouts real? Is the soft contact real? Has he been lucky or robbed?** Let's go find the liars.

---

## The whiff kings the strikeout column hasn't caught up to

Start with the best edge on the board, because it's a two-for-one: **Cristopher Sánchez**. His strikeout rate reads a pedestrian **22%**. Now look under the hood — a **30.2% CSW** and a **16.7% swinging-strike rate**, both firmly ace-tier. The engine's expected K% pegs him at **36%**, a 14-point chasm, the biggest positive strikeout gap on the slate. And there's a bonus: hitters are running a **.370 BABIP** against him (dxwOBA −.077), so the contact he *does* allow has been landing luckily for the offense — his hit multiplier sits at **0.90**. Sánchez is the rare arm where the strikeouts project *up* and the run prevention projects *up* at the same time. If his K prop is priced off that 22%, you're getting the 36% version at a discount.

**Jacob deGrom** is the same movie for a guy who's already elite. A 33% strikeout rate looks great and the engine says it's *still* light — a **32.9% CSW**, an **18.4% swinging-strike rate**, and an expected K% of **42%** with the strikeout multiplier pinned at **1.27**. He's also carrying a .333 BABIP (dxwOBA −.035), so like Sánchez the hits should recede too. You're not buying a bounce-back; you're buying a great pitcher the model refuses to fade.

**Jacob Misiorowski** rounds out the top tier with the filthiest raw stuff on the board — a **34.6% CSW** and a **19.9% swinging-strike rate** behind a 38% K rate that the engine still nudges up to 42%. **José Soriano** (23% K on a 30.7% CSW → expected **32%**) is the sneaky one: a groundball arm whose swing-and-miss is quietly running well ahead of his strikeout line.

---

## The strikeout rates borrowing against the future

Now the other side of the K ledger — arms getting *outs*, but whose *strikeouts* are a loan the stuff can't repay.

**Freddy Peralta** is the headliner, and it's a genuinely split verdict. His 20% strikeout rate is built on a soft **23.3% CSW** and a 10.7% swinging-strike rate that translate to an expected K% of just **9%** — an 11-point overhang, the largest negative strikeout gap tonight. *Do not pay for the strikeout over.* But here's the twist: he's also been mugged on contact — a **.350 BABIP** and a wOBA-allowed of .486 against an xwOBA of just **.328** (dxwOBA −.158), so his hit and home-run multipliers sit at **0.91 / 0.85**. Fade the Ks, buy the run prevention. Two truths, one pitcher.

**Janson Junk** (on the mound tonight vs San Diego) fits his name: a 20% K rate on a **25.2% CSW** and a 6.9% swinging-strike rate that the engine reads as an expected **9%**. The supporting cast of "modest K numbers still running ahead of the stuff": **Connor Prielipp** (26% → 19%), **Kyle Leahy** (23% → 16%), **Shane Baz** (25% → 19%), and **Framber Valdez** (19% → 16%), who remains a great pitcher — this is a "trim the strikeout expectation," not a "fade the man." Small-sample flags: **Jameson Taillon** (75 pitches) and **Kohl Drake** (85) post the flashiest negative gaps but on far too little data — directional only.

---

## The soft-contact magicians whose trick is about to fail

This is where the money usually hides — a shiny hit-prevention line built on luck is the most overpriced thing in the sport.

Tonight's headliner is **Parker Messick** (starting for Cleveland at Tampa Bay). His hit prevention has been excellent and it's largely a mirage: opponents are hitting just **.218 on balls in play** against him, ~70 points below league norm, on a tiny 3.9% barrel rate that won't keep suppressing everything. The engine's hit multiplier is **1.06** — it expects the singles and doubles to trickle back. He's not a bad pitcher; his hit-prevention line is just running hot.

**Carson Whisenhunt** is the loudest signal (dxwOBA **+.080**, a .281 wOBA that deserves .361) but on 172 pitches — treat it as directional. **Kyle Leahy** doubles up here (.220 BABIP, hit multiplier 1.07) on top of his shaky strikeout profile. And the nuance case of the night is **Logan Gilbert**: his strikeouts project *up* (31% → **40%**, elite whiff signals), but his **.250 BABIP** and +.055 dxwOBA mean his *hits allowed* project up too (multiplier 1.09). So the honest read is buy the strikeout over, fade the hits-allowed under — two different bets on the same arm.

---

## The unlucky arms the box score is slandering

Finally, the fun ones — pitchers whose lines look like a dumpster fire that the engine wants to buy.

**Kevin Gausman** (starting at Boston) is the cleanest buy-low. A .481 wOBA-allowed looks alarming until you see the **.393 BABIP** underneath it — ~100 points above normal — with an xwOBA of .439 and a negative dxwOBA. His hit and home-run multipliers sit at **0.89 / 0.92**: the engine sees clear positive regression coming. The results have been worse than the pitching.

**Shane Baz** (.343 BABIP, a tiny **2.9% barrel rate**, hit/HR multipliers 0.91 / 0.85) and **Framber Valdez** (.343 BABIP, 5% barrels, HR multiplier **0.855**) are the same story — genuinely good arms whose contact quality says the runs should dry up. And the two-way names carry over from the top of the page: **Cristopher Sánchez** (BABIP .370, mult 0.90) and **Jacob deGrom** (BABIP .333) both belong here too — the rare arms projecting *more strikeouts and fewer hits* at once. **Freddy Peralta** is the extreme version: his run-prevention line (xwOBA .328 vs .486 shown) is wildly unlucky even as his strikeouts fade.

---

## The one-screen cheat sheet

- **Lean strikeout OVERs:** Cristopher Sánchez, Jacob deGrom, Jacob Misiorowski, José Soriano, Logan Gilbert.
- **Fade strikeout OVERs:** Freddy Peralta, Janson Junk, Connor Prielipp, Kyle Leahy, Shane Baz, Framber Valdez.
- **Attack the offense against:** Parker Messick, Carson Whisenhunt, Logan Gilbert (hits) — soft-contact/low-BABIP mirages due to correct upward.
- **Buy-low run prevention:** Kevin Gausman, Shane Baz, Framber Valdez, Cristopher Sánchez, Jacob deGrom, Freddy Peralta.
- **Split verdicts:** Sánchez & deGrom (Ks up *and* hits down), Peralta (fade the Ks, buy the run prevention), Gilbert (buy the Ks, fade the hits-under).

*A word on what this is and isn't: this is a **skill-vs-results regression read**, not a betting card. It tells you which numbers are honest, not what price to lay. Season-to-date Statcast through the engine's rolling windows; small-sample arms (Taillon, Kohl Drake, Whisenhunt, Prielipp — all under ~300 pitches) are directional. Pair every lean with tonight's actual line before you fire.*
