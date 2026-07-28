# The Hitters' Regression Report: Who's Due, Who's Cooked, Who's Just Good

### PayoffPitch · July 26, 2026 · every lineup on the slate, read through the V2 engine

Same drill as the pitcher side, flipped. A hitter's slash line is an *outcome*, and outcomes lie constantly, because a batted ball has to survive gloves, positioning, and dumb luck before it turns into a hit. What doesn't lie is the *quality of contact* — barrels, exit velo, whether the launch angle sits in the sweet spot. Those stabilize in a few dozen batted balls; the results stacked on top of them take months.

So the V2 engine keeps two ledgers on every hitter — **what he's produced** (wOBA on contact) and **what that contact deserved** (expected wOBA, or xwOBA). The gap between them is `dxwOBA`, and it's the whole game:

- **Big positive dxwOBA** (deserved ≫ produced) = the balls are dying in gloves. He's been robbed. **Due to heat up.**
- **Big negative dxwOBA** (produced ≫ deserved) = flares, seeing-eye singles, a BABIP that defies gravity. He's been gifted. **Due to cool off.**
- **dxwOBA near zero on a high xwOBA** = the production is *real*. No regression to wait for — just a good hitter you can trust tonight.

BABIP (hits per ball in play, league ≈ .290) is the tell-tale: a .150 BABIP on hard contact screams bad luck; a .440 BABIP on weak contact screams a coming crash. Here are tonight's three lists.

---

## 🟢 The Robbed: Top 10 hitters due for positive regression

*Hard, quality contact the box score hasn't paid out yet. These bats are primed to heat up.*

| # | Hitter | Tm | Slot | BBE | wOBA | xwOBA | dxwOBA | BABIP | Barrel% | HardHit% |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | Daniel Susac | SF | 9 | 60 | .166 | .343 | **+.177** | .148 | 4.0% | 16.7% |
| 2 | Jarren Duran | BOS | 7 | 160 | .248 | .368 | +.120 | .172 | 8.5% | 21.2% |
| 3 | Colton Cowser | BAL | 6 | 89 | .270 | .376 | +.105 | .256 | 4.5% | 31.5% |
| 4 | Luke Raley | SEA | 7 | 88 | .328 | .423 | +.095 | .267 | 13.3% | 26.1% |
| 5 | Taylor Trammell | HOU | 7 | 67 | .318 | .409 | +.091 | .133 | 15.2% | 25.4% |
| 6 | Julio Rodríguez | SEA | 2 | 110 | .357 | .439 | +.082 | .317 | 9.8% | 26.4% |
| 7 | Colt Emerson | SEA | 9 | 130 | .278 | .358 | +.080 | .254 | 4.9% | 15.4% |
| 8 | Casey Schmitt | SF | 5 | 193 | .392 | .461 | +.070 | .290 | 12.5% | 31.6% |
| 9 | Carlos Narváez | BOS | 9 | 64 | .336 | .404 | +.068 | .200 | 10.3% | 21.9% |
| 10 | Blaze Jordan | STL | 8 | 138 | .251 | .316 | +.065 | .236 | 5.9% | 29.0% |

**The headliner, for the second straight day, is Jarren Duran** — and he's earned the repeat. A .248 wOBA-on-contact reads like a slump; he *deserves* .368, a 120-point robbery, and the smoking gun is a **.172 BABIP**, ~120 points below league norm, on a 21% hard-hit rate. Balls hit that hard don't keep finding gloves. His hits, doubles, and total-base overs are still priced off a slump the contact quality says is already over.

**The best pure buy is Luke Raley** — a .328 that deserves .423 with a genuine **13.3% barrel rate**; the pop is real and the results just haven't caught it. **Taylor Trammell** is the extreme-luck version: a **.133 BABIP** sitting on a 15% barrel rate is almost comedic, though the 67-BBE sample keeps it directional. And **Julio Rodríguez** (.357 → .439, elite contact, an ordinary .317 BABIP) is the *sturdiest* name on the list — this is a star the model says is still being slightly underpaid, not a variance mirage. Buy the total-bases and hits before the market re-rates him.

Two notes on the tails. **Daniel Susac** tops the raw gap (+.177) but on soft contact (4% barrels, 17% hard-hit) and just 60 batted balls — treat it as "his .166 is a fluke," not "he's about to rake." And **Casey Schmitt** barely counts as luck at all: a .461 xwOBA on a perfectly normal .290 BABIP means he's simply a very good hitter the surface line finally agrees with.

---

## 🔴 The Gifted: Top 10 hitters due for negative regression

*Production built on flares, seeing-eye knocks, and unsustainable BABIPs. Fade the hot streaks.*

| # | Hitter | Tm | Slot | BBE | wOBA | xwOBA | dxwOBA | BABIP | Barrel% | HardHit% |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | Pete Crow-Armstrong | CHC | 1 | 193 | .699 | .472 | **−.227** | .438 | 11.3% | 26.9% |
| 2 | Esmerlyn Valdez | PIT | 4 | 124 | .747 | .567 | −.179 | .429 | 27.1% | 29.0% |
| 3 | Nasim Nuñez | WSH | 9 | 150 | .452 | .291 | −.161 | .446 | 1.4% | 13.3% |
| 4 | Nick Gonzales | PIT | 6 | 200 | .481 | .331 | −.150 | .404 | 3.0% | 21.0% |
| 5 | Eugenio Suárez | CIN | 4 | 136 | .558 | .411 | −.147 | .280 | 17.2% | 22.8% |
| 6 | Pedro Ramírez | CHC | 8 | 71 | .438 | .296 | −.141 | .381 | 0.0% | 23.9% |
| 7 | Austin Hedges | CLE | 8 | 65 | .464 | .327 | −.136 | .306 | 10.8% | 21.5% |
| 8 | Caleb Durbin | BOS | 5 | 206 | .412 | .277 | −.135 | .292 | 2.0% | 16.5% |
| 9 | Kyle Karros | COL | 5 | 163 | .518 | .386 | −.133 | .354 | 11.0% | 26.4% |
| 10 | Paul Goldschmidt | NYY | 4 | 160 | .442 | .311 | −.130 | .265 | 10.8% | 17.5% |

The cleanest fades are the **empty-BABIP guys**: **Nasim Nuñez** (a **.446 BABIP** on a 1.4% barrel rate — pure fairy dust), **Nick Gonzales** (.404 BABIP, 3% barrels), **Caleb Durbin** (2% barrels, .412 wOBA that deserves .277), and **Pedro Ramírez** (a *0.0%* barrel rate under a .381 BABIP). None of these bats hit the ball hard; they're stringing bloops together, and the engine drags them toward league-average or below. Anything priced off their recent line is a sell.

**Two nuance cases you shouldn't over-fade.** **Pete Crow-Armstrong**'s −.227 is the biggest correction on the board, but a .472 xwOBA is *still* excellent — he regresses from video-game (.699, .438 BABIP) down to very good, not to bad; fade the batting-average/hits props, respect the profile. **Esmerlyn Valdez** is the same shape one tier up: an absurd .747 cooling toward a **.567** that's carried by a monster 27% barrel rate — from insane to elite. Pay neither at peak price.

**Eugenio Suárez** is the reverse-shaped fade you see every week: he genuinely barrels it (17.2%), but a .558 on a .280 BABIP is HR-or-bust variance the model prices at .411 — **fade the hits/average, respect the power.** And file **Paul Goldschmidt**: a .265 BABIP isn't even inflated, yet his contact (17.5% hard-hit) is soft enough that the engine still sees .311 — an aging-bat fade on the contact markets.

---

## ⚪ The Real Deal: Top 10 most stable, highest-xwOBA bats

*Production and contact quality in lockstep — no luck to bleed off. These are the trustworthy mashers tonight.*

| # | Hitter | Tm | Slot | BBE | wOBA | xwOBA | dxwOBA | BABIP | Barrel% | HardHit% |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | James Wood | WSH | 1 | 174 | .501 | .529 | +.027 | .302 | 15.4% | 34.5% |
| 2 | Nick Kurtz | ATH | 2 | 116 | .490 | .518 | +.027 | .326 | 17.4% | 27.6% |
| 3 | Yordan Alvarez | HOU | 2 | 179 | .495 | .506 | +.011 | .299 | 19.4% | 31.3% |
| 4 | Brandon Nimmo | TEX | 4 | 174 | .461 | .477 | +.016 | .376 | 14.8% | 29.3% |
| 5 | Heliot Ramos | SF | 3 | 124 | .444 | .474 | +.030 | .300 | 18.5% | 31.5% |
| 6 | Ben Rice | NYY | 2 | 184 | .457 | .461 | +.005 | .238 | 16.8% | 29.9% |
| 7 | Freddie Freeman | LAD | 3 | 227 | .460 | .456 | −.004 | .386 | 7.6% | 24.7% |
| 8 | Jac Caglianone | KC | 4 | 183 | .425 | .450 | +.025 | .244 | 14.0% | 31.7% |
| 9 | Mike Trout | LAA | 2 | 83 | .476 | .447 | −.029 | .341 | 11.9% | 18.1% |
| 10 | Jake Bauers | MIL | 5 | 138 | .438 | .446 | +.009 | .312 | 11.8% | 28.3% |

This is the list you *don't* wait on — the production is earned. **James Wood** tops it again: a .529 xwOBA on a 15% barrel / 35% hard-hit rate with a perfectly ordinary .302 BABIP, so none of it is luck. **Yordan Alvarez** (a slate-best **19.4% barrel rate**), **Nick Kurtz** (17.4% barrels), and **Heliot Ramos** (18.5%) are the other bats whose contact is so loud there's simply nothing to regress. These are your anchors for total-bases and HR props — no discount, but no cliff either.

The value wrinkle inside a "stable" list: **Ben Rice** and **Jac Caglianone** carry *low* BABIPs (.238, .244) under big barrel rates, so their small positive dxwOBA actually leans up — the rare "stable *and* slightly underpaid." **Freddie Freeman**, by contrast, rides a .386 BABIP but backs it with a matching xwOBA, so trust the bat without expecting more. And note **Mike Trout**'s slightly negative dxwOBA on soft-ish hard-hit — great hitter, just don't chase the peak on an 83-BBE sample.

---

## The one-screen cheat sheet

- **Buy the dip (heat up):** Jarren Duran, Luke Raley, Julio Rodríguez, Colton Cowser, Taylor Trammell — hard contact, buried BABIPs. Attack hits/singles/total-base overs.
- **Buy the star anyway:** Julio Rodríguez, Casey Schmitt — good-to-elite *and* not luck-inflated.
- **Fade the mirage (cool off):** Nasim Nuñez, Nick Gonzales, Caleb Durbin, Pedro Ramírez — empty BABIP, soft contact. Sell hits/average props.
- **Fade the peak, not the player:** Pete Crow-Armstrong, Esmerlyn Valdez — regressing from insane to (very) good.
- **Fade the hits, respect the power:** Eugenio Suárez, Paul Goldschmidt.
- **Trust, no discount:** James Wood, Yordan Alvarez, Nick Kurtz, Heliot Ramos — anchor bats for TB/HR.

*The fine print: wOBA/xwOBA here are measured **on contact** (batted balls only), so dxwOBA isolates ball-in-play luck; strikeout-rate swings are handled separately on the pitcher side. Season-to-date Statcast through the engine's rolling windows, minimum 60 batted balls to make a list — small-sample names (Susac, Trammell, Narváez, Hedges, Ramírez, Trout, all under ~90 BBE) are directional. This is a regression read, not a betting card — line it up against tonight's prices before you fire.*
