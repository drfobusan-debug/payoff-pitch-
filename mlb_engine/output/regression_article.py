"""The regression report written as an article rather than a stat card.

Same inputs as the Mound Report and the Batter Regression Watch — the starter
and hitter profiles from ``mlb_engine.output.regression_profiles`` — rendered
as one document in prose: arms first, then bats. ``mlb-engine run`` builds it
from the previews and predictions it has just written, so the daily card and
this article always describe the same slate.

The spine of the writing is a distinction the card layout blurs. A pitcher has
a *level* (SIERA, Stuff, velocity: who he is) and a *luck* term (BABIP-against
and the wOBA-minus-xwOBA gap: what the results have added on top). Only the
second one is due to move. Measured on 246 starts in the 2026 window, the level
predicts what a starter allows in his next outing (SIERA t +4.0, Stuff t -3.8,
vFA t -2.4 against next-start xwOBA, holding across a chronological split)
while the three-week *direction* of those same metrics does not — both trends
that reach significance point the wrong way. So the article states levels as
fact, the luck term as the thing about to change, and the three-week arrows as
context that is explicitly labelled unproven.
"""

from __future__ import annotations

import math
from datetime import date as Date

import pandas as pd

from mlb_engine.features.regression import (
    BL_BABIP,
    BL_FB_ALLOWED,
    BL_K_PCT,
    BL_XSLG,
)
from mlb_engine.features.swing import (
    CONFIRMED,
    CONTRADICTED,
    LEAGUE,
    UNMEASURED,
    WINDOW,
)
from mlb_engine.output.audit_insight import to_pdf
from mlb_engine.output.regression_profiles import (
    _batter_ctx,
    _batter_id_map,
    _best_batter_bet,
    _bets_for,
    _pitcher_id_map,
    build_batter_profiles,
    build_profiles,
)

BL_VFA = 93.8  # league mean four-seam/sinker velocity, 2026 window
BUY_TIERS = ("Strong buy", "Moderate buy")


def _mil(v: float) -> str:
    """.312 -> '.312' the way a box score writes it."""
    return f".{int(round(v * 1000)):03d}"


def _who_he_is(p: dict) -> str:
    siera = p["siera"]
    if siera < 3.40:
        return "a front-line arm"
    if siera < 4.30:
        return "a mid-rotation arm"
    return "a back-end arm"


def _stuff_phrase(p: dict) -> str:
    xk = p["xk"]
    if xk > BL_K_PCT + 0.04:
        return "swing-and-miss stuff"
    if xk < BL_K_PCT - 0.04:
        return "contact-manager stuff"
    return "average stuff"


def _velo_phrase(p: dict) -> str:
    vfa = p["vfa"]
    if vfa != vfa:  # NaN
        return ""
    if vfa >= BL_VFA + 1.5:
        return f"a genuine fastball at {vfa:.1f}"
    if vfa <= BL_VFA - 1.5:
        return f"a fringe fastball at {vfa:.1f}"
    return f"a fastball at {vfa:.1f}"


def _air_sentence(p: dict, positive: bool) -> str:
    """Which way the correction arrives: over the fence, or on the ground.

    Batted-ball shape does not say whether a starter is due to improve -- the
    luck term does that -- but it says what the improvement is made of, and the
    reader is pricing home-run props off exactly that distinction.
    """
    fb = p.get("fb", float("nan"))
    if fb != fb:  # NaN
        return ""
    if fb >= BL_FB_ALLOWED + 0.05:
        shape = (
            f"He is a fly-ball arm: {fb:.0%} of the contact he allows goes in "
            f"the air against a {BL_FB_ALLOWED:.0%} norm"
        )
        return shape + (
            ", so the correction arrives over the fence &mdash; the home runs are "
            "the first thing to thin out, and the strikeout props move least."
            if positive
            else ", which is where a run of good luck ends fastest: one "
            "carrying night and the flies that were dying start landing."
        )
    gb = p.get("gb", float("nan"))
    if fb <= BL_FB_ALLOWED - 0.05:
        ground = f"{gb:.0%} of his contact" if gb == gb else "most of his contact"
        return (
            f"He keeps the ball down &mdash; {ground} on the ground, {fb:.0%} "
            "in the air &mdash; so "
            + (
                "expect the correction in singles and double plays rather than "
                "in home runs."
                if positive
                else "the damage that comes back is hits rather than homers."
            )
        )
    return f"His batted-ball shape is ordinary ({fb:.0%} fly balls)."


def _bat_air_sentence(p: dict, positive: bool) -> str:
    """What the hitter's correction is made of, read off his batted-ball shape."""
    fb = p.get("fb", float("nan"))
    if fb != fb:  # NaN
        return ""
    iffb = p.get("iffb", float("nan"))
    if fb >= BL_FB_ALLOWED + 0.05:
        out = f"The shape is airborne: {fb:.0%} of his batted balls are fly balls"
        if iffb == iffb and iffb > 0.25:
            return (
                out + f", but {iffb:.0%} of them are pop-ups, and that is the "
                "share of his air contact that was never going to pay."
            )
        return out + (
            ", which is where the extra-base version of the correction lives."
            if positive
            else ", so the fall comes out of his power rather than his average."
        )
    gb = p.get("gb", float("nan"))
    if fb <= BL_FB_ALLOWED - 0.05:
        ground = f" ({gb:.0%} grounders)" if gb == gb else ""
        return (
            f"He hits the ball on the ground{ground} with only {fb:.0%} in the "
            "air, so read this as singles and total bases rather than as home runs."
        )
    return ""


def _luck_sentence(p: dict, positive: bool) -> str:
    babip, gap = p["babip"], p["dxwoba"] * 1000
    bits = []
    if positive:
        if p["unlucky_babip"] > 0.015:
            bits.append(
                f"balls in play are dropping at {_mil(babip)} against him where "
                f"{_mil(BL_BABIP)} is the league norm, and that number has no "
                "staying power"
            )
        if gap < -10:
            bits.append(
                f"the damage on the scoreboard ({_mil(p['woba'])} wOBA) is "
                f"{abs(gap):.0f} points worse than the contact he has actually "
                f"allowed ({_mil(p['xwoba'])} xwOBA)"
            )
        if not bits:
            bits.append("the luck against him is mild but it is against him")
        return "Where he has been unlucky: " + "; ".join(bits) + "."
    if p["unlucky_babip"] < -0.015:
        bits.append(
            f"balls in play are finding gloves at {_mil(babip)} against a "
            f"{_mil(BL_BABIP)} norm"
        )
    if gap > 10:
        bits.append(
            f"the scoreboard ({_mil(p['woba'])} wOBA) flatters the contact he "
            f"has given up ({_mil(p['xwoba'])} xwOBA) by {gap:.0f} points"
        )
    if not bits:
        bits.append("the luck has been mildly in his favour")
    return "Where he has been helped: " + "; ".join(bits) + "."


def _pitcher_verdict(p: dict, positive: bool) -> str:
    """The line that separates the correction from the level it corrects to."""
    declining = p["d_siera"] > 0.20 or p["d_vfa"] < -0.5
    improving = p["d_siera"] < -0.20 and p["d_vfa"] > 0
    if positive and declining:
        moves = []
        if p["d_siera"] > 0.20:
            moves.append(f"his SIERA is {p['d_siera']:.2f} higher")
        if p["d_vfa"] < -0.5:
            moves.append(f"the fastball is {abs(p['d_vfa']):.1f} mph slower")
        return (
            "Read this one carefully. The runs should come down, but the last "
            "three weeks look worse than the six before them — "
            + " and ".join(moves)
            + " — so the level he corrects back to may not be the one he "
            "pitched at earlier in the year. Buy the correction, not the name."
        )
    if positive and improving:
        return (
            "This is the clean version: nothing underneath is breaking, the "
            "results simply have not caught up yet."
        )
    if positive:
        return (
            "Nothing underneath is moving much, which is what you want — the "
            "gap is luck rather than decline."
        )
    if not positive and improving:
        return (
            "The results should get worse, but he is throwing better than he "
            "was, so treat this as a soft fade rather than a bet against him."
        )
    if not positive and declining:
        return (
            "Both halves point the same way here: the luck is due to turn and "
            "the underlying arm is going backwards. The strongest fade shape "
            "on the board."
        )
    return "Expect more hard contact to leak through as the run of luck ends."


def _today_sentence(ctx: dict | None) -> str:
    if ctx is None:
        return ""
    opp_x = ctx["opp_lineup_xwoba"]
    if opp_x < 0.310:
        who = f"a soft {ctx['opp']} lineup"
    elif opp_x > 0.335:
        who = f"a dangerous {ctx['opp']} lineup"
    else:
        who = f"an average {ctx['opp']} lineup"
    env = ""
    pf, wx = ctx.get("park_factor"), ctx.get("wx_hr_mult")
    park = None
    if pf is not None:
        park = "a hitter's park" if pf > 101 else "a pitcher's park" if pf < 99 else None
    air = None
    if wx is not None:
        air = "carrying air" if wx >= 1.03 else "dead air" if wx <= 0.97 else None
    both = [x for x in (park, air) if x]
    if both:
        env = ", in " + " and ".join(both)
    return (
        f"Tonight he draws {who} ({_mil(opp_x)} xwOBA){env}; the sim lands the "
        f"total near {ctx['total_mean']:.1f}."
    )


def _bet_sentence(bets: list[dict], whose: str) -> str:
    buys = [b for b in bets if b["tier"] in BUY_TIERS]
    if not buys:
        return f"<span class='nobet'>The model passes {whose} props today.</span>"
    parts = []
    for b in buys[:3]:
        odds = "" if b.get("market_american") is None else f" at {b['market_american']:+.0f}"
        parts.append(
            f"<b>{b['selection']}</b>{odds} <span class='pct'>"
            f"(model {b['model_prob'] * 100:.0f}%, {b['tier'].lower()})</span>"
        )
    lead = "The bet that follows: " if len(parts) == 1 else "The bets that follow: "
    return lead + "; ".join(parts) + "."


def _pitcher_entry(p: dict, ctx: dict | None, bets: list[dict], positive: bool) -> str:
    matchup = ctx["matchup"] if ctx else ""
    trend = (
        f"3wk trend: SIERA {p['d_siera']:+.2f} · Stuff xK% {p['d_xk'] * 100:+.1f} · "
        f"vFA {p['d_vfa']:+.1f} mph"
    )
    opener = (
        f"{p['name']} is {_who_he_is(p)} — SIERA {p['siera']:.2f}, "
        f"{_stuff_phrase(p)} at {p['xk'] * 100:.0f}% expected strikeouts"
    )
    velo = _velo_phrase(p)
    opener += f", {velo}." if velo else "."
    body = " ".join(
        x
        for x in (
            opener,
            _luck_sentence(p, positive),
            _air_sentence(p, positive),
            _pitcher_verdict(p, positive),
            _today_sentence(ctx),
        )
        if x
    )
    whose = p["name"].split()[-1] + "&rsquo;s"
    cls = "up" if positive else "down"
    return (
        f"<div class='entry {cls}'>"
        f"<h3>{p['name']} <span class='mu'>{matchup}</span></h3>"
        f"<p class='prose'>{body}</p>"
        f"<p class='trend'>{trend} <span class='caption'>&mdash; shown for context; "
        "three-week direction does not predict the next start</span></p>"
        f"<p class='bet'>{_bet_sentence(bets, whose)}</p>"
        "</div>"
    )


def _swing_sentence(p: dict, positive: bool) -> str:
    """Stage two: does the swing agree with what the gap says is coming?

    The gap ranks the list and it is a residual of outcomes -- it knows which
    balls fell in and nothing about the swing that hit them. Bat speed and blast
    rate, read on their own windows of tracked swings, add to next-fortnight total
    bases and home runs on top of wOBA and xwOBA (t +5.4 and t +6.6 on 3,175
    batter-windows), so they are what confirms or contradicts the gap here.
    """
    stage2 = p.get("stage2", UNMEASURED)
    if stage2 == UNMEASURED:
        return (
            "His swing is not readable at this sample &mdash; too few tracked competitive "
            "swings for bat speed and blast rate to mean anything &mdash; so the gap above "
            "stands on its own."
        )
    pz = p["power_z"]
    strength = (
        f"bat speed {p['bat_speed']:.1f} mph and a {p['blast'] * 100:.0f}% blast rate, "
        f"{pz:+.2f} standard deviations from league"
    )
    if positive and stage2 == CONFIRMED:
        return (
            f"The swing underneath agrees: {strength}. The rebound has a bat behind it "
            "rather than being a wish about batted balls."
        )
    if positive:
        return (
            f"The swing does not back it: {strength}. His results are below his contact and "
            "his swing is below league too, so the low level is closer to what he is than "
            "the shortfall suggests."
        )
    if stage2 == CONFIRMED:
        return (
            f"The swing agrees with the fade: {strength}. Nothing in how he is hitting the "
            "ball is holding the production up."
        )
    return (
        f"The swing argues against the fade: {strength}. He has been lucky and he is also "
        "good, which is the case the gap on its own gets wrong &mdash; of the hitters this "
        "cut flags, the better-swinging half went on to out-produce the ones it kept."
    )


def _lift_sentence(p: dict) -> str:
    """Which market the swing's steepness points at, when it is readable.

    Attack angle survives bat speed and blast rate on the home-run line (t +6.3)
    and is signed against hits (t -5.3), so it says where a hitter's production
    should land rather than how good he is.
    """
    lz = p.get("lift_z", math.nan)
    if lz != lz:
        return ""
    angle = p["attack_angle"]
    if lz >= 0.5:
        return (
            f" He swings up {angle:.1f}&deg;, {lz:+.2f} SD steeper than league, which is the "
            "home-run and total-base side of the board rather than the hits one."
        )
    if lz <= -0.5:
        return (
            f" His attack angle is {angle:.1f}&deg;, {lz:+.2f} SD flatter than league: the "
            "hits and H+R+RBI markets read better for him than the power ones."
        )
    return f" His attack angle is league-typical at {angle:.1f}&deg;, so it points at no market."


def _swing_line(p: dict) -> str:
    """The swing levels themselves, each over the window its measure repeats at."""
    if p.get("stage2", UNMEASURED) == UNMEASURED and p.get("swings", 0) == 0:
        return ""
    cells = [
        f"BatSpd {p['bat_speed']:.1f}" if p["bat_speed"] == p["bat_speed"] else "BatSpd &mdash;",
        f"Fast {p['fast'] * 100:.0f}%" if p["fast"] == p["fast"] else "Fast &mdash;",
        f"SqUp {p['squared_up'] * 100:.0f}%"
        if p["squared_up"] == p["squared_up"]
        else "SqUp &mdash;",
        f"Blast {p['blast'] * 100:.0f}%" if p["blast"] == p["blast"] else "Blast &mdash;",
        f"SwLen {p['swing_length']:.2f}"
        if p["swing_length"] == p["swing_length"]
        else "SwLen &mdash;",
        f"AtkAng {p['attack_angle']:.1f}&deg;"
        if p.get("attack_angle", math.nan) == p.get("attack_angle", math.nan)
        else "AtkAng &mdash;",
    ]
    lg = (
        f"league {LEAGUE['bat_speed'][0]:.1f} / {LEAGUE['fast'][0] * 100:.0f}% / "
        f"{LEAGUE['squared_up'][0] * 100:.0f}% / {LEAGUE['blast'][0] * 100:.0f}% / "
        f"{LEAGUE['swing_length'][0]:.2f} / {LEAGUE['attack_angle'][0]:.1f}&deg;"
    )
    return (
        f"<p class='trend'>Swing: {' &middot; '.join(cells)} "
        f"<span class='caption'>&mdash; {lg}; read over "
        f"{WINDOW['bat_speed']}/{WINDOW['fast']}/{WINDOW['squared_up']}/"
        f"{WINDOW['blast']}/{WINDOW['swing_length']}/{WINDOW['attack_angle']} tracked swings, "
        f"four times the sample each first half-repeats at, off {p['swings']} in the window. "
        "Attack angle comes from Savant's swing-path feed, which starts in 2025, so a frame "
        "cached before it reads as unmeasured.</span></p>"
    )


def _batter_entry(p: dict, ctx: dict | None, bet: dict | None, positive: bool) -> str:
    gap = p["dxwoba"] * 1000
    power = (
        "real power"
        if p["xslg"] > BL_XSLG + 0.03
        else "little power"
        if p["xslg"] < BL_XSLG - 0.03
        else "average power"
    )
    trend_d = p["woba3"] - p["woba6"]
    if positive:
        lead = (
            f"{p['name']} has been hitting the ball better than the box score "
            f"says: {_mil(p['xwoba'])} expected wOBA against {_mil(p['woba'])} "
            f"actual, a {gap:.0f}-point shortfall, with {power} underneath "
            f"({_mil(p['xslg'])} xSLG) and barrels on "
            f"{p['barrel'] * 100:.0f}% of his contact."
        )
        verdict = (
            "That gap is the part that moves. Nothing about the contact says "
            "he is finished; the hits simply have not landed yet."
        )
    else:
        lead = (
            f"{p['name']} has been getting more out of his contact than it "
            f"deserves: {_mil(p['woba'])} actual wOBA against {_mil(p['xwoba'])} "
            f"expected, {abs(gap):.0f} points of it unearned, on {power} "
            f"({_mil(p['xslg'])} xSLG) and barrels on "
            f"{p['barrel'] * 100:.0f}% of his contact."
        )
        verdict = (
            "Some of those hits are fool's gold. The bat is not necessarily "
            "cold — the results are simply ahead of it, and that is the part "
            "that comes back."
        )
    heat = (
        f"His three-week line reads {_mil(p['woba3'])} against the six-week "
        f"{_mil(p['woba6'])}, which is context and not a forecast."
        if abs(trend_d) > 0.020
        else ""
    )
    env = ""
    if ctx:
        bits = []
        pf = ctx.get("park_factor")
        if pf is not None and pf > 101:
            bits.append("a hitter's park")
        elif pf is not None and pf < 99:
            bits.append("a pitcher's park")
        if ctx.get("wx_hr_mult") is not None and ctx["wx_hr_mult"] >= 1.03:
            bits.append("carrying air")
        if bits:
            env = "Tonight: " + " and ".join(bits) + "."
    body = " ".join(
        x
        for x in (
            lead,
            verdict,
            _swing_sentence(p, positive) + _lift_sentence(p),
            _bat_air_sentence(p, positive),
            heat,
            env,
        )
        if x
    )
    bets = [bet] if bet else []
    cls = "up" if positive else "down"
    flag = ""
    if p.get("stage2") == CONTRADICTED:
        flag = " <span class='mu'>swing disagrees</span>"
    return (
        f"<div class='entry {cls}'>"
        f"<h3>{p['name']} <span class='mu'>{ctx['matchup'] if ctx else ''}</span>{flag}</h3>"
        f"<p class='prose'>{body}</p>"
        f"{_swing_line(p)}"
        f"<p class='bet'>{_bet_sentence(bets, 'his')}</p>"
        "</div>"
    )


LEAD = (
    "Two different questions get confused every time somebody says a player is "
    "&ldquo;due to regress&rdquo;, so this report separates them.<br><br>"
    "<b>What is about to change</b> is luck: balls in play falling at a rate "
    "nobody sustains, or results running ahead of the contact underneath them. "
    "For pitchers that is BABIP-against and the wOBA&minus;xwOBA gap; for "
    "hitters it is the same gap read the other way. That is what ranks this "
    "list.<br><br>"
    "<b>What it changes back to</b> is level — SIERA, Stuff, velocity for an "
    "arm; expected slugging and barrel rate for a bat. A pitcher can be a "
    "positive-regression candidate and a worse pitcher than he was last month "
    "at the same time: the runs come down, but they come down to a lower "
    "level. Those cases are flagged in the text rather than left for you to "
    "spot.<br><br>"
    "<b>The bats are read in two stages this month.</b> The gap still ranks "
    "them, because it is what is due to move. But a gap is a residual of "
    "outcomes &mdash; it knows which balls fell in and nothing about the swing "
    "that hit them &mdash; so every hitter is then crossed against his bat "
    "tracking: bat speed, fast-swing rate, squared-up rate, blast rate and attack "
    "angle, each "
    "read over its own window of tracked competitive swings rather than a round "
    "six weeks. Out of time on 3,175 batter-windows those levels add to the next "
    "fortnight&rsquo;s total bases and home runs on top of wOBA and xwOBA, blast "
    "rate contributing more than the two of them explain between them; "
    "squared-up rate predicts hits and is negatively signed on home runs, so it "
    "is read on the contact markets and kept off the power ones. Attack angle "
    "splits the same way and harder: a steeper swing adds home runs (t +6.3 with "
    "bat speed and blast rate already in the model) and subtracts singles "
    "(t &minus;5.3), monotonically from 2.6% to 4.6% HR/PA across its quintiles, "
    "so it is read on the line in question rather than as a verdict on the bat. "
    "When the swing "
    "disagrees with the gap the entry says so, and that disagreement is the "
    "report&rsquo;s answer to a hitter being written off for a fortnight of good "
    "luck.<br><br>"
    "Two honest caveats on the arrows. Measured over 246 starts this season, "
    "the <i>level</i> of SIERA, Stuff and velocity predicts what a starter "
    "allows next time out; the <i>three-week direction</i> of those same "
    "metrics does not. The trend line is printed because it is worth seeing, "
    "not because it has earned a bet. The same holds for the swing: the "
    "<i>level</i> of bat speed and blast rate forecasts, the recent-versus-prior "
    "<i>move</i> in them forecasts nothing (bat speed t +1.4, blast t &minus;0.3), "
    "so no swing trend is printed at all."
)

CSS = """
@page { size: A4; margin: 1.5cm 1.7cm 1.7cm; }
* { box-sizing: border-box; }
body{font-family:Georgia,'Times New Roman',serif;color:#1c1c1c;line-height:1.62;
     font-size:11pt;margin:0;}
.masthead{border-bottom:3px solid #16324f;padding-bottom:9px;margin-bottom:12px;}
.brand{font-size:11pt;letter-spacing:2.5px;color:#c8102e;font-weight:bold;
       text-transform:uppercase;}
h1{font-size:23pt;margin:5px 0 2px;color:#16324f;font-weight:normal;}
.sub{font-size:11.5pt;color:#4a5568;font-style:italic;margin:0 0 5px;}
.dateline{font-size:8.6pt;letter-spacing:1.6px;text-transform:uppercase;color:#6b7280;}
.lead{font-size:10.6pt;color:#25303c;background:#f6f8fa;border-left:3px solid #16324f;
      padding:11px 14px;margin:0 0 18px;line-height:1.55;}
h2{font-size:15pt;color:#16324f;font-weight:normal;border-bottom:1px solid #d7dbe0;
   padding-bottom:4px;margin:24px 0 2px;}
h2 .rk{font-size:8.8pt;color:#6b7280;font-style:italic;}
h3{font-size:12.4pt;color:#111;margin:14px 0 1px;font-weight:bold;}
h3 .mu{font-size:9pt;color:#6b7280;font-weight:normal;font-style:italic;
       letter-spacing:.5px;}
.entry{page-break-inside:avoid;padding-left:11px;border-left:3px solid #cfd6dd;
       margin:12px 0;}
.entry.up{border-left-color:#2e7d32;} .entry.down{border-left-color:#b23b3b;}
.prose{margin:3px 0 6px;text-align:justify;}
.trend{font-family:'DejaVu Sans',Helvetica,sans-serif;font-size:8.4pt;color:#5b6470;
       background:#f2f5f8;border-radius:3px;padding:3px 8px;margin:5px 0;}
.trend .caption{color:#8b949e;font-style:italic;}
.bet{font-family:'DejaVu Sans',Helvetica,sans-serif;font-size:9pt;color:#15612b;
     margin:4px 0 2px;}
.bet b{color:#0f4a20;} .bet .pct{color:#6b7280;}
.nobet{color:#8b949e;font-style:italic;}
.section-intro{font-size:10.4pt;color:#43505f;font-style:italic;margin:4px 0 2px;}
.fine{font-size:7.8pt;color:#9aa0a8;font-family:'DejaVu Sans',sans-serif;
      border-top:1px solid #e6e8ec;margin-top:20px;padding-top:7px;line-height:1.4;}
"""

FINE = (
    "Methodology: pitcher SIERA, Stuff (an xK% proxy fitted on CSW% and SwStr%) "
    "and the rate stats come from each arm's trailing six-week Statcast slice; "
    "vFA is mean four-seam/sinker velocity over the last three weeks, and the "
    "three-week deltas compare that window with everything before it. Fly-ball "
    "rate counts fly balls and pop-ups as a share of batted balls (36% is the "
    "league share); for hitters it is what ground balls and line drives leave "
    "behind, and it is read as the <i>shape</i> of a correction rather than as "
    "a reason to expect one. Hitter "
    "wOBA, xwOBA and xSLG use a six-week batted-ball slice with a 25-event "
    "minimum, and the three-week figure is the same measure over the recent "
    "window. Swing levels are means over each measure&rsquo;s own window of "
    "tracked competitive swings &mdash; "
    f"{WINDOW['bat_speed']} for bat speed, {WINDOW['fast']} for fast-swing rate, "
    f"{WINDOW['swing_length']} for swing length, {WINDOW['blast']} for blast rate, "
    f"{WINDOW['squared_up']} for squared-up rate, {WINDOW['attack_angle']} for attack "
    "angle, four times "
    "the sample at which each first reaches split-half r=.50 on 515,417 tracked "
    "swings &mdash; and are compared with league measured the same way. "
    "Squared-up and blast rate are reconstructed from the pitch-level collision "
    "model with the cuts calibrated to Savant&rsquo;s published league rates; per "
    "hitter that reads r +.86 and +.76 against the official leaderboard over the "
    "same dates (bat speed +.996, swing length +.997), and the reconstruction is "
    "noisier than the official figure, which attenuates the coefficients rather "
    "than inflating them. Attack angle, attack direction and swing tilt are "
    "Savant&rsquo;s own pitch-level swing-path fields, which begin in 2025; ours "
    "reproduce FanGraphs&rsquo; published season figures at r +.996 and +.9995 "
    "over 438 hitters, and a slice cached before the fields were ingested reads "
    "as unmeasured rather than as league average. Direction and tilt add nothing "
    "out of time and are not scored. "
    "Ranking is the luck term only: z(BABIP &minus; .290) + "
    "z(wOBA &minus; xwOBA) for arms, xwOBA &minus; wOBA for bats; the swing is a "
    "second stage that confirms or contradicts that ranking rather than "
    "reordering it. Levels were "
    "validated against next-start xwOBA allowed on 246 starts (SIERA t +4.0, "
    "Stuff t &minus;3.8, vFA t &minus;2.4, each holding sign across a "
    "chronological split); the three-week trends were not significant in the "
    "right direction and are printed as context only. Model preview, not "
    "investment advice."
)


def build_html(
    day: Date,
    ppos: list,
    pneg: list,
    pctxs: dict,
    bpos: list,
    bneg: list,
    bctxs: dict,
    preds: list[dict],
) -> str:
    nice = day.strftime("%A, %B %-d, %Y")
    pmap = _pitcher_id_map(preds)
    bmap = _batter_id_map(preds)

    def arms(title: str, intro: str, rows: list, positive: bool) -> str:
        entries = "".join(
            _pitcher_entry(
                p,
                pctxs.get(p["name"]),
                _bets_for(pmap.get(p["name"], -1), preds),
                positive,
            )
            for p in rows
        )
        return (
            f"<h2>{title} <span class='rk'>most &rarr; least</span></h2>"
            f"<p class='section-intro'>{intro}</p>{entries}"
        )

    def bats(title: str, intro: str, rows: list, positive: bool) -> str:
        entries = "".join(
            _batter_entry(
                p,
                bctxs.get(p["name"]),
                _best_batter_bet(bmap.get(p["name"], -1), preds),
                positive,
            )
            for p in rows
        )
        return (
            f"<h2>{title} <span class='rk'>most &rarr; least</span></h2>"
            f"<p class='section-intro'>{intro}</p>{entries}"
        )

    body = (
        "<h2 class='part'>Part one &mdash; the arms</h2>"
        + arms(
            "Due to improve",
            "Results worse than the pitching. Read each one for whether the "
            "level underneath is holding.",
            ppos,
            True,
        )
        + arms(
            "Due to give it back",
            "Results better than the pitching. The luck ends; the question is "
            "what is left when it does.",
            pneg,
            False,
        )
        + "<h2 class='part'>Part two &mdash; the bats</h2>"
        + bats(
            "Due to heat up",
            "Contact quality the results have not paid for yet &mdash; and, under each one, "
            "whether the swing agrees.",
            bpos,
            True,
        )
        + bats(
            "Due to cool off",
            "Results the contact has not earned. Where the swing disagrees the fade is the "
            "weaker read: half of what this cut flags out-produces what it keeps.",
            bneg,
            False,
        )
    )
    masthead = (
        "<div class='masthead'>"
        "<div class='brand'><span class='pp'>Payoff</span> Pitch &middot; "
        "Regression Report</div>"
        "<h1>Luck, skill, and the difference between them</h1>"
        "<p class='sub'>Every arm and bat on today's slate whose results and "
        "underlying numbers disagree &mdash; and which half of that is about "
        "to move.</p>"
        f"<div class='dateline'>Slate &middot; {nice}</div></div>"
    )
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style>"
        f"</head><body>{masthead}<div class='lead'>{LEAD}</div>{body}"
        f"<p class='fine'>{FINE}</p></body></html>"
    )


def build_article(
    day: Date,
    previews: list[dict],
    preds: list[dict],
    statcast: pd.DataFrame,
) -> str | None:
    """The article's HTML for one slate, or ``None`` when nothing is rankable.

    ``previews`` and ``preds`` are the JSON the run has just written, so the
    article ranks exactly the arms and bats the card was priced from.
    """
    ppos, pneg, pctxs = build_profiles(previews, preds, statcast)
    bpos, bneg = build_batter_profiles(preds, statcast)
    if not (ppos or pneg or bpos or bneg):
        return None
    bctxs = _batter_ctx(preds, {g["game_pk"]: g for g in previews})
    return build_html(day, ppos, pneg, pctxs, bpos, bneg, bctxs, preds)


def build_article_pdf(
    day: Date,
    previews: list[dict],
    preds: list[dict],
    statcast: pd.DataFrame,
) -> tuple[bytes, str] | None:
    """``(pdf_bytes, html)`` for the slate, or ``None`` when nothing is rankable."""
    html = build_article(day, previews, preds, statcast)
    if html is None:
        return None
    return to_pdf(html), html
