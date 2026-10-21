#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal Multiple-Object Monitoring (MOM) task using pygame --> implemented from Karimi-Rouzbahani et al., 2021, "Neural signatures of vigilance decrements predict behavioural errors before they occur", NeuroImage, 224:117405.

Dots of two colours travel from two corners toward a central white object along
visible lines, then turn away. You watch ONE colour (set by ATTEND_COLOUR). If a
dot of that colour does NOT turn and keeps heading for the centre (a "target"),
press SPACE before it reaches the centre. Everything is logged to a CSV.

Recorded per dot: onset, colour, attended?, type (target/event/distractor),
approach side, deflection direction, whether responded, reaction time, and
outcome (hit / miss[=lapse] / correct_rejection / false-alarm type). 
A summary (hits, misses/lapses, false alarms, mean RT) is printed at the end.

Like the PVT, the run starts with three cued blinks for EEG synchronisation
(see sonification/sync.py).

Usage
-----
    python tasks/mom_task.py --participant P1 --audio music --out recordings/

Output files (in --out, session ID <participant>_<audio>_<timestamp>):
    MOM_<session>_sync.csv, MOM_<session>_trials.csv, MOM_<session>_summary.txt
"""

import argparse
import os, csv, math, random
from datetime import datetime, timezone
import pygame


# ----------------------- settings (defaults; override on the command line) ------------------------
PARTICIPANT     = "P1"       # pseudonymous participant ID
AUDIO           = "none"     # audio condition played during the run: none / music / scrambled
ATTEND_COLOUR   = "green"    # colour to monitor
CONDITION       = "Active"   # "Active" = 50% targets, "Monitoring" = 6% targets
VIGILANCE_DECAY = True       # True = targets front-loaded (linear decay); False = uniform
BLOCK_SECONDS   = 60 * 20   # how long the run lasts
WINDOW          = (1000, 720)
FULLSCREEN      = False

OUTPUT_DIR = "recordings"

# --------------- fixed task parameters (from Karimi-Rouzbahani et al., 2021) -------------------
SPEED_MULT     = 1.30   # dots move this much faster than paper timings
BILATERAL_FRAC = 0.50   # fraction of event slots that spawn two opposite-side dots
T1, T2, TMOVE = 1.226/SPEED_MULT, 0.410/SPEED_MULT, 2.005/SPEED_MULT  # onset->deflect, deflect->centre, total
DEFL = T2 / (T1 + T2)                         # deflection point as fraction of radius
TARGET_FRAC = {"Active": 0.50, "Monitoring": 0.0625}
N_DOTS = max(8, round(64 * BLOCK_SECONDS / 110.0))
ISI_MEAN, ISI_SD, ISI_MIN, ISI_MAX = 1.66, 0.89, 0.30, 4.00
LAST_ONSET = BLOCK_SECONDS - 3.0

COLS  = {"green": (60, 200, 90), "red": (225, 70, 70)}
WHITE, BG, GREY = (240, 240, 240), (0, 0, 0), (70, 70, 70)
SYNC_COLOR = (100, 200, 255)
DIM_COLOR  = (60,  60,  80)

BLINK_CUE_COUNT   = 3
BLINK_CUE_GAP_MS  = 2_000
BLINK_CUE_HOLD_MS = 800

# geometry in pixels (field keeps the paper's 3.8:5 proportion, +y up)
_H = WINDOW[1]
FH = 0.40 * _H                 # field half-height
FW = FH * (3.8 / 5.0)          # field half-width
DOT_R = max(4, int(0.013 * _H))
OBJ_R = max(4, int(0.011 * _H))
CORNERS = {"left": (-FW, FH), "right": (FW, -FH)}
_u = (FW / math.hypot(FW, FH), FH / math.hypot(FW, FH))
DEFL_U = {"TR": _u, "BL": (-_u[0], -_u[1])}

# tiny vector helpers
def add(a, b):   return (a[0] + b[0], a[1] + b[1])
def scl(a, s):   return (a[0] * s, a[1] * s)
def length(a):   return math.hypot(a[0], a[1])
def unit(a):     n = length(a); return (a[0] / n, a[1] / n) if n else (0, 0)


class Dot:
    def __init__(self, idx, onset, corner_name, deflect_name, colour, is_cued, dtype):
        self.idx, self.onset = idx, onset
        self.corner = CORNERS[corner_name]
        self.approach, self.defl_name = corner_name, deflect_name
        self.P = scl(self.corner, DEFL)                  # deflection point
        self.du = DEFL_U[deflect_name]                   # deflection direction
        self.tc = unit(scl(self.corner, -1))             # toward centre
        self.colour, self.is_cued, self.dtype = colour, is_cued, dtype
        self.speed = length(self.corner) * (1 - DEFL) / T1
        self.responded = False
        self.outcome = self.rt = self.resp_t = None
        self.hit_from = self.hit_t = None

    def pos(self, t):                                    # world (x,y) at block time t
        age = t - self.onset                             # time since THIS dot appeared
        if age < 0:
            return None
        if age <= T1:
            return add(self.corner, scl((self.P[0] - self.corner[0],
                                         self.P[1] - self.corner[1]), age / T1))
        dt = age - T1
        if self.outcome == "hit":
            if age <= self.hit_t:
                return add(self.P, scl(self.tc, self.speed * dt))
            return add(self.hit_from, scl(self.du, self.speed * (age - self.hit_t)))
        if self.dtype == "event":
            return add(self.P, scl(self.du, self.speed * dt))
        return add(self.P, scl(self.tc, self.speed * dt))   # target/distractor: straight on

    def dist(self, t):
        p = self.pos(t); return length(p) if p else 1e9

    def active(self, t):
        return 0.0 <= (t - self.onset) <= TMOVE


def build_dots(rng):
    other = "red" if ATTEND_COLOUR == "green" else "green"
    frac  = TARGET_FRAC[CONDITION]

    # Each bilateral slot produces 2 dots; unilateral produces 1.
    # Choose n_slots so expected total ≈ N_DOTS.
    n_slots = max(6, round(N_DOTS / (1.0 + BILATERAL_FRAC)))
    isis = [min(max(rng.gauss(ISI_MEAN, ISI_SD), ISI_MIN), ISI_MAX)
            for _ in range(n_slots)]
    onsets, t = [], 0.0
    for x in isis:
        onsets.append(t); t += x
    if onsets[-1]:
        onsets = [o * (LAST_ONSET / onsets[-1]) for o in onsets]

    dots, idx = [], 0
    for onset in onsets:
        # time-dependent target probability (or fixed if no decay)
        if VIGILANCE_DECAY:
            p_tgt = 2.0 * frac * max(0.0, 1.0 - onset / LAST_ONSET)
        else:
            p_tgt = frac

        if rng.random() < BILATERAL_FRAC:
            # bilateral: one attended dot + one non-attended from the opposite side
            sides = ["left", "right"]
            rng.shuffle(sides)
            cued_side, dist_side = sides
            dtype_cued = "target" if rng.random() < p_tgt else "event"
            dots.append(Dot(idx, onset, cued_side, rng.choice(["TR", "BL"]),
                            ATTEND_COLOUR, True, dtype_cued))
            idx += 1
            dtype_non = "distractor" if rng.random() < frac else "event"
            dots.append(Dot(idx, onset, dist_side, rng.choice(["TR", "BL"]),
                            other, False, dtype_non))
            idx += 1
        else:
            # unilateral: random colour and side
            is_cued = rng.random() < 0.5
            colour  = ATTEND_COLOUR if is_cued else other
            dtype   = ("target" if rng.random() < p_tgt else "event") if is_cued \
                      else ("distractor" if rng.random() < frac else "event")
            dots.append(Dot(idx, onset, rng.choice(["left", "right"]),
                            rng.choice(["TR", "BL"]), colour, is_cued, dtype))
            idx += 1
    return dots


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def save_sync(sync_events: list, session_id: str, output_dir: str) -> str:
    path = os.path.join(output_dir, f"MOM_{session_id}_sync.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["blink_number", "task_clock_ms", "wall_clock_ts"])
        writer.writeheader()
        writer.writerows(sync_events)
    return path


def save_trials(dots: list, session_id: str, output_dir: str) -> str:
    path = os.path.join(output_dir, f"MOM_{session_id}_trials.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "onset_s", "colour", "attended", "type", "approach",
                    "deflect", "responded", "rt_s", "resp_time_s", "outcome"])
        for d in dots:
            w.writerow([d.idx, round(d.onset, 3), d.colour, int(d.is_cued), d.dtype,
                        d.approach, d.defl_name, int(d.responded),
                        "" if d.rt is None else round(d.rt, 3),
                        "" if d.resp_t is None else round(d.resp_t, 3), d.outcome])
    return path


def save_summary(dots: list, session_id: str, output_dir: str) -> str:
    targets  = [d for d in dots if d.is_cued and d.dtype == "target"]
    hits     = [d for d in targets if d.outcome == "hit"]
    misses   = [d for d in targets if d.outcome == "miss"]
    fas      = [d for d in dots if d.outcome and d.outcome.startswith("fa")]
    mean_rt  = sum(d.rt for d in hits) / len(hits) * 1000 if hits else float("nan")

    path = os.path.join(output_dir, f"MOM_{session_id}_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"MOM Summary — {session_id}\n" + "=" * 50 + "\n")
        f.write(f"Participant       : {PARTICIPANT}\n")
        f.write(f"Audio             : {AUDIO}\n")
        f.write(f"Condition         : {CONDITION}\n")
        f.write(f"Attended colour   : {ATTEND_COLOUR}\n")
        f.write(f"Block duration (s): {BLOCK_SECONDS}\n\n")
        f.write(f"Targets           : {len(targets)}\n")
        f.write(f"Hits              : {len(hits)}\n")
        f.write(f"Misses (lapses)   : {len(misses)}\n")
        f.write(f"False alarms      : {len(fas)}\n")
        f.write(f"Mean hit RT (ms)  : {mean_rt:.0f}\n")
    return path


def show_sync_instructions(screen, clock):
    W, H = screen.get_size()
    title_font = pygame.font.SysFont(None, 44)
    body_font  = pygame.font.SysFont(None, 30)
    lines = [
        ("SYNC PHASE",                                               title_font, SYNC_COLOR),
        ("",                                                          body_font,  GREY),
        ("Before the task, we need to sync the recording.",           body_font,  WHITE),
        ("",                                                          body_font,  GREY),
        (f"The word  BLINK  will appear {BLINK_CUE_COUNT} times.",   body_font,  WHITE),
        ("Each time it appears, blink your eyes firmly once.",        body_font,  WHITE),
        ("",                                                          body_font,  GREY),
        ("Blink only when you see the word.",                         body_font,  WHITE),
        ("Do not blink at other times.",                              body_font,  WHITE),
        ("",                                                          body_font,  GREY),
        ("Press  SPACE  when you are ready.",                         title_font, SYNC_COLOR),
    ]
    waiting = True
    while waiting:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); raise SystemExit
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                waiting = False
        screen.fill(BG)
        total_h = sum(f.get_linesize() for _, f, _ in lines)
        y = (H - total_h) // 2
        for text, f, col in lines:
            surf = f.render(text, True, col)
            screen.blit(surf, (W // 2 - surf.get_width() // 2, y))
            y += f.get_linesize()
        pygame.display.flip()
        clock.tick(30)


def run_blink_sync(screen, clock) -> list:
    W, H = screen.get_size()
    font_big  = pygame.font.SysFont(None, 80)
    font_body = pygame.font.SysFont(None, 30)
    sync_events = []

    for blink_num in range(1, BLINK_CUE_COUNT + 1):
        wait_start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - wait_start < BLINK_CUE_GAP_MS:
            screen.fill(BG)
            surf = font_body.render(
                f"Blink {blink_num - 1} / {BLINK_CUE_COUNT}  —  get ready...", True, DIM_COLOR)
            screen.blit(surf, surf.get_rect(center=(W // 2, H // 2)))
            pygame.display.flip()
            clock.tick(60)
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    pygame.quit(); raise SystemExit

        cue_task_clock_ms = pygame.time.get_ticks()
        sync_events.append({
            "blink_number":  blink_num,
            "task_clock_ms": cue_task_clock_ms,
            "wall_clock_ts": iso_now(),
        })

        cue_start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - cue_start < BLINK_CUE_HOLD_MS:
            screen.fill(BG)
            surf = font_big.render("BLINK", True, SYNC_COLOR)
            screen.blit(surf, surf.get_rect(center=(W // 2, H // 2)))
            num_surf = font_body.render(f"({blink_num} of {BLINK_CUE_COUNT})", True, DIM_COLOR)
            screen.blit(num_surf, num_surf.get_rect(center=(W // 2, H // 2 + 70)))
            pygame.display.flip()
            clock.tick(60)
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    pygame.quit(); raise SystemExit

    screen.fill(BG)
    surf = font_body.render("Sync complete. Starting task...", True, SYNC_COLOR)
    screen.blit(surf, surf.get_rect(center=(W // 2, H // 2)))
    pygame.display.flip()
    pygame.time.wait(1500)

    return sync_events


def show_instructions(screen, clock):
    W, H = screen.get_size()
    title_font = pygame.font.SysFont(None, 44)
    body_font  = pygame.font.SysFont(None, 30)
    small_font = pygame.font.SysFont(None, 26)
    dur_min    = BLOCK_SECONDS // 60

    lines = [
        ("MULTIPLE OBJECT MONITORING TASK",                                    title_font, WHITE),
        ("",                                                                    body_font,  GREY),
        (f"Watch the  {ATTEND_COLOUR.upper()}  dots only.",                    body_font,  COLS[ATTEND_COLOUR]),
        ("",                                                                    body_font,  GREY),
        ("Dots travel from the corners toward the centre along visible lines.", body_font,  WHITE),
        ("Most dots turn away before reaching the centre — let them pass.",     body_font,  WHITE),
        ("",                                                                    body_font,  GREY),
        (f"TARGET:  a {ATTEND_COLOUR.upper()} dot that does NOT turn and",     body_font,  WHITE),
        ("continues straight toward the centre.",                               body_font,  WHITE),
        (f"Press  SPACE  as soon as you see a target.",                        body_font,  WHITE),
        ("",                                                                    body_font,  GREY),
        ("Avoid pressing SPACE at any other time.",                             body_font,  GREY),
        ("",                                                                    body_font,  GREY),
        (f"Task duration:  {dur_min} min",                                     small_font, GREY),
        ("",                                                                    body_font,  GREY),
        ("Press  SPACE  to begin.",                                             title_font, WHITE),
    ]

    waiting = True
    while waiting:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); raise SystemExit
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                waiting = False
        screen.fill(BG)
        total_h = sum(f.get_linesize() for _, f, _ in lines)
        y = (H - total_h) // 2
        for text, f, col in lines:
            surf = f.render(text, True, col)
            screen.blit(surf, (W // 2 - surf.get_width() // 2, y))
            y += f.get_linesize()
        pygame.display.flip()
        clock.tick(30)


def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW, pygame.FULLSCREEN if FULLSCREEN else 0)
    pygame.display.set_caption("MOM task")
    pygame.mouse.set_visible(False)
    W, H = screen.get_size()
    cx, cy = W // 2, H // 2
    font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()
    to_screen = lambda p: (int(cx + p[0]), int(cy - p[1]))

    session_id = f"{PARTICIPANT}_{AUDIO}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    show_instructions(screen, clock)

    show_sync_instructions(screen, clock)
    sync_events = run_blink_sync(screen, clock)
    sync_path = save_sync(sync_events, session_id, OUTPUT_DIR)
    print(f"Sync saved: {sync_path}")

    rng = random.Random()
    dots = build_dots(rng)
    flash = [None, 0.0]                       # (colour, until) feedback tint on centre

    start = pygame.time.get_ticks() / 1000.0
    running = True
    completed = False
    while running:
        t = pygame.time.get_ticks() / 1000.0 - start
        if t >= BLOCK_SECONDS:
            completed = True
            break

        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                cands = [d for d in dots if d.active(t) and not d.responded]
                if cands:
                    d = min(cands, key=lambda x: x.dist(t))
                    et = t - d.onset
                    d.responded, d.rt, d.resp_t = True, et, t
                    if d.is_cued and d.dtype == "target" and T1 <= et < T1 + T2:
                        d.outcome = "hit"
                        d.hit_t = et
                        d.hit_from = add(d.P, scl(d.tc, d.speed * (et - T1)))
                        flash[:] = ["hit", t + 0.15]
                    else:
                        d.outcome = ("fa_early" if et < T1 else
                                     "fa_distractor" if not d.is_cued else
                                     "fa_event" if d.dtype == "event" else "fa_late")
                        flash[:] = ["fa", t + 0.15]

        # draw
        screen.fill(BG)
        for d in dots:
            if not d.active(t):
                continue
            pts = [to_screen(d.corner), to_screen(d.P),
                   to_screen(add(d.P, scl(d.du, d.speed * (TMOVE - T1))))]
            pygame.draw.lines(screen, GREY, False, pts, 2)
            p = d.pos(t)
            if p:
                pygame.draw.circle(screen, COLS[d.colour], to_screen(p), DOT_R)
        oc = WHITE
        if flash[0] and t < flash[1]:
            oc = COLS["green"] if flash[0] == "hit" else COLS["red"]
        pygame.draw.circle(screen, oc, (cx, cy), OBJ_R)
        screen.blit(font.render(f"watch {ATTEND_COLOUR.upper()}", True, GREY), (12, 10))
        # elapsed / total timer (top-right)
        em, es   = divmod(int(t), 60)
        tm, ts   = divmod(int(BLOCK_SECONDS), 60)
        timer_surf = font.render(f"{em:02d}:{es:02d} / {tm:02d}:{ts:02d}", True, GREY)
        screen.blit(timer_surf, (W - timer_surf.get_width() - 12, 10))
        # progress bar (bottom edge)
        pygame.draw.rect(screen, GREY, (0, H - 4, int(W * t / BLOCK_SECONDS), 4))
        pygame.display.flip()
        clock.tick(60)

    if not completed:
        os.remove(sync_path)
        print("\nSession aborted early — no data saved.")
        pygame.quit()
        return

    # finalise outcomes
    for d in dots:
        if d.outcome is None:
            d.outcome = "miss" if (d.is_cued and d.dtype == "target") else "correct_rejection"

    trials_path  = save_trials(dots, session_id, OUTPUT_DIR)
    summary_path = save_summary(dots, session_id, OUTPUT_DIR)

    targets  = [d for d in dots if d.is_cued and d.dtype == "target"]
    hits     = [d for d in targets if d.outcome == "hit"]
    misses   = [d for d in targets if d.outcome == "miss"]
    fas      = [d for d in dots if d.outcome and d.outcome.startswith("fa")]
    mean_rt  = sum(d.rt for d in hits) / len(hits) * 1000 if hits else float("nan")
    print(f"\nSession: {session_id}")
    print(f"Targets: {len(targets)} | Hits: {len(hits)} | "
          f"Misses (lapses): {len(misses)} | False alarms: {len(fas)}")
    print(f"Mean hit RT: {mean_rt:.0f} ms")
    print(f"Saved to: {OUTPUT_DIR}")
    print(f"  {os.path.basename(sync_path)}")
    print(f"  {os.path.basename(trials_path)}")
    print(f"  {os.path.basename(summary_path)}")
    pygame.quit()


def configure():
    """Apply command-line options and recompute the values derived from them."""
    global PARTICIPANT, AUDIO, ATTEND_COLOUR, CONDITION, VIGILANCE_DECAY, BLOCK_SECONDS
    global FULLSCREEN, OUTPUT_DIR, N_DOTS, LAST_ONSET
    p = argparse.ArgumentParser(description="Multiple-Object Monitoring task with blink sync")
    p.add_argument("--participant", default=PARTICIPANT, help="pseudonymous participant ID")
    p.add_argument("--audio", default=AUDIO, help="audio condition label: none / music / scrambled")
    p.add_argument("--colour", default=ATTEND_COLOUR, choices=list(COLS), help="colour to monitor")
    p.add_argument("--condition", default=CONDITION, choices=list(TARGET_FRAC),
                   help="Active = 50%% targets, Monitoring = 6%% targets")
    p.add_argument("--no-decay", action="store_true", help="uniform target probability over the run")
    p.add_argument("--minutes", type=float, default=BLOCK_SECONDS / 60, help="run duration")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--out", default=OUTPUT_DIR, help="output folder")
    a = p.parse_args()
    PARTICIPANT, AUDIO, ATTEND_COLOUR, CONDITION = a.participant, a.audio, a.colour, a.condition
    VIGILANCE_DECAY = not a.no_decay
    BLOCK_SECONDS = int(a.minutes * 60)
    FULLSCREEN, OUTPUT_DIR = a.fullscreen, a.out
    N_DOTS = max(8, round(64 * BLOCK_SECONDS / 110.0))
    LAST_ONSET = BLOCK_SECONDS - 3.0


if __name__ == "__main__":
    configure()
    main()
