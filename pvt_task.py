"""
Psychomotor Vigilance Task (PVT)  —  with Blink Sync

Standard 10-minute PVT. Begins with a blink sync phase that embeds
three cued blink artifacts into the EEG signal, creating alignment
anchors that let you map trial timestamps to EEG sample numbers even
when you don't know when the EEG recording started.

Usage
-----
    python tasks/pvt_task.py --participant P1 --condition no_music --out recordings/

Controls
--------
SPACE  – respond to stimulus / trigger false start
ESC    – quit early (results still saved)

Output files (in --out, sharing one session ID <participant>_<condition>_<timestamp>)
------------------------------------------------------------------------------------
pvt_<session>_sync.csv      – 3 blink cue times in task-clock ms
pvt_<session>_results.csv   – trial-level data
pvt_<session>_summary.txt   – aggregate stats

Key columns in pvt_results CSV
-------------------------------
task_clock_ms     – pygame ticks at stimulus onset (use this for EEG alignment)
reaction_time_ms  – ms from stimulus onset to SPACE press
lapse             – 1 if RT > 500 ms
false_start       – 1 if SPACE pressed before stimulus appeared

"""

import argparse
import pygame
import random
import csv
from datetime import datetime, timezone
from pathlib import Path

#  CONFIGURATION  
TASK_DURATION_SEC  = 10 * 60    # 600 s = 10 min
ISI_MIN_MS         = 2_000      # min inter-stimulus interval
ISI_MAX_MS         = 10_000     # max inter-stimulus interval
LAPSE_THRESHOLD_MS = 500        # RT above this = lapse
MAX_RESPONSE_MS    = 1_500      # no response within this → auto lapse

BLINK_CUE_COUNT    = 3          # number of sync blinks
BLINK_CUE_GAP_MS   = 2_000      # gap between blink cues
BLINK_CUE_HOLD_MS  = 800        # how long "BLINK" text stays on screen

SCREEN_W, SCREEN_H = 800, 500

# Colours
BG_COLOR    = (15,  15,  20)
TEXT_COLOR  = (220, 220, 230)
STIM_COLOR  = (255, 220,  50)
LAPSE_COLOR = (220,  60,  60)
SYNC_COLOR  = (100, 200, 255)
DIM_COLOR   = (60,   60,  80)

# function to give timestamps a string suitable for recording event times consistently
def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

# function to write the blink sync events to a CSV file  
def save_sync(sync_events: list[dict], output_dir: Path, session_id: str) -> Path:
    path = output_dir / f"pvt_{session_id}_sync.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["blink_number", "task_clock_ms", "wall_clock_ts"]) #wall_clock = actual UTC time when the event happened, task_clock = PVT task timer (pygame ticks since the task started)
        writer.writeheader()
        writer.writerows(sync_events)
    return path


def save_results(trials: list[dict], output_dir: Path, session_id: str) -> tuple[Path, Path]:
    csv_path = output_dir / f"pvt_{session_id}_results.csv"
    txt_path = output_dir / f"pvt_{session_id}_summary.txt"

    fieldnames = [
        "trial", "task_clock_ms", "stimulus_onset_ts", "isi_ms",
        "reaction_time_ms", "lapse", "false_start",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)

    valid_rts = [t["reaction_time_ms"] for t in trials
        if t["reaction_time_ms"] is not None and not t["false_start"]]
    n_lapses  = sum(t["lapse"]        for t in trials)
    n_false   = sum(t["false_start"]  for t in trials)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"PVT Summary — {session_id}\n" + "=" * 40 + "\n")
        f.write(f"Total trials          : {len(trials)}\n")
        f.write(f"False starts          : {n_false}\n")
        f.write(f"Lapses (RT > 500 ms)  : {n_lapses}\n")
        if valid_rts:
            s = sorted(valid_rts)
            f.write(f"Mean RT (ms)          : {sum(s)/len(s):.1f}\n")
            f.write(f"Median RT (ms)        : {s[len(s)//2]:.1f}\n")
            f.write(f"Min RT (ms)           : {s[0]:.1f}\n")
            f.write(f"Max RT (ms)           : {s[-1]:.1f}\n")
            n10 = max(1, len(s) // 10)
            f.write(f"Fastest 10% RT (ms)   : {s[n10-1]:.1f}\n")
            f.write(f"Slowest 10% RT (ms)   : {s[-n10]:.1f}\n")
        else:
            f.write("No valid RTs recorded.\n")

    return csv_path, txt_path


# helper functions for the PVT user-interface
def draw_lines(screen, font, lines, start_y=80, line_h=36):
    """Render a list of (text, color) tuples centred on screen."""
    for i, item in enumerate(lines):
        text, color = (item, TEXT_COLOR) if isinstance(item, str) else item
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(SCREEN_W // 2, start_y + i * line_h))
        screen.blit(surf, rect)


def wait_for_space(screen):
    """Block until SPACE is pressed. ESC quits."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); raise SystemExit
                if event.key == pygame.K_SPACE:
                    return


# screens and user interface
def show_intro(screen, font_title, font_body):
    screen.fill(BG_COLOR)
    draw_lines(screen, font_title, [("PSYCHOMOTOR VIGILANCE TASK", STIM_COLOR)], start_y=70)
    draw_lines(screen, font_body, [
        "",
        "A yellow counter will appear at random intervals.",
        "Press SPACE as quickly as possible when you see it.",
        "",
        ("Do NOT press SPACE before the counter appears.", LAPSE_COLOR),
        "",
        f"Task duration: {TASK_DURATION_SEC // 60} minutes",
        "",
        ("Press SPACE to begin.", SYNC_COLOR),
    ], start_y=120)
    pygame.display.flip()
    wait_for_space(screen)


def show_sync_instructions(screen, font_title, font_body):
    screen.fill(BG_COLOR)
    draw_lines(screen, font_title, [("SYNC PHASE", SYNC_COLOR)], start_y=70)
    draw_lines(screen, font_body, [
        "",
        "Before the task, we need to sync the recording.",
        "",
        f"The word  BLINK  will appear {BLINK_CUE_COUNT} times.",
        "Each time it appears, blink your eyes firmly once.",
        "",
        "Blink only when you see the word.",
        "Do not blink at other times.",
        "",
        ("Press SPACE when you are ready.", SYNC_COLOR),
    ], start_y=110)
    pygame.display.flip()
    wait_for_space(screen)


def show_end_screen(screen, font_title, font_body, csv_path, txt_path, trials):
    n_lapses  = sum(t["lapse"]        for t in trials)
    n_false   = sum(t["false_start"]  for t in trials)
    valid_rts = [t["reaction_time_ms"] for t in trials
        if t["reaction_time_ms"] is not None and not t["false_start"]]
    mean_rt   = f"{sum(valid_rts)/len(valid_rts):.0f} ms" if valid_rts else "N/A"

    screen.fill(BG_COLOR)
    draw_lines(screen, font_title, [("TASK COMPLETE", STIM_COLOR)], start_y=70)
    draw_lines(screen, font_body, [
        "",
        f"Trials completed : {len(trials)}",
        (f"Lapses (>500 ms) : {n_lapses}", LAPSE_COLOR if n_lapses else TEXT_COLOR),
        (f"False starts     : {n_false}",  LAPSE_COLOR if n_false  else TEXT_COLOR),
        f"Mean RT          : {mean_rt}",
        "",
        ("Results saved:", SYNC_COLOR),
        (csv_path.name,   SYNC_COLOR),
        (txt_path.name,   SYNC_COLOR),
        "",
        ("Press ESC to exit.", DIM_COLOR),
    ], start_y=110)
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return


#  BLINK SYNC PHASE
def run_blink_sync(screen, font_big, font_body, clock) -> list[dict]:
    """
    Show BLINK_CUE_COUNT cued blinks.
    Records task_clock_ms at the moment each cue appears.
    Returns list of sync event dicts.
    """
    sync_events = []

    for blink_num in range(1, BLINK_CUE_COUNT + 1):

        # ── Blank wait between cues ──────────
        wait_start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - wait_start < BLINK_CUE_GAP_MS:
            screen.fill(BG_COLOR)
            progress = f"{blink_num - 1} / {BLINK_CUE_COUNT}"
            surf = font_body.render(f"Blink {progress}  —  get ready...", True, DIM_COLOR)
            screen.blit(surf, surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2)))
            pygame.display.flip()
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); raise SystemExit

        # ── Show BLINK cue — record time NOW ─
        cue_task_clock_ms = pygame.time.get_ticks()
        sync_events.append({
            "blink_number":   blink_num,
            "task_clock_ms":  cue_task_clock_ms,
            "wall_clock_ts":  iso_now(),
        })

        cue_start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - cue_start < BLINK_CUE_HOLD_MS:
            screen.fill(BG_COLOR)
            surf = font_big.render("BLINK", True, SYNC_COLOR)
            screen.blit(surf, surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2)))
            num_surf = font_body.render(f"({blink_num} of {BLINK_CUE_COUNT})", True, DIM_COLOR)
            screen.blit(num_surf, num_surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 70)))
            pygame.display.flip()
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); raise SystemExit

    # Brief "sync complete" message
    screen.fill(BG_COLOR)
    surf = font_body.render("Sync complete. Starting task...", True, SYNC_COLOR)
    screen.blit(surf, surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2)))
    pygame.display.flip()
    pygame.time.wait(1500)

    return sync_events


#  MAIN TASK LOOP
def run_task(screen, font_counter, font_info, font_small, clock) -> list[dict]:
    trials        = []
    trial_num     = 0
    task_start    = pygame.time.get_ticks()
    task_end      = task_start + TASK_DURATION_SEC * 1000

    state         = "waiting_isi"
    isi_ms        = random.randint(ISI_MIN_MS, ISI_MAX_MS)
    isi_start     = pygame.time.get_ticks()
    current_isi   = isi_ms

    stim_ticks    = 0
    stim_ts       = ""

    false_flash   = 0   # ticks until false-start flash clears

    running = True
    while running:
        now = pygame.time.get_ticks()

        if now >= task_end:
            break

        # ── Events ───────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:

                    if state == "waiting_isi":
                        trial_num += 1
                        trials.append({
                            "trial":            trial_num,
                            "task_clock_ms":    None,
                            "stimulus_onset_ts": None,
                            "isi_ms":           current_isi,
                            "reaction_time_ms": None,
                            "lapse":            0,
                            "false_start":      1,
                        })
                        false_flash  = now + 800
                        isi_ms       = random.randint(ISI_MIN_MS, ISI_MAX_MS)
                        isi_start    = now
                        current_isi  = isi_ms

                    elif state == "stimulus_on":
                        rt_ms = now - stim_ticks
                        trial_num += 1
                        trials.append({
                            "trial":            trial_num,
                            "task_clock_ms":    stim_ticks,   # ← key column for EEG alignment
                            "stimulus_onset_ts": stim_ts,
                            "isi_ms":           current_isi,
                            "reaction_time_ms": rt_ms,
                            "lapse":            1 if rt_ms > LAPSE_THRESHOLD_MS else 0,
                            "false_start":      0,
                        })
                        state       = "waiting_isi"
                        isi_ms      = random.randint(ISI_MIN_MS, ISI_MAX_MS)
                        isi_start   = now
                        current_isi = isi_ms

        # ── State transitions ─────────────────
        if state == "waiting_isi":
            if now - isi_start >= isi_ms:
                state      = "stimulus_on"
                stim_ticks = now
                stim_ts    = iso_now()

        elif state == "stimulus_on":
            if now - stim_ticks > MAX_RESPONSE_MS:
                trial_num += 1
                trials.append({
                    "trial":            trial_num,
                    "task_clock_ms":    stim_ticks,
                    "stimulus_onset_ts": stim_ts,
                    "isi_ms":           current_isi,
                    "reaction_time_ms": MAX_RESPONSE_MS,
                    "lapse":            1,
                    "false_start":      0,
                })
                state       = "waiting_isi"
                isi_ms      = random.randint(ISI_MIN_MS, ISI_MAX_MS)
                isi_start   = now
                current_isi = isi_ms

        # ── Draw ─────────────────────────────
        screen.fill(BG_COLOR)

        # Progress bar
        elapsed = now - task_start
        prog    = min(elapsed / (TASK_DURATION_SEC * 1000), 1.0)
        pygame.draw.rect(screen, (40, 40, 50), (40, 18, SCREEN_W - 80, 10), border_radius=5)
        pygame.draw.rect(screen, SYNC_COLOR,   (40, 18, int((SCREEN_W - 80) * prog), 10), border_radius=5)

        # Time remaining
        rem      = max(0, (task_end - now) // 1000)
        mins, s  = divmod(rem, 60)
        screen.blit(font_small.render(f"Time remaining: {mins:02d}:{s:02d}", True, TEXT_COLOR), (40, 36))

        # Stats top-right
        n_lap = sum(t["lapse"]       for t in trials)
        n_fal = sum(t["false_start"] for t in trials)
        for i, line in enumerate([f"Trials      : {trial_num}",
            f"Lapses      : {n_lap}",
            f"False starts: {n_fal}"]):
            screen.blit(font_small.render(line, True, TEXT_COLOR), (SCREEN_W - 220, 36 + i * 22))

        # Stimulus / waiting
        if state == "stimulus_on":
            elapsed_stim = now - stim_ticks
            color = LAPSE_COLOR if elapsed_stim > LAPSE_THRESHOLD_MS else STIM_COLOR
            for text, y in [(f"{elapsed_stim:04d}", 0), ("Press SPACE!", 70)]:
                surf = (font_counter if y == 0 else font_info).render(text, True, color)
                screen.blit(surf, surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + y)))

        elif state == "waiting_isi":
            msg   = "FALSE START!" if now < false_flash else "· · ·"
            color = LAPSE_COLOR   if now < false_flash else DIM_COLOR
            surf  = font_info.render(msg, True, color)
            screen.blit(surf, surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2)))

        pygame.display.flip()
        clock.tick(120)

    return trials


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def run_pvt(participant="P1", condition="session", output_dir="recordings"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = f"{participant}_{condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Psychomotor Vigilance Task")
    clock = pygame.time.Clock()

    font_counter = pygame.font.SysFont("Courier New", 80, bold=True)
    font_big     = pygame.font.SysFont("Courier New", 64, bold=True)
    font_info    = pygame.font.SysFont("Courier New", 24)
    font_small   = pygame.font.SysFont("Courier New", 18)

    # 1. Task instructions
    show_intro(screen, font_info, font_info)

    # 2. Blink sync instructions + sync phase
    show_sync_instructions(screen, font_info, font_info)
    sync_events = run_blink_sync(screen, font_big, font_info, clock)
    sync_path   = save_sync(sync_events, output_dir, session_id)
    print(f"Sync saved: {sync_path}")

    # 3. PVT task
    trials = run_task(screen, font_counter, font_info, font_small, clock)

    # 4. Save & show results
    if trials:
        csv_path, txt_path = save_results(trials, output_dir, session_id)
        show_end_screen(screen, font_info, font_info, csv_path, txt_path, trials)

    pygame.quit()


def main():
    global TASK_DURATION_SEC
    p = argparse.ArgumentParser(description="Psychomotor Vigilance Task with blink sync")
    p.add_argument("--participant", default="P1", help="pseudonymous participant ID")
    p.add_argument("--condition", default="session", help="e.g. no_music, music")
    p.add_argument("--out", default="recordings", help="output folder")
    p.add_argument("--minutes", type=float, default=TASK_DURATION_SEC / 60, help="task duration")
    a = p.parse_args()
    TASK_DURATION_SEC = int(a.minutes * 60)
    run_pvt(a.participant, a.condition, a.out)


if __name__ == "__main__":
    main()