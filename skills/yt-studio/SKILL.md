---
name: yt-studio
description: >-
  Turn a plain talking-head recording into a branded news-studio video - the
  speaker cut out of their room and placed in a studio with the channel logo,
  a lower third, a small speed-up, and the channel's intro and outro joined
  with transitions. Use when the user sends a video of someone talking and
  asks to "change the background", "make it look like a studio", "add my
  intro/outro", or "speed him up".
---

# yt-studio

A webcam or phone recording into something that looks like it came out of a newsroom.

One tool lives in this folder and it actually runs. Use it, do not describe it.

```bash
python3 studio.py talk.mp4 --bg studio.jpg --logo logo.png \
    --intro intro.mp4 --outro outro.mp4 \
    --channel "BB NEWS" --tagline "BEYOND THE BOUNDARIES" --speed 1.1 -o final.mp4
```

Needs `ffmpeg` plus `pip install mediapipe opencv-python-headless pillow numpy`.
The segmentation model (250 KB) downloads itself on first run.

## Before you run it

1. **Look at the video first.** Pull a contact sheet (`ffmpeg -vf fps=1/30,tile=6x3`). One
   speaker, static camera, reasonable light: this works. Several people, a moving camera, or a
   speaker who leaves the frame: say so before burning 40 minutes of render.
2. **Ask for the brand, do not invent it.** Logo, channel name, tagline, intro, outro. If the user
   sends two intros, use the one that fits a news show and say which. If two files are identical
   (check `md5sum`), use one and say so.
3. **The background.** If the user has no studio image, generate one: an empty TV news studio in the
   channel's colours, no people, no text. The logo goes on the wall by the tool, not by the image
   model - image models misspell logos.

## Defaults and why

- `--speed 1.1`: tightens a slow speaker without audible pitch change. Above 1.2 sounds rushed.
- Loudness is normalised to -16 LUFS so it matches the intro and outro.
- Transitions only at the joins (intro -> body, body -> outro). A single continuous take has no
  cuts that need one; adding transitions mid-speech looks like a mistake.

## Be honest about the limits

- **Low-resolution source stays soft.** A 848x392 phone clip scaled into 1080p looks soft. The studio
  and logos stay sharp. Say this up front.
- **Hair edges can shimmer.** This is AI segmentation, not a green screen.
- **Speed:** about 6 frames a second per CPU core. A 12-minute video on 2 cores is ~35-45 minutes.
  Run it in the background and tell the user when to expect it.
- **CapCut has no public API.** Do not claim to drive it. This tool replaces that step.

## What to hand back

- the finished MP4, sent as a file
- one frame from the middle so they can judge the cutout before watching 12 minutes
- the final runtime and the speed used

## The gate

Nothing here publishes. Last line of every run: **ship it, or change it?**
