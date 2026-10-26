#!/usr/bin/env python3
"""yt-studio: put a talking-head recording into a branded news studio.

Pipeline: speed up -> cut the speaker out (MediaPipe selfie segmentation) ->
composite over a studio background with logo, corner bug and lower third ->
join intro + body + outro with transitions.

    python3 studio.py talk.mp4 --bg studio.jpg --logo logo.png \
        --intro intro.mp4 --outro outro.mp4 --channel "BB NEWS" \
        --tagline "BEYOND THE BOUNDARIES" --speed 1.1 -o final.mp4

Needs: ffmpeg on PATH, pip install mediapipe opencv-python-headless pillow numpy
"""
import argparse, os, subprocess, sys, tempfile, urllib.request
from concurrent.futures import ProcessPoolExecutor

W, H = 1920, 1080
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def sh(*a):
    subprocess.run(list(a), check=True)


def dur(f):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", f]).strip())


def has_audio(f):
    return bool(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a",
                                         "-show_entries", "stream=index", "-of", "csv=p=0", f]).strip())


def load_font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def build_layers(a):
    """Background plate (studio + wall logo) and foreground overlay (bug + lower third)."""
    from PIL import Image, ImageDraw
    st = Image.open(a.bg).convert("RGB")
    st = st.resize((W, max(H, int(st.height * W / st.width)))).crop((0, 0, W, H))
    plate = st.convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if a.logo:
        logo = Image.open(a.logo).convert("RGBA")
        wall = logo.resize((300, int(300 * logo.height / logo.width)))
        plate.alpha_composite(wall, (W - 400, 140))
        bug = logo.resize((120, int(120 * logo.height / logo.width)))
        ov.alpha_composite(bug, (W - 150, 25))
    d = ImageDraw.Draw(ov)
    if a.channel or a.tagline:
        d.rectangle([0, 960, W, 1030], fill=(190, 10, 20, 235))
        d.rectangle([0, 960, 330, 1030], fill=(15, 15, 15, 245))
        d.text((165, 995), a.channel or "", font=load_font(40), fill="white", anchor="mm")
        d.text((360, 995), a.tagline or "", font=load_font(36), fill="white", anchor="lm")
    return plate.convert("RGB"), ov


def composite(job):
    """Worker: segment and composite one chunk of the sped-up video."""
    inp, out, model, plate_png, ov_png = job
    import cv2, numpy as np, mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    bg = cv2.imread(plate_png).astype(np.float32)
    ova = cv2.imread(ov_png, cv2.IMREAD_UNCHANGED).astype(np.float32)
    ovc, ovA = ova[:, :, :3], ova[:, :, 3:] / 255
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True))
    cap = cv2.VideoCapture(inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sw0, sh0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    SH = 900; SW = int(sw0 * SH / sh0)          # speaker fills 900px of height, bottom-aligned
    ox, oy = (W - SW) // 2, H - SH - 50
    x0, x1 = max(ox, 0), min(ox + SW, W); sx0 = x0 - ox; sx1 = sx0 + (x1 - x0)
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    i, prev = 0, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        m = seg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)),
                                  int(i * 1000 / fps)).confidence_masks[0].numpy_view().copy()
        m = np.clip((m - 0.3) / 0.4, 0, 1)
        prev = m if prev is None else 0.6 * m + 0.4 * prev   # temporal smoothing = less edge flicker
        m = cv2.GaussianBlur(prev, (7, 7), 0)
        fs = cv2.resize(f, (SW, SH), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        ms = cv2.resize(m, (SW, SH))[..., None][:, sx0:sx1]
        fr = bg.copy()
        reg = fr[oy:oy + SH, x0:x1]
        fr[oy:oy + SH, x0:x1] = fs[:, sx0:sx1] * ms + reg * (1 - ms)
        fr = ovc * ovA + fr * (1 - ovA)
        wr.write(fr.astype(np.uint8)); i += 1
    wr.release()
    return i


def norm_clip(src, dst, d=None):
    """Re-encode an intro/outro to 1920x1080/30fps with a stereo track (silence if none)."""
    args = ["ffmpeg", "-v", "error", "-y", "-i", src]
    if not has_audio(src):
        args += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-map", "0:v", "-map", "1:a", "-shortest"]
    args += ["-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p",
             "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-ar", "44100", "-ac", "2", dst]
    sh(*args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video"); p.add_argument("-o", "--out", default="studio_final.mp4")
    p.add_argument("--bg", required=True, help="studio background image")
    p.add_argument("--logo"); p.add_argument("--intro"); p.add_argument("--outro")
    p.add_argument("--channel", default=""); p.add_argument("--tagline", default="")
    p.add_argument("--speed", type=float, default=1.1, help="1.0 = unchanged; 1.1 tightens without chipmunking")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--transition-in", default="circleopen"); p.add_argument("--transition-out", default="fadeblack")
    p.add_argument("--workdir")
    a = p.parse_args()

    wd = os.path.abspath(a.workdir or tempfile.mkdtemp(prefix="ytstudio_"))
    os.makedirs(wd, exist_ok=True)
    model = os.path.join(wd, "seg.tflite")
    if not os.path.exists(model):
        urllib.request.urlretrieve(MODEL_URL, model)

    plate, ov = build_layers(a)
    plate.save(f"{wd}/plate.png"); ov.save(f"{wd}/overlay.png")

    print("1/4 speed + loudness")
    fast = f"{wd}/fast.mp4"
    sh("ffmpeg", "-v", "error", "-y", "-i", a.video, "-filter_complex",
       f"[0:v]setpts=PTS/{a.speed},fps=30[v];[0:a]atempo={a.speed},loudnorm=I=-16:TP=-1.5[a]",
       "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
       "-c:a", "aac", "-b:a", "160k", fast)

    print(f"2/4 background replace on {a.workers} workers (slowest step)")
    D = dur(fast); n = max(1, a.workers); step = D / n; jobs = []
    for k in range(n):
        part = f"{wd}/part{k}.mp4"
        sh("ffmpeg", "-v", "error", "-y", "-ss", str(k * step), "-t", str(step), "-i", fast, "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", part)
        jobs.append((part, f"{wd}/comp{k}.mp4", model, f"{wd}/plate.png", f"{wd}/overlay.png"))
    with ProcessPoolExecutor(n) as ex:
        print("   frames:", list(ex.map(composite, jobs)))

    print("3/4 join body")
    with open(f"{wd}/list.txt", "w") as fh:
        fh.writelines(f"file '{j[1]}'\n" for j in jobs)
    body = f"{wd}/body.mp4"
    sh("ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{wd}/list.txt", "-i", fast,
       "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
       "-c:a", "aac", "-ar", "44100", "-ac", "2", "-shortest", body)

    print("4/4 intro / outro + transitions")
    clips = []
    if a.intro:
        norm_clip(a.intro, f"{wd}/in.mp4"); clips.append(f"{wd}/in.mp4")
    clips.append(body)
    if a.outro:
        norm_clip(a.outro, f"{wd}/out.mp4"); clips.append(f"{wd}/out.mp4")
    if len(clips) == 1:
        os.replace(body, a.out)
    else:
        inputs, fv, fa, off, prev_v, prev_a = [], [], [], 0.0, "0:v", "0:a"
        for c in clips:
            inputs += ["-i", c]
        for k in range(1, len(clips)):
            tr = a.transition_in if (a.intro and k == 1) else a.transition_out
            td = 0.8 if tr == a.transition_in else 1.0
            off += dur(clips[k - 1]) - td
            fv.append(f"[{prev_v}][{k}:v]xfade=transition={tr}:duration={td}:offset={off:.3f}[v{k}]")
            fa.append(f"[{prev_a}][{k}:a]acrossfade=d={td}[a{k}]")
            prev_v, prev_a = f"v{k}", f"a{k}"
        sh("ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(fv + fa),
           "-map", f"[{prev_v}]", "-map", f"[{prev_a}]", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", a.out)
    print(f"done  {a.out}  {dur(a.out):.1f}s")


if __name__ == "__main__":
    sys.exit(main())
