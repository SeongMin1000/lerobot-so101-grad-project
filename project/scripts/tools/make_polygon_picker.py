#!/usr/bin/env python3
"""Build a browser page for clicking the target zone's four inner corners.

The OpenCV build here is headless, so `cv2.imshow` cannot be used to pick the
corners interactively. This grabs one frame, embeds it in a self-contained
HTML page, and lets the browser do the clicking; the page prints the polygon
in the exact form `apply_target_polygon.py` expects.

    python project/scripts/tools/make_polygon_picker.py
    xdg-open /home/eslab/lerobot/var/dryrun_debug/pick_polygon.html
"""

import base64

import cv2

from lerobot.grad_project.paths import lerobot_root

OUT = "var/dryrun_debug/pick_polygon.html"

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>target zone corners</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#111; color:#eee; margin:16px; }}
  #wrap {{ position:relative; display:inline-block; }}
  img {{ display:block; image-rendering: pixelated; }}
  canvas {{ position:absolute; left:0; top:0; cursor:crosshair; }}
  #out {{ font-family: ui-monospace, monospace; font-size:15px; background:#000; color:#6f6;
          padding:10px; margin-top:12px; user-select:all; }}
  button {{ font-size:14px; padding:6px 12px; margin-right:8px; }}
  p {{ max-width:44em; line-height:1.5; }}
</style>
<h2>Click the four inner corners of the white area</h2>
<p>In this order, <b>as they appear on screen</b>:
   <b>1</b> top-left, <b>2</b> top-right, <b>3</b> bottom-right, <b>4</b> bottom-left.
   Click inside the dark border, on the white part. Zoom the page in if it helps.</p>
<button onclick="reset()">start over</button>
<div id="wrap">
  <img id="im" src="data:image/jpeg;base64,{b64}" width="{w}" height="{h}">
  <canvas id="cv" width="{w}" height="{h}"></canvas>
</div>
<div id="out">click 4 corners…</div>
<script>
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const names = ['TL','TR','BR','BL'];
let pts = [];
function reset() {{ pts = []; draw(); }}
function draw() {{
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.lineWidth = 2; ctx.strokeStyle = '#0f0'; ctx.fillStyle = '#0f0';
  if (pts.length > 1) {{
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for (const p of pts.slice(1)) ctx.lineTo(p[0], p[1]);
    if (pts.length === 4) ctx.closePath();
    ctx.stroke();
  }}
  pts.forEach((p,i) => {{
    ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, 7); ctx.fill();
    ctx.font = '14px sans-serif'; ctx.fillText(names[i], p[0]+7, p[1]-7);
  }});
  document.getElementById('out').textContent = pts.length === 4
    ? JSON.stringify(pts)
    : 'click ' + names[pts.length] + '  (' + pts.length + '/4)';
}}
cv.addEventListener('click', e => {{
  if (pts.length >= 4) return;
  const r = cv.getBoundingClientRect();
  pts.push([Math.round((e.clientX - r.left) * cv.width / r.width),
            Math.round((e.clientY - r.top) * cv.height / r.height)]);
  draw();
}});
draw();
</script>
"""


def main() -> None:
    capture = cv2.VideoCapture("/dev/cam_top", cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame = None
    for _ in range(10):
        ok, frame = capture.read()
    capture.release()
    if frame is None:
        raise SystemExit("could not read from /dev/cam_top")

    height, width = frame.shape[:2]
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise SystemExit("failed to encode the frame")

    out = lerobot_root() / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        PAGE.format(b64=base64.b64encode(buffer).decode(), w=width, h=height), encoding="utf-8"
    )
    print(f"open this in a browser:\n  {out}")


if __name__ == "__main__":
    main()
