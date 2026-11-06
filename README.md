# TrafficView

Count vehicles on public traffic cameras. Give it a stream URL (for example a camera linked from [opencctv.org](https://opencctv.org)), draw a few lines across the lanes, and it detects, tracks and counts cars, motorcycles, buses and trucks per lane and direction. Counts are written to a CSV file once per minute.

It runs on a normal laptop (GPU optional) and is built for the typical weakness of public cameras: low resolution, low frame rate and small, distant vehicles.

> **Note:** The command-line interface and console output are in German (e.g. `--spuren`, `--zusammenfassen`, `mit_pfeil`). This README documents the exact option names.

## Features

- **Works with common stream types:** HLS (`.m3u8`), MJPEG, RTSP, or single JPEG snapshot URLs polled at an interval.
- **Per-lane counting lines:** draw as many counting lines as you like with the mouse. Each line counts both directions separately and splits by vehicle type.
- **Multiple regions of interest (ROIs):** for example one per driving direction. Each region is cropped, upscaled and processed with its own model instance and tracker, which improves detection of small, distant vehicles and cuts compute compared to processing the whole frame.
- **Built-in tracker for jumpy, low-quality video:** a lightweight motion tracker that needs no confirmation phase and tolerates large jumps between processed frames. It uses hysteresis (new vehicles need a confident detection, but existing tracks can be kept alive by weaker detections, which prevents flickering) and plausibility checks (search area along the direction of travel, no backward jumps, similar box size and shape, speed limit). The Ultralytics ByteTrack tracker is available as an alternative.
- **Filters false detections:** objects that do not move (lane markings, signs, painted arrows) are not counted or shown as vehicles. A track only counts as a vehicle after it has moved steadily along a straight path.
- **Clean exits:** when a vehicle leaves a region, its track ends immediately. No ghost boxes are extrapolated outside the region.
- **Counts vehicles that vanish right before the line:** if a tracked vehicle is no longer detected shortly before a counting line but its predicted (yellow) box crosses the line, it is counted, with the type it was recognized as (majority over all frames in which it was detected). If the boxes of two crossings would have overlapped closely at the line at about the same time, only the first one is counted, so a vehicle that reappears as a new track is not counted twice.
- **Stable vehicle type:** the type of each vehicle is decided by a confidence-weighted vote over all frames in which it was seen. Optionally merge all types into a single `Fahrzeug` (vehicle) class.
- **Adapts to your hardware:** `--skip auto` skips just enough frames to keep up with a live stream. Uses CUDA or Apple MPS automatically if available.
- **Auto-reconnect** after dropped connections.
- **Live preview and debug view** with counting lines, regions, boxes, track IDs and "ghost" boxes for tracked vehicles the model missed in the current frame.

## How it works

```
stream URL ─► frame ─► crop to ROI(s) ─► YOLO detection ─► tracker ─► line-crossing test ─► per-minute CSV
                                                              │
                                              vehicle type = confidence-weighted vote over the track
```

A vehicle is counted when the movement of its tracked position between two processed frames crosses a counting line segment. Each vehicle is counted once per line. Brief detection dropouts do not lose vehicles, because the test compares the last known position with the new one. If a track is only classified as a vehicle after the crossing (for example because the line is very close to where vehicles first appear), its recent path is checked retroactively.

If a tracked vehicle drops out of detection right before the line, the tracker keeps it for a few frames and shows its predicted position as a yellow box. When that box crosses the line, the vehicle is counted. To avoid counting the same vehicle twice (for example when it comes back as a new track), the tool estimates where and when each crossing happened. If a crossing involving a yellow box would have overlapped closely with a crossing counted a moment earlier, it is not counted again.

## Requirements

- Python 3.9 or newer (recommended)
- A desktop environment for `--setup` and `--show` (they open windows). Counting itself also works headless, see [Headless use](#headless-use).
- Optional: NVIDIA GPU (CUDA) or Apple Silicon (MPS) for speed

## Installation

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install ultralytics opencv-python requests numpy
```

YOLO model weights are downloaded automatically on first use.

## Quick start

**1. Get a stream URL.** Open a camera on [opencctv.org](https://opencctv.org) or the operator's page and find the live stream address. It usually ends in `.m3u8`, `.mjpg` or `.jpg`. If your browser shows it, open the developer tools (F12), go to the Network tab and filter for `m3u8`.

**2. Set up regions and counting lines** (once per camera):

```bash
python trafficview.py "https://example.com/live/cam1/playlist.m3u8" --setup
```

A window with a still frame opens:

| Key / action | Effect |
|---|---|
| 2 mouse clicks | Draw a counting line (default mode) |
| `r`, then 2 clicks | Draw a region of interest (opposite corners). Press `r` again for the next region |
| `u` | Undo the last point, region or line |
| `c` | Delete all regions (lines are kept) |
| `Enter` | Save to `spuren.json` |
| `q` / `Esc` | Cancel |

Tips: draw one line per lane, each inside the region of that lane. Place lines where traffic flows freely, not at a stop line. Regions should **not overlap**, otherwise a vehicle in the overlap can be counted twice (a warning is printed). The small arrow on each line shows the `mit_pfeil` direction.

**3. Count:**

```bash
python trafficview.py "https://example.com/live/cam1/playlist.m3u8" --skip auto --show
```

Press `q` in the preview window or `Ctrl+C` in the terminal to stop. A summary per line, direction and type is printed on exit.

## Examples

```bash
# Better accuracy on small vehicles with a larger model
python trafficview.py "<url>" --skip auto --model yolo11s.pt --show

# Only the total traffic volume matters (avoids car/truck/bus confusion)
python trafficview.py "<url>" --skip auto --zusammenfassen

# Define regions on the command line instead of using spuren.json
python trafficview.py "<url>" --roi 0,0.4,0.5,1 --roi 0.5,0.4,1,1

# Camera that only provides a single JPEG snapshot (polled every 0.5 s)
python trafficview.py "http://example.com/snapshot.jpg" --snapshot --interval 0.5

# See raw detections vs. tracked vehicles and timing
python trafficview.py "<url>" --show --debug
```

## Options

| Option | Default | Description |
|---|---|---|
| `url` | required | Stream or snapshot URL |
| `--setup` | off | Draw regions and counting lines with the mouse, then continue |
| `--spuren FILE` | `spuren.json` | Configuration file (regions and lines) |
| `--roi x1,y1,x2,y2` | from config | Region of interest in relative coordinates (0 to 1). Repeatable. Overrides the config file |
| `--line H` | `0.55` | Height (0 to 1) of a default horizontal line if no config file exists |
| `--model NAME` | `yolov8n.pt` | Any Ultralytics YOLO model (`n` = fastest, `s`/`m` = more accurate) |
| `--conf C` | `0.2` | Minimum confidence to start a new track |
| `--conf-low C` | `0.08` | Lower confidence: weaker detections may keep an existing track alive (`--tracker einfach` only) |
| `--ohne-geisterzaehlung` | off | Do not count vehicles whose yellow box crosses the line while they are not detected (`--tracker einfach` only) |
| `--ueberlappung F` | `0.3` | If the boxes of two crossings would have overlapped by at least this share of the smaller box and one of them involves a yellow box, only the first is counted. `0` disables the check (`--tracker einfach` only) |
| `--min-bewegung F` | `0.25` | A track only counts as a vehicle after moving steadily by this many box sizes. Filters static false detections such as lane markings. `0` disables the filter (`--tracker einfach` only) |
| `--imgsz N` | `0` (auto) | YOLO input size. Auto upscales small crops about 1.5x (min 640, max 1280) |
| `--skip N\|auto` | `1` | Evaluate only every Nth frame, or `auto` to match your hardware |
| `--max-skip N` | `4` | Upper limit for `--skip auto` |
| `--tracker` | `einfach` | `einfach` (built-in, robust to frame jumps) or `bytetrack` (Ultralytics) |
| `--zusammenfassen` | off | Count all vehicle types as one class `Fahrzeug` |
| `--csv FILE` | `verkehr.csv` | Output file for the per-minute counts |
| `--snapshot` | off | URL is a single JPEG image that is polled |
| `--interval S` | `0.5` | Polling interval in seconds (snapshot mode) |
| `--show` | off | Live preview window |
| `--debug` | off | Print detection/tracking statistics; draws raw detections in red with `--tracker bytetrack`. Costs extra compute |

### Preview colors

- **Green box:** detected and tracked vehicle (label shows type and track ID)
- **Thin yellow box `#id (nicht erkannt)`:** the tracker is holding a vehicle the model did not detect in this frame (shown for at most a few frames and only inside its region). If it crosses a counting line, the vehicle is counted
- **Colored lines with an arrow:** counting lines; the arrow marks the `mit_pfeil` direction
- **Yellow rectangles:** regions of interest

## Output

On exit, the console shows the totals per line, direction and type, plus how many crossings were counted via a yellow box and how many were not counted because of close overlap. A row is appended to the CSV for every counting line and direction at the end of each minute:

```csv
zeit,linie,richtung,Pkw,Motorrad,Bus,Lkw,gesamt
2026-11-04T21:11:00,Spur1,mit_pfeil,18,0,1,2,21
2026-11-04T21:11:00,Spur1,gegen_pfeil,0,0,0,0,0
2026-11-04T21:11:00,Spur2,mit_pfeil,0,0,0,0,0
2026-11-04T21:11:00,Spur2,gegen_pfeil,15,1,0,3,19
```

(Illustrative values.) `richtung` is `mit_pfeil` (with the arrow drawn in setup) or `gegen_pfeil` (against it). With `--zusammenfassen` there is a single `Fahrzeug` column. The line names (`Spur1`, ...) can be renamed in `spuren.json`.

## Configuration file

`spuren.json` is created by `--setup`. All coordinates are relative to the frame size (0 to 1), so it keeps working if the stream resolution changes. You can also write it by hand, for example on a headless machine:

```json
{
  "rois": [
    [0.00, 0.40, 0.50, 1.00],
    [0.50, 0.40, 1.00, 1.00]
  ],
  "linien": [
    { "name": "Lane left",  "p1": [0.05, 0.65], "p2": [0.45, 0.65] },
    { "name": "Lane right", "p1": [0.55, 0.65], "p2": [0.95, 0.65] }
  ]
}
```

Older files that contain only a list of lines, or a single `roi`, are still read.

## Tuning for low-quality cameras

Public cameras are often 720x480 or smaller. If detections are patchy:

1. **Draw tighter regions** around the road. This enlarges vehicles relative to the model input and removes false detections from sky, buildings and sidewalks.
2. **Use a larger model** (`--model yolo11s.pt` or `yolo11m.pt`). This helps most with classification, especially rear views of cars that smaller models often mistake for trucks.
3. **Lower `--conf`** (for example `0.15`). The hysteresis in the built-in tracker limits the extra false positives.
4. **Use `--zusammenfassen`** if you only need total volume and car/truck/bus confusion is a problem.
5. **Avoid large `--skip` values** (4 and above) on fast traffic. `--skip auto` is usually a good choice.

### Lane markings or signs detected as vehicles

Static false detections are filtered automatically (`--min-bewegung`), and they are neither counted nor drawn. If you still see them, check these in order:

1. **Exclude them with the region.** Draw the region (`--setup`, key `r`) so that it only contains the lanes. Markings at the edge of the road, stop lines and crosswalks outside the region can then not be detected at all.
2. **Raise `--conf`** (for example `0.3`). New tracks need this confidence, weaker detections can only continue existing tracks.
3. **Raise `--min-bewegung`** (for example `0.5`) if slow, jittery false detections get through. Lower it, or set `0`, if creeping vehicles in stop-and-go traffic are missed.

### Performance

Larger models are the most expensive lever. In order of effect:

- Smaller or tighter regions reduce the pixels the model has to process.
- `--imgsz 480` (or `512`) instead of the automatic value (up to 1280) trades some accuracy on small vehicles for speed.
- `--skip auto` keeps up with a live stream by dropping frames. The built-in tracker handles frame jumps well, so higher values are less harmful than with other trackers.
- Detection is stateless with the built-in tracker, so one model instance serves all regions (less memory, faster start).
- Optionally export the model to a faster runtime for your hardware (for example OpenVINO on Intel CPUs, see the Ultralytics export documentation) and pass its path to `--model`. This is not tested with TrafficView.

## Troubleshooting

**`[tls @ ...] Error in the pull function` / `IO error: End of file`**
FFmpeg reports that the server closed the HTTPS connection. If counting continues it is harmless (servers often close connections after each segment). If it loops with no picture: test the URL in VLC or with `ffprobe "<url>"`, check that it has not expired (some URLs contain tokens), and reduce the load with `--skip auto` or a smaller model, because falling behind a live stream also causes this. TrafficView enables FFmpeg reconnect options by default.

**The server rejects the request / no video**
Some servers require a `Referer` or `User-Agent` header. Set it before starting (Linux/macOS):

```bash
export OPENCV_FFMPEG_CAPTURE_OPTIONS="headers;Referer: https://page-where-the-camera-is-embedded/\r\nUser-Agent: Mozilla/5.0\r\n|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5"
```

**Nothing is counted**
Run with `--show --debug`. If vehicles are detected but not counted, the counting lines do not lie across their path, or they are outside every region of interest. Lines must be crossed completely by a tracked vehicle.

**Too slow**
Use `--skip auto`, `--model yolov8n.pt`, smaller regions, or a GPU. Check the `ms/Frame` value in the preview.

**A warning mentions `half` / `quantize`**
Handled automatically for both old and new Ultralytics versions.

### Headless use

`--setup` and `--show` need a window. On a server, create `spuren.json` by hand (see above) or create it on your desktop and copy it over, then run without `--show`. On machines without a display you can install `opencv-python-headless` instead of `opencv-python`.

## Limitations

- Accuracy depends heavily on camera resolution, viewing angle, lighting and weather. Always verify the counts against a manual count for your camera before relying on them.
- The built-in tracker was validated on simulated traffic (see [Tests](#tests)) and in first runs on live public streams. There is no formal accuracy benchmark on real footage. In dense or stop-and-go traffic, identity switches can occur.
- Counting via the yellow box only helps for vehicles that are lost shortly before the line. The box is shown for at most 4 frames with decaying speed. In simulations (about 10 px per frame, 40 px tall boxes) it reached the line for vehicles lost up to roughly a third of a box height before it. Vehicles lost further away are not counted, as before.
- The overlap check only applies when a yellow box is involved. Two vehicles that are both detected normally are both counted, even if their boxes overlap. If a yellow box and another vehicle really do overlap closely at the line, only the first crossing is counted.
- The tracker holds a lost vehicle for at most 8 frames. If the same vehicle reappears later as a new track, it can be counted a second time.
- A vehicle needs about three processed frames of steady movement before it counts as a vehicle. Counting lines very close to the edge of a region still work (the path is checked retroactively), but vehicles that stand still for the whole time they are visible are neither shown nor counted. They do not cross lines, so counts are not affected.
- Vehicle type classification is the weakest part with low-resolution video. Use `--zusammenfassen` when you only need totals.
- Only the COCO classes car, motorcycle, bus and truck are counted. Bicycles and pedestrians are not.
- Stream timestamps in the CSV are your computer's clock. HLS streams are typically delayed by 5 to 30 seconds.

## Tests

The tests need neither PyTorch, YOLO nor a network connection:

```bash
python -m unittest -v test_trafficview
```

They cover the geometry, the tracker (exits, speed limit, static false detections), the crossing counter, simulated traffic with typical camera problems (clipped boxes at region borders, flickering detections of lane markings, detection dropouts, large frame jumps) and an end-to-end run of the main loop on a generated video with a stand-in for YOLO (including a vehicle that vanishes right before the line). The simulations are checked against count accuracy, the behavior of the ghost boxes and the double-count protection. They do not replace a manual check on real footage.

## Responsible use

- Only use cameras that are publicly accessible, and respect the terms of use of the camera operator and of the directory that lists them.
- Do not poll at excessive rates and do not redistribute video.
- TrafficView counts vehicle types only. It does not read license plates or identify people, and it should not be extended to do so without a lawful basis.

## Acknowledgements and licensing

- Detection models and tooling by [Ultralytics](https://github.com/ultralytics/ultralytics). Please check the Ultralytics license terms (AGPL-3.0 or a commercial license) for how they apply to your use and distribution.
- Video I/O by [OpenCV](https://opencv.org/).
- Camera directory: [opencctv.org](https://opencctv.org). This project is not affiliated with it.

## License

See details about the MIT license in the seperate License file.
