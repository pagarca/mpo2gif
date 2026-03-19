import argparse
import glob
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image


def get_raw_frames(mpo_path):
    img = Image.open(mpo_path)
    if img.n_frames < 2:
        return None, None
    left = img.copy()
    img.seek(1)
    right = img.copy()
    img.close()
    return left, right


def crop_frames(left, right, crop_px):
    w, h = left.size
    cropped_w = w - crop_px
    if cropped_w % 2 != 0:
        crop_px += 1
    return left.crop((0, 0, w - crop_px, h)), right.crop((crop_px, 0, w, h))


def make_wiggle_gif(left, right, output_path, duration_ms):
    left.save(
        output_path,
        save_all=True,
        append_images=[right],
        duration=duration_ms,
        loop=0,
    )


def find_optimal_crop(left, right):
    left_cv = cv2.cvtColor(np.array(left), cv2.COLOR_RGB2GRAY)
    right_cv = cv2.cvtColor(np.array(right), cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(left_cv, None)
    kp2, des2 = orb.detectAndCompute(right_cv, None)

    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        return 0

    disparities = []
    for m in matches:
        dx = kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]
        disparities.append(dx)

    median_disp = float(np.median(disparities))
    return max(0, round(abs(median_disp) / 5) * 5)


def make_wiggle_mp4(left, right, output_path, duration_ms, length_s):
    fps = round(1000 / duration_ms)
    total_frames = fps * length_s

    tmpdir = tempfile.mkdtemp()

    try:
        for i in range(total_frames):
            frame = left if i % 2 == 0 else right
            frame.save(os.path.join(tmpdir, f"frame_{i:04d}.png"))

        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", os.path.join(tmpdir, "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18",
                output_path,
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    finally:
        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Nintendo 3DS MPO stereoscopic images to wiggle GIFs/MP4s.",
    )
    parser.add_argument(
        "-i", "--input",
        default=".",
        help="Path to a single .MPO file or a directory containing .MPO files (default: current directory)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for generated files (default: <input-dir>/gifs)",
    )
    parser.add_argument(
        "-c", "--crop",
        type=int,
        default=None,
        help="Manual crop in pixels to reduce stereo offset (default: auto-detect)",
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=150,
        help="Frame duration in milliseconds (default: 150)",
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help="Also generate MP4 videos",
    )
    parser.add_argument(
        "--mp4-length",
        type=int,
        default=5,
        help="MP4 video length in seconds (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = os.path.abspath(args.input)

    if os.path.isfile(input_path):
        mpo_files = [input_path]
        input_dir = os.path.dirname(input_path)
    else:
        input_dir = input_path
        mpo_files = sorted(glob.glob(os.path.join(input_dir, "*.MPO")))

    if not mpo_files:
        print(f"No .MPO files found in {input_path}")
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(input_dir, "gifs")

    os.makedirs(output_dir, exist_ok=True)

    for mpo_path in mpo_files:
        basename = os.path.splitext(os.path.basename(mpo_path))[0]

        try:
            left, right = get_raw_frames(mpo_path)
            if left is None:
                print(f"SKIP: {basename} (not enough frames)")
                continue

            if args.crop is not None:
                crop_px = args.crop
                print(f"{basename}: crop = {crop_px}px (manual)")
            else:
                crop_px = find_optimal_crop(left, right)
                print(f"{basename}: crop = {crop_px}px (auto)")

            left_cropped, right_cropped = crop_frames(left, right, crop_px)

            gif_path = os.path.join(output_dir, f"{basename}.gif")
            make_wiggle_gif(left_cropped, right_cropped, gif_path, args.duration)
            saved = f"{basename}.gif"

            if args.mp4:
                mp4_path = os.path.join(output_dir, f"{basename}.mp4")
                make_wiggle_mp4(left_cropped, right_cropped, mp4_path, args.duration, args.mp4_length)
                saved += f" / {basename}.mp4"

            print(f"  Saved {saved}")

        except Exception as e:
            print(f"ERROR: {basename}: {e}")


if __name__ == "__main__":
    main()
