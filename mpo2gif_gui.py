import argparse
import glob
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from PIL.Image import Resampling

from mpo2gif import get_raw_frames, make_wiggle_gif, make_wiggle_mp4


def find_matching_point(
    left: Image.Image,
    right: Image.Image,
    point: tuple[int, int],
    window_size: int = 40,
    search_radius: int = 150,
) -> tuple[int, int]:
    """Find matching point in right image using template matching.

    Extracts a small template around the point in the left image and
    searches for it in the right image using normalized cross-correlation.
    """
    left_gray = cv2.cvtColor(np.array(left), cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(np.array(right), cv2.COLOR_RGB2GRAY)

    px, py = int(point[0]), int(point[1])
    w, h = left.size

    x1 = max(0, px - window_size)
    y1 = max(0, py - window_size)
    x2 = min(w, px + window_size)
    y2 = min(h, py + window_size)
    template = left_gray[y1:y2, x1:x2]

    if template.size == 0:
        return point

    sx1 = max(0, px - search_radius)
    sy1 = max(0, py - search_radius)
    sx2 = min(w, px + search_radius)
    sy2 = min(h, py + search_radius)
    search_area = right_gray[sy1:sy2, sx1:sx2]

    if (
        search_area.size == 0
        or search_area.shape[0] < template.shape[0]
        or search_area.shape[1] < template.shape[1]
    ):
        return point

    result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.3:
        return point

    match_x = max_loc[0] + sx1 + (x2 - x1) // 2
    match_y = max_loc[1] + sy1 + (y2 - y1) // 2

    return (match_x, match_y)


def _shift_image(img: Image.Image, shift_x: int, shift_y: int) -> Image.Image:
    """Shift image by (shift_x, shift_y) pixels, padding edges with nearest pixels."""
    w, h = img.size
    if shift_x == 0 and shift_y == 0:
        return img.copy()

    arr = np.array(img)
    y_indices, x_indices = np.mgrid[0:h, 0:w]
    src_y = np.clip(y_indices - shift_y, 0, h - 1)
    src_x = np.clip(x_indices - shift_x, 0, w - 1)
    out = arr[src_y, src_x]

    return Image.fromarray(out)


def align_to_focal_point(
    left: Image.Image,
    right: Image.Image,
    left_pt: tuple[int, int],
    right_pt: tuple[int, int],
) -> tuple[Image.Image, Image.Image, int, int]:
    """Align images so the focal point stays stationary during the wiggle effect.

    Shifts both images symmetrically by half the disparity so that the
    focal point appears at the same screen position in both frames.
    Returns (left_aligned, right_aligned, half_dx, half_dy).
    """
    lx, ly = left_pt
    rx, ry = right_pt

    dx = lx - rx
    dy = ly - ry

    half_dx = round(dx / 2)
    half_dy = round(dy / 2)

    left_aligned = _shift_image(left, -half_dx, -half_dy)
    right_aligned = _shift_image(right, half_dx, half_dy)

    return left_aligned, right_aligned, half_dx, half_dy


def crop_border_glitch(
    left: Image.Image,
    right: Image.Image,
    half_dx: int,
    half_dy: int,
) -> tuple[Image.Image, Image.Image]:
    """Crop away edge regions that were padded by the alignment shift.

    When images are shifted to align a focal point, the edges get duplicated
    with nearest-neighbor padding, causing visible glitches during the wiggle.
    This crops both images to only the region where neither image has padding.
    """
    crop_px = abs(half_dx)
    crop_py = abs(half_dy)

    w, h = left.size
    box = (crop_px, crop_py, w - crop_px, h - crop_py)

    if box[0] >= box[2] or box[1] >= box[3]:
        return left, right

    return left.crop(box), right.crop(box)


def rotate_pair(
    left: Image.Image, right: Image.Image, angle: int
) -> tuple[Image.Image, Image.Image]:
    """Rotate both images by the given angle (degrees, counter-clockwise)."""
    if angle == 0:
        return left.copy(), right.copy()
    return left.rotate(angle, expand=True), right.rotate(angle, expand=True)


class FocalPointGUI:
    CANVAS_H = 400
    PREVIEW_H = 200
    MARKER_COLOR = "#FF0000"
    MARKER_SIZE = 12

    def __init__(self, root: tk.Tk, initial_path: str | None = None):
        self.root = root
        self.root.title("MPO to Wiggle GIF \u2014 Focal Point Selector")
        self.root.geometry("1200x850")
        self.root.minsize(800, 600)

        self.file_list: list[str] = []
        self.file_index = -1
        self.left_orig: Image.Image | None = None
        self.right_orig: Image.Image | None = None
        self.left_img: Image.Image | None = None
        self.right_img: Image.Image | None = None
        self.rotation = 0
        self.focal_left: tuple[int, int] | None = None
        self.focal_right: tuple[int, int] | None = None
        self.half_dx = 0
        self.half_dy = 0
        self.left_aligned: Image.Image | None = None
        self.right_aligned: Image.Image | None = None
        self.preview_running = False
        self.preview_frame = 0
        self._preview_job: str | None = None

        self._left_photo: ImageTk.PhotoImage | None = None
        self._right_photo: ImageTk.PhotoImage | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None

        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if initial_path:
            self._load_path(initial_path)

    def _build_ui(self):
        toolbar = tk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=5, pady=(5, 0))

        nav_frame = tk.Frame(toolbar)
        nav_frame.pack(side=tk.LEFT)

        tk.Button(nav_frame, text="Open File", command=self._open_file).pack(
            side=tk.LEFT, padx=2
        )
        tk.Button(nav_frame, text="Open Dir", command=self._open_dir).pack(
            side=tk.LEFT, padx=2
        )
        tk.Button(nav_frame, text="\u25C4 Prev", command=self._prev_file).pack(
            side=tk.LEFT, padx=2
        )
        tk.Button(nav_frame, text="Next \u25BA", command=self._next_file).pack(
            side=tk.LEFT, padx=2
        )

        self.file_label = tk.Label(toolbar, text="No file loaded", anchor=tk.W)
        self.file_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        rot_frame = tk.LabelFrame(toolbar, text="Rotate")
        rot_frame.pack(side=tk.RIGHT)

        tk.Button(
            rot_frame, text="90\u00B0 CW", width=7, command=lambda: self._rotate(270)
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            rot_frame, text="90\u00B0 CCW", width=7, command=lambda: self._rotate(90)
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            rot_frame, text="180\u00B0", width=5, command=lambda: self._rotate(180)
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            rot_frame, text="Reset", width=5, command=lambda: self._rotate(0)
        ).pack(side=tk.LEFT, padx=2)

        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = tk.LabelFrame(canvas_frame, text="Left (click to set focal point)")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))

        self.left_canvas = tk.Canvas(left_frame, bg="gray20")
        self.left_canvas.pack(fill=tk.BOTH, expand=True)
        self.left_canvas.bind("<Button-1>", self._on_left_click)
        self.left_canvas.bind("<Configure>", lambda e: self._update_display())

        right_frame = tk.LabelFrame(canvas_frame, text="Right (auto-matched)")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 2))

        self.right_canvas = tk.Canvas(right_frame, bg="gray20")
        self.right_canvas.pack(fill=tk.BOTH, expand=True)
        self.right_canvas.bind("<Configure>", lambda e: self._update_display())

        preview_col = tk.Frame(canvas_frame)
        preview_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))

        ctrl_frame = tk.Frame(preview_col)
        ctrl_frame.pack(fill=tk.X, pady=(0, 2))

        tk.Label(ctrl_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=150)
        tk.Scale(
            ctrl_frame,
            from_=50,
            to=500,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            showvalue=True,
            length=120,
            label="ms",
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(ctrl_frame, text="\u25B6 Preview", command=self._toggle_preview).pack(
            side=tk.LEFT, padx=2
        )
        self.crop_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            ctrl_frame,
            text="Crop borders",
            variable=self.crop_var,
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl_frame, text="Save GIF", command=self._save_gif).pack(
            side=tk.LEFT, padx=2
        )
        tk.Button(ctrl_frame, text="Save MP4", command=self._save_mp4).pack(
            side=tk.LEFT, padx=2
        )

        preview_frame = tk.LabelFrame(preview_col, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_canvas = tk.Canvas(preview_frame, bg="black")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        info_frame = tk.Frame(self.root)
        info_frame.pack(fill=tk.X, padx=5)

        self.info_var = tk.StringVar(value="Click on the left image to set the focal point")
        tk.Label(info_frame, textvariable=self.info_var, anchor=tk.W).pack(fill=tk.X)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        ).pack(fill=tk.X)

    def _load_path(self, path: str):
        path = os.path.abspath(path)
        if os.path.isfile(path):
            directory = os.path.dirname(path)
            self.file_list = sorted(glob.glob(os.path.join(directory, "*.MPO")))
            self.file_index = (
                self.file_list.index(path) if path in self.file_list else 0
            )
            if not self.file_list:
                self.file_list = [path]
                self.file_index = 0
            self._load_file(self.file_list[self.file_index])
        elif os.path.isdir(path):
            self.file_list = sorted(glob.glob(os.path.join(path, "*.MPO")))
            if self.file_list:
                self.file_index = 0
                self._load_file(self.file_list[0])
            else:
                messagebox.showinfo("No Files", f"No .MPO files found in {path}")
        else:
            messagebox.showerror("Error", f"Path not found: {path}")

    def _load_file(self, filepath: str):
        self._stop_preview()
        self.focal_left = None
        self.focal_right = None
        self.rotation = 0
        self.left_aligned = None
        self.right_aligned = None

        try:
            left, right = get_raw_frames(filepath)
            if left is None or right is None:
                messagebox.showwarning("Skip", f"Not enough frames in {filepath}")
                return
            self.left_orig = left
            self.right_orig = right
            self.left_img = left.copy()
            self.right_img = right.copy()

            nav_text = os.path.basename(filepath)
            if len(self.file_list) > 1:
                nav_text += f"  ({self.file_index + 1}/{len(self.file_list)})"
            self.file_label.config(text=nav_text)
            self.info_var.set("Click on the left image to set the focal point")
            self.status_var.set(f"Loaded: {os.path.basename(filepath)}")
            self._update_display()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {filepath}:\n{e}")

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open MPO file",
            filetypes=[("MPO files", "*.MPO"), ("All files", "*.*")],
        )
        if path:
            self._load_path(path)

    def _open_dir(self):
        path = filedialog.askdirectory(title="Open Directory with MPO files")
        if path:
            self._load_path(path)

    def _prev_file(self):
        if not self.file_list:
            return
        self.file_index = (self.file_index - 1) % len(self.file_list)
        self._load_file(self.file_list[self.file_index])

    def _next_file(self):
        if not self.file_list:
            return
        self.file_index = (self.file_index + 1) % len(self.file_list)
        self._load_file(self.file_list[self.file_index])

    def _rotate(self, angle: int):
        if self.left_orig is None or self.right_orig is None:
            return

        if angle == 0:
            self.rotation = 0
        else:
            self.rotation = (self.rotation + angle) % 360

        left_orig = self.left_orig
        right_orig = self.right_orig
        self.left_img, self.right_img = rotate_pair(
            left_orig, right_orig, self.rotation
        )

        self.focal_left = None
        self.focal_right = None
        self.left_aligned = None
        self.right_aligned = None

        self.info_var.set("Click on the left image to set the focal point")
        self._update_display()

    def _canvas_to_img(
        self,
        canvas_x: float,
        canvas_y: float,
        canvas: tk.Canvas,
        img: Image.Image,
    ) -> tuple[float, float] | None:
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        iw, ih = img.size

        if cw <= 1 or ch <= 1 or iw <= 0 or ih <= 0:
            return None

        scale = min(cw / iw, ch / ih)
        display_w = iw * scale
        display_h = ih * scale
        offset_x = (cw - display_w) / 2
        offset_y = (ch - display_h) / 2

        img_x = (canvas_x - offset_x) / scale
        img_y = (canvas_y - offset_y) / scale

        if 0 <= img_x < iw and 0 <= img_y < ih:
            return (img_x, img_y)
        return None

    def _get_display_params(
        self, canvas: tk.Canvas, img: Image.Image
    ) -> tuple[float, float, float, int, int]:
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        iw, ih = img.size

        if cw <= 1 or ch <= 1 or iw <= 0 or ih <= 0:
            return (1.0, 0.0, 0.0, iw, ih)

        scale = min(cw / iw, ch / ih)
        display_w = int(iw * scale)
        display_h = int(ih * scale)
        offset_x = (cw - display_w) / 2
        offset_y = (ch - display_h) / 2

        return (scale, offset_x, offset_y, display_w, display_h)

    def _on_left_click(self, event):
        if self.left_img is None or self.right_img is None:
            return

        left_img = self.left_img
        right_img = self.right_img

        coords = self._canvas_to_img(event.x, event.y, self.left_canvas, left_img)
        if coords is None:
            return

        img_x, img_y = int(round(coords[0])), int(round(coords[1]))

        self.status_var.set(f"Finding match for ({img_x}, {img_y})...")
        self.root.update_idletasks()

        self.focal_left = (img_x, img_y)
        self.focal_right = find_matching_point(left_img, right_img, (img_x, img_y))

        dx = self.focal_left[0] - self.focal_right[0]
        dy = self.focal_left[1] - self.focal_right[1]

        self.info_var.set(
            f"Focal: ({img_x}, {img_y}) \u2192 ({self.focal_right[0]}, {self.focal_right[1]}) "
            f"| Disparity: dx={dx}, dy={dy}"
        )

        self.left_aligned, self.right_aligned, self.half_dx, self.half_dy = (
            align_to_focal_point(
                left_img, right_img, self.focal_left, self.focal_right
            )
        )

        self.status_var.set(
            f"Match found at ({self.focal_right[0]}, {self.focal_right[1]})"
        )
        self._update_display()
        if self.preview_running:
            self._stop_preview()
            self._start_preview()

    def _update_display(self):
        if self.left_img is None or self.right_img is None:
            return

        left_img = self.left_img
        right_img = self.right_img
        self._draw_image_with_marker(self.left_canvas, left_img, self.focal_left)
        self._draw_image_with_marker(self.right_canvas, right_img, self.focal_right)

    def _draw_image_with_marker(
        self,
        canvas: tk.Canvas,
        img: Image.Image,
        marker_pt: tuple[int, int] | None,
    ):
        scale, ox, oy, dw, dh = self._get_display_params(canvas, img)
        if dw <= 0 or dh <= 0:
            return

        display_img = img.resize((dw, dh), Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_img)

        canvas.delete("all")
        canvas.create_image(ox, oy, anchor=tk.NW, image=photo)

        if marker_pt:
            mx = ox + marker_pt[0] * scale
            my = oy + marker_pt[1] * scale
            r = self.MARKER_SIZE
            canvas.create_line(
                mx - r, my, mx + r, my, fill=self.MARKER_COLOR, width=2
            )
            canvas.create_line(
                mx, my - r, mx, my + r, fill=self.MARKER_COLOR, width=2
            )
            canvas.create_oval(
                mx - r // 2,
                my - r // 2,
                mx + r // 2,
                my + r // 2,
                outline=self.MARKER_COLOR,
                width=2,
            )

        if canvas is self.left_canvas:
            self._left_photo = photo
        elif canvas is self.right_canvas:
            self._right_photo = photo

    def _get_output_frames(
        self,
    ) -> tuple[Image.Image, Image.Image] | None:
        if self.left_aligned is None or self.right_aligned is None:
            return None

        if self.crop_var.get():
            return crop_border_glitch(
                self.left_aligned, self.right_aligned, self.half_dx, self.half_dy
            )
        return self.left_aligned, self.right_aligned

    def _toggle_preview(self):
        if self.preview_running:
            self._stop_preview()
        else:
            self._start_preview()

    def _start_preview(self):
        frames = self._get_output_frames()
        if frames is None:
            messagebox.showinfo(
                "Info", "Set a focal point first by clicking on the left image"
            )
            return

        self._preview_left, self._preview_right = frames
        self.preview_running = True
        self.preview_frame = 0
        self._preview_step()

    def _stop_preview(self):
        self.preview_running = False
        if self._preview_job is not None:
            self.root.after_cancel(self._preview_job)
            self._preview_job = None

    def _preview_step(self):
        if not self.preview_running:
            return

        frame = self._preview_left if self.preview_frame == 0 else self._preview_right
        self.preview_frame = 1 - self.preview_frame

        scale, ox, oy, dw, dh = self._get_display_params(self.preview_canvas, frame)
        if dw <= 0 or dh <= 0:
            self._preview_job = self.root.after(
                self.speed_var.get(), self._preview_step
            )
            return

        display_img = frame.resize((dw, dh), Resampling.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(display_img)

        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(ox, oy, anchor=tk.NW, image=self._preview_photo)

        delay = self.speed_var.get()
        self._preview_job = self.root.after(delay, self._preview_step)

    def _save_gif(self):
        frames = self._get_output_frames()
        if frames is None:
            messagebox.showinfo("Info", "Set a focal point first")
            return

        left_out, right_out = frames

        if self.file_index < 0 or self.file_index >= len(self.file_list):
            return

        default_name = (
            os.path.splitext(os.path.basename(self.file_list[self.file_index]))[0]
            + ".gif"
        )
        path = filedialog.asksaveasfilename(
            title="Save GIF",
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif")],
            initialfile=default_name,
        )
        if not path:
            return

        try:
            make_wiggle_gif(left_out, right_out, path, self.speed_var.get())
            self.status_var.set(f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save GIF:\n{e}")

    def _save_mp4(self):
        frames = self._get_output_frames()
        if frames is None:
            messagebox.showinfo("Info", "Set a focal point first")
            return

        left_out, right_out = frames

        if self.file_index < 0 or self.file_index >= len(self.file_list):
            return

        default_name = (
            os.path.splitext(os.path.basename(self.file_list[self.file_index]))[0]
            + ".mp4"
        )
        path = filedialog.asksaveasfilename(
            title="Save MP4",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")],
            initialfile=default_name,
        )
        if not path:
            return

        try:
            make_wiggle_mp4(
                left_out,
                right_out,
                path,
                self.speed_var.get(),
                5,
            )
            self.status_var.set(f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save MP4:\n{e}")

    def _on_close(self):
        self._stop_preview()
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive GUI to select focal points for MPO wiggle GIF/MP4 conversion.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to an .MPO file or directory containing .MPO files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    FocalPointGUI(root, initial_path=args.path)
    root.mainloop()


if __name__ == "__main__":
    main()
