#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert repaired white-background brow patches back to transparent PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_OUTPUT_DIR = Path("/media/jinsha/娱乐1/眉毛/透明贴片输出")


def smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    denom = max(float(edge1) - float(edge0), 1e-6)
    t = np.clip((value.astype(np.float32) - float(edge0)) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def crop_by_alpha(rgba: np.ndarray, pad: int = 8) -> np.ndarray:
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if ys.size == 0:
        return rgba
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(rgba.shape[0], int(ys.max()) + pad + 1)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(rgba.shape[1], int(xs.max()) + pad + 1)
    return rgba[y1:y2, x1:x2]


def white_bg_to_alpha(
    image: Image.Image,
    threshold: float = 6.0,
    transition: float = 185.0,
    density: float = 1.15,
    crop: bool = True,
    pad: int = 8,
) -> Image.Image:
    """Remove a white matte while keeping hand-repaired brow strokes.

    The alpha is inferred from distance to white, then RGB is un-matted from a
    white background. This is the same practical idea behind "remove white
    matte": pure white becomes transparent, anti-aliased brow pixels keep soft
    transparency, and the visible stroke color is no longer diluted by white.
    """
    rgba = np.array(image.convert("RGBA")).astype(np.float32)
    rgb = rgba[:, :, :3]
    existing_alpha = rgba[:, :, 3] / 255.0

    distance_from_white = 255.0 - np.min(rgb, axis=2)
    alpha = smoothstep(threshold, threshold + transition, distance_from_white)
    alpha = np.clip(alpha * float(density), 0.0, 1.0)
    alpha *= existing_alpha

    # Drop tiny scanner/compression noise in the white background.
    alpha[distance_from_white <= threshold] = 0.0
    alpha[alpha < 0.006] = 0.0

    safe_alpha = np.maximum(alpha, 1.0 / 255.0)
    unmatted_rgb = 255.0 + (rgb - 255.0) / safe_alpha[:, :, None]
    unmatted_rgb = np.clip(unmatted_rgb, 0, 255)

    out = np.zeros_like(rgba, dtype=np.uint8)
    out[:, :, :3] = unmatted_rgb.astype(np.uint8)
    out[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    out[:, :, :3][out[:, :, 3] == 0] = 0

    if crop:
        out = crop_by_alpha(out, pad=pad)
    return Image.fromarray(out, "RGBA")


def output_path_for(source: Path, output_dir: Path) -> Path:
    stem = source.stem.strip() or "brow_patch"
    return output_dir / f"{stem}_transparent.png"


def convert_file(
    source: Path,
    output_dir: Path,
    threshold: float,
    transition: float,
    density: float,
    crop: bool,
    pad: int,
) -> Path:
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"文件不存在: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(source)
    result = white_bg_to_alpha(image, threshold, transition, density, crop, pad)
    out_path = output_path_for(source, output_dir)
    result.save(out_path)
    return out_path


def iter_images(path: Path):
    if path.is_file():
        yield path
        return
    for item in sorted(path.iterdir()):
        if item.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
            yield item


def run_cli(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    sources: list[Path] = []
    for value in args.inputs:
        path = Path(value).expanduser().resolve()
        sources.extend(iter_images(path))
    if not sources:
        print("没有找到可转换的图片。", file=sys.stderr)
        return 2

    for source in sources:
        out = convert_file(
            source,
            output_dir,
            args.threshold,
            args.transition,
            args.density,
            not args.no_crop,
            args.pad,
        )
        print(out)
    return 0


def run_gui() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("白底眉毛转透明底")
    root.geometry("560x430")

    selected: list[Path] = []
    output_dir = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
    threshold = tk.DoubleVar(value=6.0)
    transition = tk.DoubleVar(value=185.0)
    density = tk.DoubleVar(value=1.15)
    crop = tk.BooleanVar(value=True)
    status = tk.StringVar(value="先选择你在别的软件里修好的白底 PNG。")

    def choose_files():
        files = filedialog.askopenfilenames(
            title="选择白底眉毛图片",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.webp *.tif *.tiff"),
                ("所有文件", "*.*"),
            ],
        )
        if files:
            selected.clear()
            selected.extend(Path(f) for f in files)
            status.set(f"已选择 {len(selected)} 张图片。")

    def choose_folder():
        folder = filedialog.askdirectory(title="选择一整个图片目录")
        if folder:
            selected.clear()
            selected.extend(iter_images(Path(folder)))
            status.set(f"已从目录找到 {len(selected)} 张图片。")

    def choose_output():
        folder = filedialog.askdirectory(title="选择输出目录")
        if folder:
            output_dir.set(folder)

    def convert_selected():
        if not selected:
            messagebox.showwarning("没有图片", "先选择一张或一个目录。")
            return
        try:
            out_dir = Path(output_dir.get()).expanduser().resolve()
            outputs = []
            for source in selected:
                outputs.append(
                    convert_file(
                        source,
                        out_dir,
                        threshold.get(),
                        transition.get(),
                        density.get(),
                        crop.get(),
                        8,
                    )
                )
            status.set(f"完成 {len(outputs)} 张。输出目录：{out_dir}")
            messagebox.showinfo("转换完成", f"已输出 {len(outputs)} 张透明 PNG。\n{out_dir}")
        except Exception as exc:
            messagebox.showerror("转换失败", str(exc))

    def add_slider(label: str, var: tk.DoubleVar, start: float, end: float, row: int, resolution: float = 1):
        tk.Label(root, text=label, anchor="w").grid(row=row, column=0, sticky="ew", padx=14, pady=(10, 0))
        tk.Scale(root, variable=var, from_=start, to=end, orient="horizontal", resolution=resolution).grid(
            row=row,
            column=1,
            columnspan=2,
            sticky="ew",
            padx=14,
            pady=(10, 0),
        )

    root.columnconfigure(1, weight=1)
    tk.Button(root, text="选择白底图片", command=choose_files).grid(row=0, column=0, sticky="ew", padx=14, pady=14)
    tk.Button(root, text="选择图片目录", command=choose_folder).grid(row=0, column=1, sticky="ew", padx=14, pady=14)
    tk.Button(root, text="输出目录", command=choose_output).grid(row=0, column=2, sticky="ew", padx=14, pady=14)

    tk.Label(root, textvariable=output_dir, anchor="w", wraplength=520).grid(
        row=1,
        column=0,
        columnspan=3,
        sticky="ew",
        padx=14,
        pady=(0, 10),
    )

    add_slider("去白强度：越大越干净，也越容易吃掉浅毛", threshold, 0, 45, 2)
    add_slider("边缘过渡：越大越柔，毛尖越不硬", transition, 40, 240, 3)
    add_slider("线条密度：越大贴上去越深", density, 0.5, 2.2, 4, 0.05)

    tk.Checkbutton(root, text="自动裁掉四周空白", variable=crop).grid(
        row=5,
        column=0,
        columnspan=3,
        sticky="w",
        padx=14,
        pady=12,
    )
    tk.Button(root, text="开始转换为透明 PNG", command=convert_selected, height=2).grid(
        row=6,
        column=0,
        columnspan=3,
        sticky="ew",
        padx=14,
        pady=12,
    )
    tk.Label(root, textvariable=status, anchor="w", justify="left", wraplength=520).grid(
        row=7,
        column=0,
        columnspan=3,
        sticky="ew",
        padx=14,
        pady=12,
    )
    root.mainloop()
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把白底眉毛贴片转成透明 PNG")
    parser.add_argument("inputs", nargs="*", help="图片文件或目录")
    parser.add_argument("--gui", action="store_true", help="打开图形界面")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--threshold", type=float, default=6.0, help="去白强度，越大越干净")
    parser.add_argument("--transition", type=float, default=185.0, help="边缘过渡，越大越柔")
    parser.add_argument("--density", type=float, default=1.15, help="线条密度")
    parser.add_argument("--pad", type=int, default=8, help="裁切留边")
    parser.add_argument("--no-crop", action="store_true", help="不自动裁掉透明空白")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.gui or not args.inputs:
        return run_gui()
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
