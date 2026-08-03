from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fig"
W, H = 1800, 1125

NAVY = "#081633"
NAVY_2 = "#10264d"
BLUE = "#1769aa"
CYAN = "#15aebe"
MAGENTA = "#d34f7d"
GREEN = "#2d9b78"
AMBER = "#e3a243"
RED = "#d95562"
VIOLET = "#7358a6"
INK = "#17233b"
MUTED = "#5b6b82"
LINE = "#cbd6e5"
PALE = "#f4f7fb"
WHITE = "#ffffff"

FONT_REGULAR = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD = Path("C:/Windows/Fonts/msyhbd.ttc")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def canvas(title: str, subtitle: str, index: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, W, 132), fill=NAVY)
    draw.rectangle((0, 126, W * 0.72, 132), fill=CYAN)
    draw.rectangle((W * 0.72, 126, W, 132), fill=MAGENTA)
    draw.text((64, 28), title, font=font(48, True), fill=WHITE)
    draw.text((66, 88), subtitle, font=font(22), fill="#c7d8ee")
    draw.text((W - 118, 34), index, font=font(42, True), fill="#75d2df")
    draw.text((64, H - 42), "SceneRepair_v3  |  Furniture-stage relation-aware scene repair", font=font(18), fill=MUTED)
    draw.line((64, H - 58, W - 64, H - 58), fill=LINE, width=2)
    return image, draw


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None, width: int = 2, radius: int = 14) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for char in paragraph:
            candidate = current + char
            if current and draw.textlength(candidate, font=fnt) > max_width:
                lines.append(current)
                current = char
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


def text_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: str = "",
    *,
    fill: str = PALE,
    outline: str = LINE,
    accent: str = BLUE,
    title_size: int = 28,
    body_size: int = 21,
    center: bool = False,
    dashed: bool = False,
) -> None:
    x1, y1, x2, y2 = box
    rounded(draw, box, fill, None if dashed else outline, 2)
    if dashed:
        dashed_rect(draw, box, outline, 3)
    draw.rectangle((x1, y1, x1 + 8, y2), fill=accent)
    tf = font(title_size, True)
    bf = font(body_size)
    title_y = y1 + 20
    if center:
        tw = draw.textlength(title, font=tf)
        draw.text(((x1 + x2 - tw) / 2, title_y), title, font=tf, fill=INK)
    else:
        draw.text((x1 + 24, title_y), title, font=tf, fill=INK)
    if body:
        lines = wrap_lines(draw, body, bf, x2 - x1 - 48)
        line_h = body_size + 10
        y = title_y + title_size + 17
        for line in lines:
            if center:
                lw = draw.textlength(line, font=bf)
                draw.text(((x1 + x2 - lw) / 2, y), line, font=bf, fill=MUTED)
            else:
                draw.text((x1 + 24, y), line, font=bf, fill=MUTED)
            y += line_h


def dashed_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: str, width: int = 3, dash: int = 14) -> None:
    x1, y1, x2, y2 = box
    for x in range(x1, x2, dash * 2):
        draw.line((x, y1, min(x + dash, x2), y1), fill=color, width=width)
        draw.line((x, y2, min(x + dash, x2), y2), fill=color, width=width)
    for y in range(y1, y2, dash * 2):
        draw.line((x1, y, x1, min(y + dash, y2)), fill=color, width=width)
        draw.line((x2, y, x2, min(y + dash, y2)), fill=color, width=width)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = BLUE, width: int = 5, dashed: bool = False) -> None:
    x1, y1 = start
    x2, y2 = end
    if dashed:
        segments = 14
        for i in range(0, segments, 2):
            a = i / segments
            b = min((i + 1) / segments, 1.0)
            draw.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), fill=color, width=width)
    else:
        draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 16
    p1 = (x2 - size * math.cos(angle - 0.55), y2 - size * math.sin(angle - 0.55))
    p2 = (x2 - size * math.cos(angle + 0.55), y2 - size * math.sin(angle + 0.55))
    draw.polygon(((x2, y2), p1, p2), fill=color)


def pill(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str, text_fill: str = WHITE, size: int = 20) -> tuple[int, int, int, int]:
    x, y = xy
    fnt = font(size, True)
    tw = int(draw.textlength(text, font=fnt))
    box = (x, y, x + tw + 30, y + size + 18)
    rounded(draw, box, fill, radius=16)
    draw.text((x + 15, y + 7), text, font=fnt, fill=text_fill)
    return box


def save(image: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / name, dpi=(220, 220), optimize=True)


def _iso_positions(
    scene: dict[str, object],
    box: tuple[int, int, int, int],
    *,
    scale_factor: float = 1.0,
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]], tuple[int, int], callable]:
    """Return a stable room-local isometric projection for the 3D figure panels."""
    x1, y1, x2, y2 = box
    length, room_width = float(scene["length"]), float(scene["width"])
    scale = min((x2 - x1) / (length * 1.55), (y2 - y1) / (room_width * 1.35)) * scale_factor
    sx, sy, sz = scale, scale * 0.72, scale * 1.35
    ix, iy = scale * 0.48, scale * 0.52
    cx, base = (x1 + x2) // 2, y2 - int(scale * 0.34)

    def project(x: float, y: float, z: float) -> tuple[int, int]:
        return (int(cx + x * sx + y * sy), int(base - x * ix + y * iy - z * sz))

    furniture: dict[str, tuple[int, int]] = {}
    for obj in scene["furniture"]:
        furniture[str(obj["object_id"])] = project(float(obj["x"]), float(obj["y"]), float(obj["bbox"]["height"]) * 0.52)
    openings: dict[str, tuple[int, int]] = {}
    for opening in scene["openings"]:
        cx_m, cy_m, cz_m = opening["clearance_center_xyz_m"]
        openings[str(opening["opening_id"])] = project(float(cx_m), float(cy_m), float(cz_m))
    return furniture, openings, project(0.0, 0.0, 1.55), project


def draw_3d_scene(
    draw: ImageDraw.ImageDraw,
    scene: dict[str, object],
    box: tuple[int, int, int, int],
    *,
    show_labels: bool = False,
    scale_factor: float = 1.0,
    show_contract_label: bool = True,
    highlight_ids: set[str] | None = None,
    highlight_color: str = RED,
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]], tuple[int, int]]:
    """Draw a lightweight 3D bbox scene using the scene's real dimensions and heights."""
    x1, y1, x2, y2 = box
    length, room_width = float(scene["length"]), float(scene["width"])
    furniture_pos, opening_pos, room_pos, project = _iso_positions(scene, box, scale_factor=scale_factor)
    scale = min((x2 - x1) / (length * 1.55), (y2 - y1) / (room_width * 1.35)) * scale_factor
    floor = [
        project(-length / 2, -room_width / 2, 0),
        project(length / 2, -room_width / 2, 0),
        project(length / 2, room_width / 2, 0),
        project(-length / 2, room_width / 2, 0),
    ]
    floor_shadow = [(px + 3, py + 8) for px, py in floor]
    draw.polygon(floor_shadow, fill="#8ca0b526")
    draw.polygon(floor, fill="#f5f8fb", outline="#506987")
    # Sparse floor grid improves depth perception at poster viewing distance.
    for fraction in (0.25, 0.5, 0.75):
        x_m = -length / 2 + length * fraction
        draw.line((project(x_m, -room_width / 2, 0), project(x_m, room_width / 2, 0)), fill="#cbd8e550", width=1)
        y_m = -room_width / 2 + room_width * fraction
        draw.line((project(-length / 2, y_m, 0), project(length / 2, y_m, 0)), fill="#cbd8e550", width=1)
    draw.line(floor + [floor[0]], fill="#2c4668", width=max(2, int(scale * 0.06)))
    # Back and side walls establish the z-up coordinate frame.
    for a, b in ((floor[2], floor[3]), (floor[1], floor[2])):
        top_a = (a[0], a[1] - int(scale * 1.8))
        top_b = (b[0], b[1] - int(scale * 1.8))
        draw.polygon((a, b, top_b, top_a), fill="#dfe9f34a")
        draw.line((a, top_a), fill="#9fb3c8", width=2)
        draw.line((b, top_b), fill="#9fb3c8", width=2)
        draw.line((top_a, top_b), fill="#9fb3c8", width=2)

    # Door/window clearance prisms include both the wall plane and the floor exclusion zone.
    for opening in scene["openings"]:
        cx_m, cy_m, _ = opening["clearance_center_xyz_m"]
        sx_m, sy_m, sz_m = opening["clearance_size_xyz_m"]
        wall = str(opening.get("wall", ""))
        color = RED if opening["kind"] == "door" else CYAN
        ground = [
            project(cx_m - sx_m / 2, cy_m - sy_m / 2, 0.012),
            project(cx_m + sx_m / 2, cy_m - sy_m / 2, 0.012),
            project(cx_m + sx_m / 2, cy_m + sy_m / 2, 0.012),
            project(cx_m - sx_m / 2, cy_m + sy_m / 2, 0.012),
        ]
        draw.polygon(ground, fill=color + "34", outline=color)
        for fraction in (0.25, 0.5, 0.75):
            left = (
                ground[0][0] + (ground[3][0] - ground[0][0]) * fraction,
                ground[0][1] + (ground[3][1] - ground[0][1]) * fraction,
            )
            right = (
                ground[1][0] + (ground[2][0] - ground[1][0]) * fraction,
                ground[1][1] + (ground[2][1] - ground[1][1]) * fraction,
            )
            draw.line((left, right), fill=color + "72", width=1)
        if wall in {"east", "west"}:
            x_wall = length / 2 if wall == "east" else -length / 2
            corners = [project(x_wall, cy_m - sy_m / 2, 0), project(x_wall, cy_m + sy_m / 2, 0), project(x_wall, cy_m + sy_m / 2, sz_m), project(x_wall, cy_m - sy_m / 2, sz_m)]
        else:
            y_wall = room_width / 2 if wall == "north" else -room_width / 2
            corners = [project(cx_m - sx_m / 2, y_wall, 0), project(cx_m + sx_m / 2, y_wall, 0), project(cx_m + sx_m / 2, y_wall, sz_m), project(cx_m - sx_m / 2, y_wall, sz_m)]
        draw.polygon(corners, fill=color + "48", outline=color, width=2)
        draw.line((corners[2], corners[3]), fill=WHITE + "b8", width=2)

    family_colors = {"seating": GREEN, "support_surface": AMBER, "storage": VIOLET, "lighting": "#c6aa28"}
    for obj in sorted(scene["furniture"], key=lambda value: float(value["y"])):
        base_xy = _object_corners(obj)
        height = float(obj["bbox"]["height"])
        bottom = [project(px, py, 0) for px, py in base_xy]
        top = [project(px, py, height) for px, py in base_xy]
        color = family_colors.get(str(obj.get("family")), MAGENTA)
        shadow = [(px + 4, py + 7) for px, py in bottom]
        draw.polygon(shadow, fill="#18273d24")
        # Six closed faces; rear and bottom faces stay translucent under the isometric camera.
        faces = [
            (bottom, color + "28"),
            ([bottom[2], bottom[3], top[3], top[2]], color + "48"),
            ([bottom[3], bottom[0], top[0], top[3]], color + "5c"),
            ([bottom[0], bottom[1], top[1], top[0]], color + "a6"),
            ([bottom[1], bottom[2], top[2], top[1]], color + "78"),
            (top, color + "dc"),
        ]
        for face, face_color in faces:
            draw.polygon(face, fill=face_color, outline="#ffffff76")
        draw.polygon(top, fill=color + "dc", outline=WHITE)
        for index in range(4):
            draw.line((bottom[index], top[index]), fill=color + "9a", width=1)
        draw.line(bottom + [bottom[0]], fill=color + "68", width=1)
        draw.line((top[0], top[1], top[2]), fill="#ffffffc8", width=2)
        if highlight_ids and str(obj["object_id"]) in highlight_ids:
            draw.line(bottom + [bottom[0]], fill=highlight_color, width=3)
            draw.line(top + [top[0]], fill=highlight_color, width=3)
            for index in range(4):
                draw.line((bottom[index], top[index]), fill=highlight_color, width=2)
        if show_labels:
            center = project(float(obj["x"]), float(obj["y"]), height + 0.05)
            label = str(obj["category"]).replace("_", " ")
            lf = font(12, True)
            draw.text((center[0] - draw.textlength(label, font=lf) / 2, center[1] - 7), label, font=lf, fill=INK)
    if show_contract_label:
        draw.text((x1 + 15, y1 + 12), "z-up · bbox height from scene contract", font=font(11), fill=MUTED)
    return furniture_pos, opening_pos, room_pos


FLOOR_TEXTURE = ROOT.parent / "07-38-35/07-38-35/scene_000/materials/Planks010/Planks010_2K-JPG_Color.jpg"
WALL_TEXTURE = ROOT.parent / "07-38-35/07-38-35/scene_000/materials/PaintedPlaster015/PaintedPlaster015_2K-JPG_Color.jpg"
HSSD_HAB_ROOT = ROOT.parent / "hssd-hab"
_HSSD_MESH_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _mix(color: str, target: str, amount: float) -> str:
    a = _rgb(color)
    b = _rgb(target)
    mixed = tuple(int(a[i] * (1 - amount) + b[i] * amount) for i in range(3))
    return "#%02x%02x%02x" % mixed


def _tile_texture(path: Path, size: tuple[int, int], tint: str, opacity: int) -> Image.Image | None:
    width, height = size
    if width <= 0 or height <= 0 or not path.exists():
        return None
    texture = Image.open(path).convert("RGB")
    texture.thumbnail((360, 360))
    tile = Image.new("RGB", (width, height), _rgb(tint))
    for px in range(0, width, texture.width):
        for py in range(0, height, texture.height):
            tile.paste(texture, (px, py))
    tile = Image.blend(tile, Image.new("RGB", (width, height), _rgb(tint)), 0.18)
    tile.putalpha(opacity)
    return tile


def _paste_texture_polygon(
    image: Image.Image,
    points: list[tuple[int, int]],
    texture_path: Path,
    *,
    tint: str,
    opacity: int = 225,
) -> None:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    patch = _tile_texture(texture_path, (right - left + 1, bottom - top + 1), tint, opacity)
    if patch is None:
        return
    mask = Image.new("L", patch.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.polygon([(x - left, y - top) for x, y in points], fill=255)
    image.paste(patch, (left, top), mask)


def _local_project(obj: dict[str, object], lx: float, ly: float, z: float, project: callable) -> tuple[int, int]:
    yaw = float(obj["yaw_rad"])
    c, s = math.cos(yaw), math.sin(yaw)
    x = float(obj["x"]) + c * lx - s * ly
    y = float(obj["y"]) + s * lx + c * ly
    return project(x, y, z)


def _fit_scene_projector(
    scene: dict[str, object],
    box: tuple[int, int, int, int],
    *,
    wall_height: float = 2.25,
    padding: int = 24,
    scale_factor: float = 1.0,
) -> callable:
    length, room_width = float(scene["length"]), float(scene["width"])

    def raw_project(x: float, y: float, z: float) -> tuple[float, float]:
        return x + y * 0.72, -x * 0.48 + y * 0.52 - z * 1.35

    samples: list[tuple[float, float, float]] = []
    for x in (-length / 2, length / 2):
        for y in (-room_width / 2, room_width / 2):
            samples.append((x, y, 0))
    for x, y in ((-length / 2, room_width / 2), (length / 2, room_width / 2), (length / 2, -room_width / 2)):
        samples.append((x, y, wall_height))

    for opening in scene["openings"]:
        cx_m, cy_m, _ = opening["clearance_center_xyz_m"]
        sx_m, sy_m, sz_m = opening["clearance_size_xyz_m"]
        for px in (cx_m - sx_m / 2, cx_m + sx_m / 2):
            for py in (cy_m - sy_m / 2, cy_m + sy_m / 2):
                samples.append((px, py, 0))
                samples.append((px, py, sz_m))

    for obj in scene["furniture"]:
        height = float(obj["bbox"]["height"])
        for px, py in _object_corners(obj):
            samples.append((px, py, 0))
            samples.append((px, py, height))

    projected = [raw_project(x, y, z) for x, y, z in samples]
    min_u, max_u = min(p[0] for p in projected), max(p[0] for p in projected)
    min_v, max_v = min(p[1] for p in projected), max(p[1] for p in projected)
    x1, y1, x2, y2 = box
    scale = min((x2 - x1 - padding * 2) / (max_u - min_u), (y2 - y1 - padding * 2) / (max_v - min_v)) * scale_factor
    offset_x = x1 + padding - min_u * scale + ((x2 - x1 - padding * 2) - (max_u - min_u) * scale) / 2
    offset_y = y1 + padding - min_v * scale + ((y2 - y1 - padding * 2) - (max_v - min_v) * scale) / 2

    def project(x: float, y: float, z: float) -> tuple[int, int]:
        u, v = raw_project(x, y, z)
        return int(offset_x + u * scale), int(offset_y + v * scale)

    return project


def _draw_prism(
    draw: ImageDraw.ImageDraw,
    obj: dict[str, object],
    project: callable,
    lx1: float,
    ly1: float,
    lx2: float,
    ly2: float,
    z1: float,
    z2: float,
    color: str,
    *,
    outline: str = "#ffffff",
    alpha: str = "ee",
    shadow: bool = False,
) -> None:
    corners = [(lx1, ly1), (lx2, ly1), (lx2, ly2), (lx1, ly2)]
    bottom = [_local_project(obj, lx, ly, z1, project) for lx, ly in corners]
    top = [_local_project(obj, lx, ly, z2, project) for lx, ly in corners]
    if shadow and z1 <= 0.02:
        draw.polygon([(x + 5, y + 8) for x, y in bottom], fill="#14203328")
    faces = [
        ([bottom[2], bottom[3], top[3], top[2]], _mix(color, "#000000", 0.16) + "cc"),
        ([bottom[3], bottom[0], top[0], top[3]], _mix(color, "#000000", 0.10) + "cc"),
        ([bottom[0], bottom[1], top[1], top[0]], _mix(color, "#ffffff", 0.08) + "cc"),
        ([bottom[1], bottom[2], top[2], top[1]], _mix(color, "#000000", 0.05) + "cc"),
        (top, color + alpha),
    ]
    for face, fill in faces:
        draw.polygon(face, fill=fill, outline=outline + "55")
    draw.line(top + [top[0]], fill=outline + "bb", width=1)


def _draw_surface_detail(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: str, count: int = 3) -> None:
    for index in range(1, count + 1):
        t = index / (count + 1)
        left = (
            int(points[0][0] * (1 - t) + points[3][0] * t),
            int(points[0][1] * (1 - t) + points[3][1] * t),
        )
        right = (
            int(points[1][0] * (1 - t) + points[2][0] * t),
            int(points[1][1] * (1 - t) + points[2][1] * t),
        )
        draw.line((left, right), fill=color, width=1)


def _load_hssd_mesh(hssd_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not hssd_id:
        return None
    if hssd_id in _HSSD_MESH_CACHE:
        return _HSSD_MESH_CACHE[hssd_id]
    path = HSSD_HAB_ROOT / "objects" / hssd_id[0] / f"{hssd_id}.glb"
    if not path.exists():
        return None
    scene = trimesh.load(path, force="scene")
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        return None
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    # Large HSSD lamps/chairs have far more triangles than needed at poster scale.
    max_faces = 5200
    if len(faces) > max_faces:
        keep = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int32)
        faces = faces[keep]
    result = (vertices, faces, bounds_min, bounds_max)
    _HSSD_MESH_CACHE[hssd_id] = result
    return result


def _category_mesh_color(category: str) -> str:
    if category in {"chair", "armchair", "sofa", "bench", "ottoman"}:
        return "#4aa37f"
    if category in {"desk", "table", "coffee_table", "side_table", "console_table"}:
        return "#c79a5a"
    if category in {"cabinet", "wardrobe", "bookcase", "tv_bench"}:
        return "#8b68aa"
    if "lamp" in category:
        return "#d9bd58"
    if category in {"bed", "nightstand"}:
        return "#5e8ac5"
    return "#8ca0b5"


def _draw_hssd_mesh_furniture(
    draw: ImageDraw.ImageDraw,
    obj: dict[str, object],
    project: callable,
    *,
    highlight: str | None = None,
) -> bool:
    hssd_id = str(obj.get("hssd_id") or "")
    loaded = _load_hssd_mesh(hssd_id)
    if loaded is None:
        return False
    vertices, faces, bounds_min, bounds_max = loaded
    bbox = obj["bbox"]
    width, depth, height = float(bbox["width"]), float(bbox["depth"]), float(bbox["height"])
    extents = np.maximum(bounds_max - bounds_min, 1e-6)
    center_x = (bounds_min[0] + bounds_max[0]) / 2
    center_z = (bounds_min[2] + bounds_max[2]) / 2
    # HSSD GLBs are Y-up. SceneRepair panels are room-local Z-up:
    #   local scene x = asset x, local scene y = asset z, local scene z = asset y.
    local_x = (vertices[:, 0] - center_x) * (width / extents[0])
    local_y = (vertices[:, 2] - center_z) * (depth / extents[2])
    local_z = (vertices[:, 1] - bounds_min[1]) * (height / extents[1])
    yaw = float(obj["yaw_rad"])
    c, s = math.cos(yaw), math.sin(yaw)
    world_x = float(obj["x"]) + c * local_x - s * local_y
    world_y = float(obj["y"]) + s * local_x + c * local_y
    world_z = local_z
    screen = np.array([project(float(x), float(y), float(z)) for x, y, z in zip(world_x, world_y, world_z)], dtype=np.float64)
    world = np.stack([world_x, world_y, world_z], axis=1)

    base_color = _category_mesh_color(str(obj["category"]))
    base_rgb = np.array(_rgb(base_color), dtype=np.float64)
    light = np.array([-0.35, -0.45, 0.82], dtype=np.float64)
    light = light / np.linalg.norm(light)

    base = [_local_project(obj, lx, ly, 0.025, project) for lx, ly in ((-width / 2, -depth / 2), (width / 2, -depth / 2), (width / 2, depth / 2), (-width / 2, depth / 2))]
    draw.polygon([(x + 5, y + 8) for x, y in base], fill="#10213a24")

    face_rows: list[tuple[float, list[tuple[int, int]], tuple[int, int, int]]] = []
    for face in faces:
        tri_world = world[face]
        tri_screen = screen[face]
        area = abs(np.cross(tri_screen[1] - tri_screen[0], tri_screen[2] - tri_screen[0]))
        if area < 0.35:
            continue
        normal = np.cross(tri_world[1] - tri_world[0], tri_world[2] - tri_world[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        shade = 0.58 + 0.34 * max(0.0, float(np.dot(normal, light))) + 0.10 * max(0.0, float(normal[2]))
        rgb = tuple(int(max(0, min(255, channel * shade))) for channel in base_rgb)
        # Draw upper/far triangles first. This painter order is stable enough for isometric poster views.
        order = float(tri_screen[:, 1].mean() - tri_world[:, 2].mean() * 14)
        points = [(int(x), int(y)) for x, y in tri_screen]
        face_rows.append((order, points, rgb))
    for _, points, rgb in sorted(face_rows, key=lambda item: item[0]):
        draw.polygon(points, fill=rgb + (224,))

    if highlight:
        draw.line(base + [base[0]], fill=highlight + "e8", width=3)
    return True


def _draw_realistic_furniture(
    draw: ImageDraw.ImageDraw,
    obj: dict[str, object],
    project: callable,
    *,
    highlight: str | None = None,
) -> None:
    bbox = obj["bbox"]
    w, d, h = float(bbox["width"]), float(bbox["depth"]), float(bbox["height"])
    category = str(obj["category"])
    wood = "#c79a5a"
    dark_wood = "#8d6841"
    fabric = "#4f9f83"
    leather = "#7d609f"
    metal = "#66788d"
    brass = "#d8bc54"

    if category == "desk":
        _draw_prism(draw, obj, project, -w / 2, -d / 2, w / 2, d / 2, h * 0.82, h, wood, shadow=True)
        leg_w, leg_d = w * 0.10, d * 0.12
        for sx in (-1, 1):
            for sy in (-1, 1):
                _draw_prism(
                    draw,
                    obj,
                    project,
                    sx * w * 0.39 - leg_w / 2,
                    sy * d * 0.36 - leg_d / 2,
                    sx * w * 0.39 + leg_w / 2,
                    sy * d * 0.36 + leg_d / 2,
                    0,
                    h * 0.82,
                    dark_wood,
                    outline="#f7ead5",
                )
        _draw_prism(draw, obj, project, -w * 0.48, d * 0.18, -w * 0.22, d * 0.43, 0, h * 0.78, "#b9884d")
        monitor = [
            _local_project(obj, -w * 0.05, d * 0.10, h + 0.02, project),
            _local_project(obj, w * 0.24, d * 0.10, h + 0.02, project),
            _local_project(obj, w * 0.24, d * 0.10, h + 0.28, project),
            _local_project(obj, -w * 0.05, d * 0.10, h + 0.28, project),
        ]
        draw.polygon(monitor, fill="#1d2b3fcc", outline="#b7c7d7")
        keyboard = [_local_project(obj, lx, -d * 0.14, h + 0.015, project) for lx in (-w * 0.20, w * 0.22)]
        draw.line((keyboard[0], keyboard[1]), fill="#2d3546cc", width=4)
    elif category == "chair":
        _draw_prism(draw, obj, project, -w * 0.38, -d * 0.28, w * 0.38, d * 0.28, h * 0.38, h * 0.52, leather, shadow=True)
        _draw_prism(draw, obj, project, -w * 0.40, d * 0.24, w * 0.40, d * 0.38, h * 0.48, h, _mix(leather, "#ffffff", 0.08))
        _draw_prism(draw, obj, project, -w * 0.48, -d * 0.24, -w * 0.36, d * 0.18, h * 0.36, h * 0.62, _mix(leather, "#ffffff", 0.05))
        _draw_prism(draw, obj, project, w * 0.36, -d * 0.24, w * 0.48, d * 0.18, h * 0.36, h * 0.62, _mix(leather, "#ffffff", 0.05))
        base = _local_project(obj, 0, 0, h * 0.18, project)
        seat = _local_project(obj, 0, 0, h * 0.42, project)
        draw.line((base, seat), fill=metal, width=4)
        for lx, ly in ((0, -d * 0.34), (-w * 0.30, d * 0.22), (w * 0.30, d * 0.22)):
            wheel = _local_project(obj, lx, ly, 0.04, project)
            draw.ellipse((wheel[0] - 6, wheel[1] - 3, wheel[0] + 6, wheel[1] + 5), fill="#2c3445cc")
    elif category == "cabinet":
        _draw_prism(draw, obj, project, -w / 2, -d / 2, w / 2, d / 2, 0, h, "#b28b59", shadow=True)
        front = [_local_project(obj, lx, -d / 2, z, project) for lx, z in ((-w / 2, h * 0.18), (w / 2, h * 0.18), (w / 2, h * 0.82), (-w / 2, h * 0.82))]
        draw.polygon(front, fill="#d5b27ccc", outline="#87633eaa")
        _draw_surface_detail(draw, front, "#8b6a45aa", 2)
        divider_top = _local_project(obj, 0, -d / 2 - 0.01, h * 0.82, project)
        divider_bottom = _local_project(obj, 0, -d / 2 - 0.01, h * 0.18, project)
        draw.line((divider_top, divider_bottom), fill="#7b5a39aa", width=1)
        for lx in (-w * 0.18, w * 0.18):
            knob = _local_project(obj, lx, -d / 2 - 0.01, h * 0.52, project)
            draw.ellipse((knob[0] - 3, knob[1] - 3, knob[0] + 3, knob[1] + 3), fill="#4a3826")
    elif category == "armchair":
        _draw_prism(draw, obj, project, -w * 0.36, -d * 0.28, w * 0.36, d * 0.25, h * 0.24, h * 0.55, fabric, shadow=True)
        _draw_prism(draw, obj, project, -w * 0.45, d * 0.18, w * 0.45, d * 0.42, h * 0.35, h, _mix(fabric, "#000000", 0.05))
        _draw_prism(draw, obj, project, -w * 0.50, -d * 0.28, -w * 0.34, d * 0.24, h * 0.22, h * 0.68, _mix(fabric, "#ffffff", 0.05))
        _draw_prism(draw, obj, project, w * 0.34, -d * 0.28, w * 0.50, d * 0.24, h * 0.22, h * 0.68, _mix(fabric, "#ffffff", 0.05))
        cushion = [_local_project(obj, lx, ly, h * 0.57, project) for lx, ly in ((-w * 0.30, -d * 0.18), (w * 0.30, -d * 0.18), (w * 0.30, d * 0.12), (-w * 0.30, d * 0.12))]
        draw.line(cushion + [cushion[0]], fill="#d7eee3", width=2)
        seam_a = [_local_project(obj, -w * 0.04, ly, h * 0.575, project) for ly in (-d * 0.16, d * 0.10)]
        draw.line((seam_a[0], seam_a[1]), fill="#d7eee3aa", width=1)
    elif category == "floor_lamp":
        base = _local_project(obj, 0, 0, 0.02, project)
        top = _local_project(obj, 0, 0, h * 0.78, project)
        draw.ellipse((base[0] - 18, base[1] - 7, base[0] + 18, base[1] + 8), fill="#816d3fcc", outline="#f6e5a1")
        draw.line((base, top), fill="#574a38", width=4)
        shade = [
            _local_project(obj, -w * 0.62, -d * 0.20, h * 0.70, project),
            _local_project(obj, w * 0.62, -d * 0.20, h * 0.70, project),
            _local_project(obj, w * 0.48, d * 0.16, h, project),
            _local_project(obj, -w * 0.48, d * 0.16, h, project),
        ]
        draw.polygon(shade, fill=brass + "dd", outline="#fff3b0")
        for radius, alpha in ((34, "18"), (24, "24"), (15, "38")):
            draw.ellipse((top[0] - radius, top[1] - radius, top[0] + radius, top[1] + radius), fill="#ffe783" + alpha)
    else:
        _draw_prism(draw, obj, project, -w / 2, -d / 2, w / 2, d / 2, 0, h, "#9aa9b8", shadow=True)

    if highlight:
        base = [_local_project(obj, lx, ly, 0.03, project) for lx, ly in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2))]
        draw.line(base + [base[0]], fill=highlight + "dd", width=3)


def draw_realistic_3d_scene(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    scene: dict[str, object],
    box: tuple[int, int, int, int],
    *,
    title: str,
    accent: str,
    highlight_ids: set[str] | None = None,
    show_motion_to: dict[str, dict[str, object]] | None = None,
) -> None:
    x1, y1, x2, y2 = box
    length, room_width = float(scene["length"]), float(scene["width"])
    wall_h = 2.25
    project = _fit_scene_projector(scene, (x1 + 22, y1 + 82, x2 - 22, y2 - 20), wall_height=wall_h, padding=8, scale_factor=1.05)
    floor = [
        project(-length / 2, -room_width / 2, 0),
        project(length / 2, -room_width / 2, 0),
        project(length / 2, room_width / 2, 0),
        project(-length / 2, room_width / 2, 0),
    ]

    rounded(draw, box, "#fbfcfe", "#d2dce8", 2, radius=20)
    draw.rectangle((x1 + 1, y1 + 1, x2 - 1, y1 + 64), fill="#f8fafc")
    draw.rounded_rectangle((x1 + 26, y1 + 17, x1 + 56, y1 + 47), radius=15, fill=accent)
    draw.text((x1 + 70, y1 + 12), title, font=font(27, True), fill=INK)
    draw.text((x1 + 70, y1 + 45), "HSSD GLB meshes · v10 pose · z-up scene", font=font(12), fill=MUTED)
    draw.line((x1 + 24, y1 + 65, x2 - 24, y1 + 65), fill="#e3eaf2", width=1)

    # Textured room shell.
    shadow = [(px + 7, py + 14) for px, py in floor]
    draw.polygon(shadow, fill="#10213a25")
    back_wall = [
        project(length / 2, room_width / 2, 0),
        project(-length / 2, room_width / 2, 0),
        project(-length / 2, room_width / 2, wall_h),
        project(length / 2, room_width / 2, wall_h),
    ]
    right_wall = [
        project(length / 2, -room_width / 2, 0),
        project(length / 2, room_width / 2, 0),
        project(length / 2, room_width / 2, wall_h),
        project(length / 2, -room_width / 2, wall_h),
    ]
    _paste_texture_polygon(image, back_wall, WALL_TEXTURE, tint="#f2f5f8", opacity=205)
    _paste_texture_polygon(image, right_wall, WALL_TEXTURE, tint="#e9eef4", opacity=185)
    _paste_texture_polygon(image, floor, FLOOR_TEXTURE, tint="#d2b37e", opacity=220)
    draw.polygon(back_wall, outline="#b3c3d5")
    draw.polygon(right_wall, outline="#b3c3d5")
    draw.line(floor + [floor[0]], fill="#2f4868", width=3)

    for fraction in (0.2, 0.4, 0.6, 0.8):
        x_m = -length / 2 + length * fraction
        draw.line((project(x_m, -room_width / 2, 0.015), project(x_m, room_width / 2, 0.015)), fill="#ffffff55", width=1)
        y_m = -room_width / 2 + room_width * fraction
        draw.line((project(-length / 2, y_m, 0.015), project(length / 2, y_m, 0.015)), fill="#6f543222", width=1)

    # Door/window planes plus ground clearance zones.
    for opening in scene["openings"]:
        cx_m, cy_m, _ = opening["clearance_center_xyz_m"]
        sx_m, sy_m, sz_m = opening["clearance_size_xyz_m"]
        wall = str(opening.get("wall", ""))
        color = RED if opening["kind"] == "door" else CYAN
        ground = [
            project(cx_m - sx_m / 2, cy_m - sy_m / 2, 0.035),
            project(cx_m + sx_m / 2, cy_m - sy_m / 2, 0.035),
            project(cx_m + sx_m / 2, cy_m + sy_m / 2, 0.035),
            project(cx_m - sx_m / 2, cy_m + sy_m / 2, 0.035),
        ]
        draw.polygon(ground, fill=color + "28", outline=color + "aa")
        if wall in {"east", "west"}:
            x_wall = length / 2 if wall == "east" else -length / 2
            plane = [
                project(x_wall, cy_m - sy_m / 2, 0.05),
                project(x_wall, cy_m + sy_m / 2, 0.05),
                project(x_wall, cy_m + sy_m / 2, sz_m),
                project(x_wall, cy_m - sy_m / 2, sz_m),
            ]
        else:
            y_wall = room_width / 2 if wall == "north" else -room_width / 2
            plane = [
                project(cx_m - sx_m / 2, y_wall, 0.05),
                project(cx_m + sx_m / 2, y_wall, 0.05),
                project(cx_m + sx_m / 2, y_wall, sz_m),
                project(cx_m - sx_m / 2, y_wall, sz_m),
            ]
        fill = "#a8eff266" if opening["kind"] == "window" else "#f7ced566"
        draw.polygon(plane, fill=fill, outline=color, width=2)
        draw.line((plane[0], plane[2]), fill="#ffffff99", width=1)
        draw.line((plane[1], plane[3]), fill="#ffffff99", width=1)

    for obj in sorted(scene["furniture"], key=lambda value: float(value["y"])):
        object_id = str(obj["object_id"])
        if show_motion_to and object_id in show_motion_to:
            start = _local_project(obj, 0, 0, float(obj["bbox"]["height"]) + 0.05, project)
            target_obj = {**obj, **show_motion_to[object_id]}
            end = _local_project(target_obj, 0, 0, float(obj["bbox"]["height"]) + 0.05, project)
            arrow(draw, start, end, accent, width=2, dashed=True)
        mesh_drawn = _draw_hssd_mesh_furniture(
            draw,
            obj,
            project,
            highlight=accent if highlight_ids and object_id in highlight_ids else None,
        )
        if not mesh_drawn:
            _draw_realistic_furniture(
                draw,
                obj,
                project,
                highlight=accent if highlight_ids and object_id in highlight_ids else None,
            )


def figure_model() -> None:
    scene_path = ROOT / "data/clean_layout_production_v2/accepted/scenes/clean_v2_prod_0017_living_room_0001.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    image, draw = canvas(
        "从真实场景到逐家具修复动作",
        "One accepted scene traced through typed encoding, heterogeneous relations and action heads",
        "01",
    )

    # A faint technical grid gives the panel depth without competing with the data.
    for gx in range(55, W - 50, 42):
        draw.line((gx, 150, gx, 865), fill="#eaf0f6", width=1)
    for gy in range(150, 866, 42):
        draw.line((55, gy, W - 55, gy), fill="#eaf0f6", width=1)

    def stage(x: int, number: str, title: str, color: str) -> None:
        draw.ellipse((x, 157, x + 42, 199), fill=color)
        nf = font(20, True)
        draw.text((x + 21 - draw.textlength(number, font=nf) / 2, 164), number, font=nf, fill=WHITE)
        draw.text((x + 53, 162), title, font=font(25, True), fill=INK)

    stage(55, "1", "场景输入", GREEN)
    stage(430, "2", "节点与边特征", BLUE)
    stage(775, "3", "异构关系图", MAGENTA)
    stage(1280, "4", "关系推理与动作头", CYAN)

    # Stage 1: one integrated specimen card keeps the real scene as the visual focus.
    rounded(draw, (55, 218, 402, 823), "#fbfcfe", "#91abc5", 2)
    draw.rectangle((73, 239, 79, 269), fill=GREEN)
    draw.text((91, 238), "LIVING ROOM", font=font(23, True), fill=INK)
    draw.line((73, 281, 384, 281), fill="#d8e3ed", width=1)

    # A softly framed viewport separates the scene specimen from its metadata.
    rounded(draw, (67, 295, 390, 671), "#f6f9fc", "#d3dfea", 1, radius=10)
    draw.text((82, 310), "METRIC BBOX · Z-UP", font=font(10, True), fill="#72869c")
    draw_3d_scene(
        draw,
        scene,
        (68, 300, 390, 548),
        show_labels=False,
        show_contract_label=False,
    )

    # Three equal summary cells replace the previous stack of detached badges.
    metric_cells = [
        ("ROOM SIZE", "6.30 × 4.90", "m", BLUE),
        ("OPENINGS", "2", "door / window", CYAN),
        ("FURNITURE", "7", "objects", MAGENTA),
    ]
    cell_x = (67, 176, 285)
    for index, (label, value, suffix, color) in enumerate(metric_cells):
        mx = cell_x[index]
        rounded(draw, (mx, 686, mx + 105, 767), color + "0b", color + "55", 1, radius=9)
        draw.rectangle((mx, 686, mx + 105, 691), fill=color)
        draw.text((mx + 10, 700), label, font=font(9, True), fill=MUTED)
        value_font = font(15 if index == 0 else 24, True)
        draw.text((mx + 10, 720 if index == 0 else 713), value, font=value_font, fill=INK)
        if index == 0:
            draw.text((mx + 10, 744), suffix, font=font(10, True), fill=color)
        else:
            draw.text((mx + 42, 737), suffix, font=font(9), fill=MUTED)

    # The floor overlays are part of the input contract, not decorative marks.
    draw.rectangle((76, 788, 87, 798), fill=RED + "38", outline=RED)
    draw.text((94, 785), "door clearance", font=font(9), fill=MUTED)
    draw.rectangle((237, 788, 248, 798), fill=CYAN + "38", outline=CYAN)
    draw.text((255, 785), "window clearance", font=font(9), fill=MUTED)

    # Stage 2: compact node encoders plus the four typed edge feature contracts.
    rounded(draw, (430, 218, 744, 555), WHITE, "#aabbd0", 2)
    draw.text((450, 234), "NODE FEATURES", font=font(17, True), fill=INK)
    pill(draw, (605, 228), "validity-aware", BLUE, size=11)
    node_rows = [
        ("R", "ROOM", "[1 × 2]", "[6.30, 4.90]", BLUE),
        ("O", "OPENING", "[2 × 8]", "[cx/L, cy/W, …, nx, ny]", CYAN),
        ("F", "FURNITURE", "[7 × 7]", "sofa [−.103, .411, 0, 1, …]", MAGENTA),
    ]
    row_y = 274
    for letter, title, shape, sample, color in node_rows:
        rounded(draw, (444, row_y, 730, row_y + 78), color + "0d", color + "66", 1, radius=10)
        draw.ellipse((456, row_y + 17, 496, row_y + 57), fill=color)
        lf = font(17, True)
        draw.text((476 - draw.textlength(letter, font=lf) / 2, row_y + 25), letter, font=lf, fill=WHITE)
        draw.text((509, row_y + 10), title, font=font(17, True), fill=INK)
        draw.text((650, row_y + 11), shape, font=font(13, True), fill=color)
        draw.text((509, row_y + 37), sample, font=font(12), fill=MUTED)
        draw.text((509, row_y + 57), "MLP → 128D", font=font(11, True), fill=color)
        for token in range(4):
            draw.rounded_rectangle((650 + token * 17, row_y + 52, 662 + token * 17, row_y + 64), radius=3, fill=color + "70")
        row_y += 88

    rounded(draw, (430, 575, 744, 823), "#f7f9fc", "#9fb3ca", 2)
    draw.text((450, 592), "EDGE FEATURES", font=font(17, True), fill=INK)
    pill(draw, (607, 586), "4 relations", VIOLET, size=11)
    edge_cells = [
        ("ROOM", "18 edges", "cont 6 + cat 2", BLUE),
        ("SPATIAL", "42 edges", "continuous 8", MAGENTA),
        ("FUNCTIONAL", "10 edges", "cont 4 + cat 2", VIOLET),
        ("OPENING", "14 edges", "continuous 4", CYAN),
    ]
    for index, (title, count, dimensions, color) in enumerate(edge_cells):
        ex = 444 + (index % 2) * 145
        ey = 630 + (index // 2) * 87
        rounded(draw, (ex, ey, ex + 137, ey + 76), WHITE, color + "88", 1, radius=9)
        draw.rectangle((ex, ey, ex + 6, ey + 76), fill=color)
        draw.text((ex + 16, ey + 9), title, font=font(13, True), fill=INK)
        draw.text((ex + 16, ey + 31), count, font=font(12, True), fill=color)
        draw.text((ex + 16, ey + 51), dimensions, font=font(10), fill=MUTED)
    arrow(draw, (405, 450), (423, 450), NAVY_2, width=4)
    arrow(draw, (750, 500), (770, 500), NAVY_2, width=4)

    # Stage 3: preserve the real room topology while making relation density visible.
    graph_box = (775, 218, 1254, 823)
    rounded(draw, graph_box, "#fbfcfe", "#9fb1c8", 2)
    draw.text((800, 238), "LIVING ROOM · HETEROGENEOUS GRAPH", font=font(17, True), fill=MUTED)
    draw.text((800, 270), "10 nodes  ·  84 directed edges", font=font(22, True), fill=INK)
    draw.text((800, 307), "3D room-local z-up", font=font(12, True), fill=CYAN)

    length, room_width = float(scene["length"]), float(scene["width"])
    labels = {
        "sofa_01": "SOFA",
        "coffee_table_01": "CT",
        "armchair_01": "AC",
        "side_table_01": "ST",
        "floor_lamp_01": "LAMP",
        "console_table_01": "CONS",
        "cabinet_01": "CAB",
    }
    furniture_pos, opening_pos, room_pos = draw_3d_scene(
        draw,
        scene,
        (795, 292, 1238, 620),
        show_labels=False,
        scale_factor=0.88,
        show_contract_label=False,
    )

    # Offset graph badges from their 3D object centers so dense lounge objects remain readable.
    badge_offsets = {
        "sofa_01": (8, 17),
        "coffee_table_01": (-13, -3),
        "armchair_01": (27, -22),
        "side_table_01": (-9, 14),
        "floor_lamp_01": (27, 5),
        "console_table_01": (-9, -10),
        "cabinet_01": (-16, 5),
    }
    furniture_anchor = dict(furniture_pos)
    furniture_pos = {
        object_id: (position[0] + badge_offsets[object_id][0], position[1] + badge_offsets[object_id][1])
        for object_id, position in furniture_pos.items()
    }
    for object_id, anchor in furniture_anchor.items():
        draw.line((*anchor, *furniture_pos[object_id]), fill="#71859d88", width=1)
    opening_pos = {
        object_id: (position[0] + (-7 if index == 0 else 8), position[1] - 7)
        for index, (object_id, position) in enumerate(opening_pos.items())
    }
    room_pos = (room_pos[0] - 3, room_pos[1] - 10)

    # Furniture spatial topology is complete: 7 × 6 = 42 directed edges.
    fids = list(furniture_pos)
    for i, left in enumerate(fids):
        for right in fids[i + 1 :]:
            draw.line((*furniture_pos[left], *furniture_pos[right]), fill=MAGENTA + "36", width=2)
    # Every Furniture node connects to every Opening node: 7 × 2 = 14 directed edges.
    for opos in opening_pos.values():
        for fpos in furniture_pos.values():
            draw.line((*opos, *fpos), fill=CYAN + "30", width=2)
    # Room membership is bidirectional for every non-room node: 18 edges.
    for pos in list(furniture_pos.values()) + list(opening_pos.values()):
        draw.line((*room_pos, *pos), fill=BLUE + "48", width=2)
    # Frozen functional relations are emphasized over the dense geometric substrate.
    functional_pairs: set[tuple[str, str]] = set()
    for rel in scene["functional_partners"]:
        pair = (rel["source_id"], rel["target_id"])
        reverse = (rel["target_id"], rel["source_id"])
        if reverse in functional_pairs:
            continue
        functional_pairs.add(pair)
        draw.line((*furniture_pos[pair[0]], *furniture_pos[pair[1]]), fill=VIOLET + "72", width=3)

    # Room uses a double-ring hub; openings are diamonds; furniture keeps family colors.
    rx, ry = room_pos
    draw.ellipse((rx - 28, ry - 28, rx + 28, ry + 28), fill=BLUE + "20", outline=BLUE, width=3)
    draw.ellipse((rx - 19, ry - 19, rx + 19, ry + 19), fill=NAVY_2)
    draw.text((rx - 13, ry - 8), "ROOM", font=font(9, True), fill=WHITE)
    for oid, (ox, oy) in opening_pos.items():
        points = ((ox, oy - 13), (ox + 13, oy), (ox, oy + 13), (ox - 13, oy))
        draw.polygon(points, fill=CYAN, outline=WHITE)
        draw.text((ox - 4, oy - 6), "O", font=font(10, True), fill=WHITE)
    family_colors = {
        "seating": GREEN,
        "support_surface": AMBER,
        "storage": VIOLET,
        "lighting": "#c6aa28",
    }
    for obj in scene["furniture"]:
        fx, fy2 = furniture_pos[obj["object_id"]]
        fill = family_colors.get(obj["family"], MAGENTA)
        draw.ellipse((fx - 15, fy2 - 15, fx + 15, fy2 + 15), fill=fill, outline=WHITE, width=2)
        draw.ellipse((fx - 15, fy2 - 15, fx + 15, fy2 + 15), outline=MAGENTA, width=2)
        short = labels[obj["object_id"]]
        sf = font(8, True)
        draw.text((fx - draw.textlength(short, font=sf) / 2, fy2 - 5), short, font=sf, fill=WHITE)

    relation_rows = [
        ("ROOM", "18", BLUE),
        ("SPATIAL", "42", MAGENTA),
        ("FUNCTIONAL", "10", VIOLET),
        ("OPENING", "14", CYAN),
    ]
    for index, (label, count, color) in enumerate(relation_rows):
        lx = 793 + (index % 2) * 224
        ly = 738 + (index // 2) * 39
        draw.line((lx, ly + 13, lx + 26, ly + 13), fill=color, width=6)
        draw.text((lx + 35, ly), label, font=font(14, True), fill=INK)
        draw.text((lx + 142, ly), count, font=font(15, True), fill=color)
    arrow(draw, (1260, 400), (1275, 400), NAVY_2, width=4)

    # Stage 4: show the computation explicitly: typed attention, gated fusion and two layers.
    def flow_arrow(start: tuple[int, int], end: tuple[int, int], color: str, width: int = 2) -> None:
        draw.line((*start, *end), fill=color, width=width)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        size = 7
        left = (end[0] - size * math.cos(angle - 0.55), end[1] - size * math.sin(angle - 0.55))
        right = (end[0] - size * math.cos(angle + 0.55), end[1] - size * math.sin(angle + 0.55))
        draw.polygon((end, left, right), fill=color)

    rounded(draw, (1280, 218, 1745, 535), "#f2f9fa", "#72c1ca", 2)
    draw.text((1305, 238), "关系感知图 Transformer × 2", font=font(24, True), fill=INK)
    draw.text((1305, 274), "边特征参与注意力计算，逐层更新节点表示", font=font(13), fill=MUTED)

    # A local graph neighborhood makes the difference from sequence attention visible.
    rounded(draw, (1305, 318, 1438, 480), WHITE, "#9eb5c9", 1, radius=10)
    draw.text((1322, 332), "局部图邻域", font=font(14, True), fill=INK)
    target = (1371, 405)
    neighbors = [
        ((1328, 370), BLUE),
        ((1412, 367), MAGENTA),
        ((1329, 442), VIOLET),
        ((1411, 445), CYAN),
    ]
    for index, (neighbor, color) in enumerate(neighbors):
        flow_arrow(neighbor, target, color + "cc", width=2)
        nx, ny = neighbor
        draw.ellipse((nx - 10, ny - 10, nx + 10, ny + 10), fill=color, outline=WHITE, width=2)
        # Small edge tokens indicate that relation geometry enters K/V computation.
        ex = int(nx + (target[0] - nx) * 0.43)
        ey = int(ny + (target[1] - ny) * 0.43)
        rounded(draw, (ex - 7, ey - 6, ex + 7, ey + 6), WHITE, color, 1, radius=4)
        draw.text((ex - 3, ey - 5), "e", font=font(7, True), fill=color)
    draw.ellipse((target[0] - 22, target[1] - 22, target[0] + 22, target[1] + 22), fill=NAVY_2, outline=WHITE, width=2)
    target_font = font(7, True)
    draw.text((target[0] - draw.textlength("TARGET", font=target_font) / 2, target[1] - 5), "TARGET", font=target_font, fill=WHITE)

    flow_arrow((1445, 405), (1460, 405), "#8399ad", width=3)
    rounded(draw, (1467, 318, 1612, 480), WHITE, "#63b8c5", 2, radius=10)
    draw.text((1483, 332), "单层结构", font=font(13, True), fill=INK)
    rounded(draw, (1551, 328, 1598, 354), CYAN, None, 0, radius=9)
    heads_font = font(8, True)
    draw.text((1574 - draw.textlength("4 heads", font=heads_font) / 2, 335), "4 heads", font=heads_font, fill=WHITE)
    rounded(draw, (1482, 364, 1597, 417), "#eef5fb", "#83add0", 1, radius=7)
    draw.text((1493, 370), "图注意力", font=font(10, True), fill=BLUE)
    draw.text((1493, 387), "Q   目标节点", font=font(8, True), fill=BLUE)
    draw.text((1493, 401), "K/V  邻居 + 边", font=font(8, True), fill=VIOLET)
    flow_arrow((1539, 420), (1539, 427), "#8ca1b5", width=2)
    rounded(draw, (1482, 432, 1528, 461), "#f4f1f9", "#a995c7", 1, radius=6)
    ffn_font = font(9, True)
    draw.text((1505 - draw.textlength("FFN", font=ffn_font) / 2, 440), "FFN", font=ffn_font, fill=VIOLET)
    flow_arrow((1532, 447), (1538, 447), "#8ca1b5", width=2)
    rounded(draw, (1542, 432, 1601, 461), "#eef8f4", "#8bc5ae", 1, radius=6)
    norm_text = "ADD + NORM"
    norm_font = font(6, True)
    draw.text((1571 - draw.textlength(norm_text, font=norm_font) / 2, 441), norm_text, font=norm_font, fill=GREEN)

    flow_arrow((1619, 405), (1645, 405), "#8399ad", width=3)
    draw.ellipse((1655, 358, 1729, 432), fill=GREEN + "18", outline=GREEN, width=2)
    output_font = font(19, True)
    draw.text((1692 - draw.textlength("128D", font=output_font) / 2, 378), "128D", font=output_font, fill=GREEN)
    output_label = "更新节点特征"
    output_label_font = font(9, True)
    draw.text((1692 - draw.textlength(output_label, font=output_label_font) / 2, 441), output_label, font=output_label_font, fill=GREEN)

    # The final furniture feature produces two independent, easy-to-read outputs.
    rounded(draw, (1280, 560, 1745, 823), "#fff9fb", "#dca9bd", 2)
    draw.text((1305, 579), "逐家具动作预测", font=font(24, True), fill=INK)
    draw.text((1305, 613), "融合图上下文与显式冲突信息，输出修复动作", font=font(13), fill=MUTED)

    # Two sources are fused before the shared representation branches into two heads.
    input_specs = [
        ((1305, 654, 1408, 696), "图特征", "128D", BLUE),
        ((1305, 713, 1408, 755), "冲突特征", "32D", VIOLET),
    ]
    for input_box, label, dims, color in input_specs:
        ix1, iy1, ix2, iy2 = input_box
        rounded(draw, input_box, color + "0d", color + "88", 1, radius=8)
        draw.rectangle((ix1, iy1, ix1 + 6, iy2), fill=color)
        draw.text((ix1 + 16, iy1 + 7), label, font=font(11, True), fill=INK)
        draw.text((ix1 + 70, iy1 + 7), dims, font=font(12, True), fill=color)

    fusion_point = (1432, 705)
    flow_arrow((1412, 675), fusion_point, BLUE + "c8", width=2)
    flow_arrow((1412, 734), fusion_point, VIOLET + "c8", width=2)
    draw.ellipse((1414, 687, 1450, 723), fill=NAVY_2, outline=WHITE, width=2)
    plus_font = font(23, True)
    plus_bbox = draw.textbbox((0, 0), "+", font=plus_font)
    plus_width = plus_bbox[2] - plus_bbox[0]
    plus_height = plus_bbox[3] - plus_bbox[1]
    draw.text((1432 - plus_width / 2, 705 - plus_height / 2 - plus_bbox[1]), "+", font=plus_font, fill=WHITE)
    flow_arrow((1454, 705), (1459, 705), NAVY_2, width=3)

    rounded(draw, (1460, 663, 1555, 747), "#eef3f8", "#8299b1", 2, radius=10)
    draw.text((1476, 676), "家具表示", font=font(12, True), fill=INK)
    feature_font = font(22, True)
    draw.text((1507 - draw.textlength("160D", font=feature_font) / 2, 703), "160D", font=feature_font, fill=NAVY_2)
    draw.text((1481, 731), "共享特征", font=font(9), fill=MUTED)

    output_heads = [
        ((1585, 642, 1725, 704), "动作分类", MAGENTA, ("K", "T", "R", "B")),
        ((1585, 721, 1725, 783), "位姿回归", VIOLET, ("dx", "dy", "dyaw")),
    ]
    for index, (head_box, title, color, tokens) in enumerate(output_heads):
        hx1, hy1, hx2, hy2 = head_box
        flow_arrow((1559, 705), (hx1 - 8, (hy1 + hy2) // 2), color + "c8", width=3)
        rounded(draw, head_box, WHITE, color + "99", 2, radius=9)
        draw.rectangle((hx1, hy1, hx1 + 7, hy2), fill=color)
        draw.text((hx1 + 16, hy1 + 8), title, font=font(12, True), fill=INK)
        draw.text((hx1 + 16, hy1 + 28), "[4]" if index == 0 else "[3]", font=font(9, True), fill=color)
        token_width = 18 if index == 0 else 27
        token_start = hx1 + 43
        token_gap = 3
        for token_index, token in enumerate(tokens):
            tx = token_start + token_index * (token_width + token_gap)
            draw.rounded_rectangle((tx, hy1 + 31, tx + token_width, hy1 + 52), radius=5, fill=color + "28", outline=color)
            tf = font(8, True)
            draw.text((tx + token_width / 2 - draw.textlength(token, font=tf) / 2, hy1 + 36), token, font=tf, fill=color)

    draw.text((1305, 790), "K 保持  ·  T 平移  ·  R 旋转  ·  B 同时调整", font=font(10), fill=MUTED)
    # The poster template supplies its own title/header. Export only the central figure body.
    save(image.crop((0, 139, W, 886)), "01_model_architecture.png")


def _figure_engine_previous_design() -> None:
    image, draw = canvas("可审计的纯净布局数据引擎", "LLM makes scene decisions; deterministic code compiles, verifies and records evidence", "02")
    scene_path = ROOT / "data/clean_layout_production_v2/accepted/scenes/clean_v2_prod_0017_living_room_0001.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))

    # Scientific figures distinguish artifacts, operators and review evidence explicitly.
    def centered_text(cx: int, y: int, text: str, fnt: ImageFont.FreeTypeFont, fill: str) -> None:
        draw.text((cx - draw.textlength(text, font=fnt) / 2, y), text, font=fnt, fill=fill)

    def signal_arrow(start: tuple[int, int], end: tuple[int, int], color: str) -> None:
        """Draw a restrained vector connector for the scientific computation graph."""
        x1, y1 = start
        x2, y2 = end
        draw.line((x1, y1, x2, y2), fill=color, width=2)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 8
        left = (x2 - size * math.cos(angle - 0.48), y2 - size * math.sin(angle - 0.48))
        right = (x2 - size * math.cos(angle + 0.48), y2 - size * math.sin(angle + 0.48))
        draw.polygon(((x2, y2), left, right), fill=color)

    draw.text((56, 170), "可审计纯净布局数据引擎", font=font(34, True), fill=INK)
    draw.text((57, 218), "模型负责提案，代码负责几何校验，独立 Reviewer 决定是否接收", font=font(17), fill=MUTED)
    draw.line((56, 258, 1344, 258), fill="#cbd6e5", width=2)

    # Unframed section rules organize the reading order without adding more cards.
    section_labels = (
        (70, 336, "生成提案", MAGENTA),
        (390, 953, "确定性校验与审查", BLUE),
        (1030, 1340, "接收场景", GREEN),
    )
    for x1, x2, label, color in section_labels:
        draw.text((x1, 318), label, font=font(13, True), fill=color)
        draw.line((x1, 347, x2, 347), fill=color + "66", width=2)

    flow_y = 486

    # Probabilistic generator: density contours and a parameterized distribution.
    for bbox, alpha, width in (
        ((72, 437, 170, 535), "55", 2),
        ((82, 447, 160, 525), "88", 2),
        ((92, 457, 150, 515), "bb", 2),
    ):
        draw.arc(bbox, start=28, end=332, fill=MAGENTA + alpha, width=width)
    for px, py, radius in ((80, 464, 3), (91, 438, 4), (164, 477, 3), (149, 528, 3)):
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=MAGENTA + "aa")
    draw.ellipse((101, 466, 141, 506), fill=MAGENTA)
    centered_text(121, 471, "pθ", font(18, True), WHITE)
    centered_text(121, 410, "LLM GENERATOR", font(9, True), MAGENTA)
    signal_arrow((171, flow_y), (194, flow_y), MAGENTA)

    # Proposal artifact as a compact state tensor, not a document icon.
    centered_text(258, 410, "PROPOSAL STATE", font(11, True), INK)
    draw.text((318, 410), "N×4", font=font(8, True), fill=MAGENTA)
    draw.line((202, 438, 202, 540), fill="#8299b1", width=2)
    draw.line((202, 438, 210, 438), fill="#8299b1", width=2)
    draw.line((202, 540, 210, 540), fill="#8299b1", width=2)
    draw.line((342, 438, 342, 540), fill="#8299b1", width=2)
    draw.line((334, 438, 342, 438), fill="#8299b1", width=2)
    draw.line((334, 540, 342, 540), fill="#8299b1", width=2)
    tensor_x = (216, 246, 276, 306)
    tensor_colors = (MAGENTA, BLUE, CYAN, VIOLET)
    for col, (tx, label, color) in enumerate(zip(tensor_x, ("x", "y", "ψ", "b"), tensor_colors)):
        centered_text(tx + 10, 441, label, font(9, True), color)
        for row, cell_y in enumerate((469, 493, 517)):
            intensity = ("18", "28", "38")[(row + col) % 3]
            draw.rectangle((tx, cell_y, tx + 20, cell_y + 13), fill=color + intensity, outline=color + "66")
    centered_text(272, 566, "proposal.json", font(10, True), MAGENTA)
    signal_arrow((350, flow_y), (378, flow_y), "#7d90a5")

    # Semantic compiler shown as a mathematical operator with explicit I/O ports.
    draw.line((388, 430, 548, 430), fill="#8fb4d3", width=2)
    draw.line((388, 542, 548, 542), fill="#8fb4d3", width=2)
    draw.ellipse((383, flow_y - 4, 391, flow_y + 4), fill=WHITE, outline=BLUE, width=2)
    draw.ellipse((545, flow_y - 4, 553, flow_y + 4), fill=WHITE, outline=BLUE, width=2)
    centered_text(468, 443, "SEMANTIC COMPILER", font(11, True), BLUE)
    centered_text(468, 469, "C_sem", font(20, True), INK)
    centered_text(468, 507, "intent  ↦  (yaw, clearance)", font(9), MUTED)
    signal_arrow((553, flow_y), (578, flow_y), BLUE)

    # Deterministic geometry is a vector of constraint residuals g(s-hat).
    centered_text(668, 410, "HARD CONSTRAINTS", font(11, True), BLUE)
    draw.line((584, 430, 584, 544), fill="#8fb4d3", width=2)
    draw.line((584, 430, 592, 430), fill="#8fb4d3", width=2)
    draw.line((584, 544, 592, 544), fill="#8fb4d3", width=2)
    draw.line((752, 430, 752, 544), fill="#8fb4d3", width=2)
    draw.line((744, 430, 752, 430), fill="#8fb4d3", width=2)
    draw.line((744, 544, 752, 544), fill="#8fb4d3", width=2)
    centered_text(668, 439, "g(ŝ) = 0", font(18, True), INK)
    constraint_rows = (("BND", 605, 482), ("COL", 681, 482), ("CLR", 605, 514), ("PATH", 681, 514))
    for label, tx, ty in constraint_rows:
        draw.text((tx, ty), label, font=font(8, True), fill=MUTED)
        draw.line((tx + 28, ty + 7, tx + 52, ty + 7), fill="#b8c7d7", width=2)
        draw.ellipse((tx + 48, ty + 3, tx + 56, ty + 11), fill=GREEN, outline=WHITE)
        draw.text((tx + 59, ty - 1), "0", font=font(8, True), fill=GREEN)
    centered_text(668, 566, "compile_report.json", font(10, True), BLUE)
    signal_arrow((758, flow_y), (778, flow_y), BLUE)

    # Independent review is visualized as two measured evidence scores.
    centered_text(866, 410, "INDEPENDENT REVIEW", font(11, True), VIOLET)
    centered_text(866, 434, "R_blind(ŝ, e)", font(12, True), INK)
    score_rows = (("semantic", 472, 0.88), ("diversity", 511, 0.78))
    for label, ty, score in score_rows:
        draw.text((784, ty - 6), label, font=font(8, True), fill=MUTED)
        x_start, x_end = 837, 944
        threshold_x = 909
        score_x = int(x_start + (x_end - x_start) * score)
        draw.line((x_start, ty, x_end, ty), fill="#b8c7d7", width=3)
        draw.line((threshold_x, ty - 7, threshold_x, ty + 7), fill=VIOLET + "88", width=1)
        draw.ellipse((score_x - 5, ty - 5, score_x + 5, ty + 5), fill=VIOLET, outline=WHITE)
    draw.text((900, 529), "v = 1", font=font(9, True), fill=GREEN)
    centered_text(866, 566, "review_verdict.json", font(10, True), VIOLET)
    signal_arrow((962, flow_y), (1040, flow_y), VIOLET)

    # The verified 3D scene is the visual endpoint; record bookkeeping is omitted here.
    draw_3d_scene(draw, scene, (970, 405, 1390, 605), show_labels=False, scale_factor=0.88, show_contract_label=False)

    # Rejections become an explicit residual/evidence vector returned to the generator.
    for x, start_y in ((610, 548), (790, 548)):
        for dash_y in range(start_y, 721, 13):
            draw.line((x, dash_y, x, dash_y + 7), fill=RED + "66", width=2)
    centered_text(470, 679, "revision evidence   e = [ g(ŝ), review verdict ]", font(11, True), RED)
    centered_text(470, 702, "reject / revise  ·  no silent repair", font(9), RED)
    for dash_x in range(790, 133, -50):
        draw.line((dash_x, 721, max(dash_x - 26, 121), 721), fill=RED + "a0", width=2)
    arrow(draw, (121, 721), (121, 546), RED, width=2, dashed=True)

    # The poster template supplies the page title and footer; export only the figure body.
    save(image.crop((0, 139, 1400, 839)), "02_data_engine_pipeline.png")


def figure_engine() -> None:
    # This figure uses a dedicated near-square canvas instead of the shared landscape page.
    image = Image.new("RGB", (1200, 1260), WHITE)
    draw = ImageDraw.Draw(image, "RGBA")
    scene_path = ROOT / "data/clean_layout_production_v2/accepted/scenes/clean_v2_prod_0017_living_room_0001.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))

    def centered_text(cx: int, y: int, text: str, fnt: ImageFont.FreeTypeFont, fill: str) -> None:
        draw.text((cx - draw.textlength(text, font=fnt) / 2, y), text, font=fnt, fill=fill)

    def signal_arrow(start: tuple[int, int], end: tuple[int, int], color: str, width: int = 2) -> None:
        x1, y1 = start
        x2, y2 = end
        draw.line((x1, y1, x2, y2), fill=color, width=width)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 8
        left = (x2 - size * math.cos(angle - 0.48), y2 - size * math.sin(angle - 0.48))
        right = (x2 - size * math.cos(angle + 0.48), y2 - size * math.sin(angle + 0.48))
        draw.polygon(((x2, y2), left, right), fill=color)

    def draw_topdown(box: tuple[int, int, int, int], *, draft: bool = False, validated: bool = False) -> None:
        x1, y1, x2, y2 = box
        length, room_width = float(scene["length"]), float(scene["width"])
        scale = min((x2 - x1 - 18) / length, (y2 - y1 - 18) / room_width)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        def point(x: float, y: float) -> tuple[int, int]:
            return int(cx + x * scale), int(cy - y * scale)

        floor = (
            *point(-length / 2, room_width / 2),
            *point(length / 2, -room_width / 2),
        )
        draw.rectangle(floor, fill="#f7f9fc", outline=GREEN if validated else "#56708f", width=3 if validated else 2)

        for fraction in (0.25, 0.5, 0.75):
            gx = -length / 2 + length * fraction
            gy = -room_width / 2 + room_width * fraction
            draw.line((point(gx, -room_width / 2), point(gx, room_width / 2)), fill="#dbe4ee", width=1)
            draw.line((point(-length / 2, gy), point(length / 2, gy)), fill="#dbe4ee", width=1)

        for opening in scene["openings"]:
            ox, oy, _ = opening["clearance_center_xyz_m"]
            ow, od, _ = opening["clearance_size_xyz_m"]
            a = point(float(ox) - float(ow) / 2, float(oy) + float(od) / 2)
            b = point(float(ox) + float(ow) / 2, float(oy) - float(od) / 2)
            color = RED if str(opening["kind"]) == "door" else CYAN
            draw.rectangle((*a, *b), fill=color + "20", outline=color + "aa", width=1)

        object_colors = (GREEN, AMBER, VIOLET, CYAN, MAGENTA, "#8aa34a", "#a87945")
        for index, obj in enumerate(scene["furniture"]):
            width_m = float(obj["bbox"]["width"])
            depth_m = float(obj["bbox"]["depth"])
            yaw = float(obj["yaw_rad"])
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            corners: list[tuple[int, int]] = []
            for lx, ly in ((-width_m / 2, -depth_m / 2), (width_m / 2, -depth_m / 2), (width_m / 2, depth_m / 2), (-width_m / 2, depth_m / 2)):
                wx = float(obj["x"]) + lx * cos_yaw - ly * sin_yaw
                wy = float(obj["y"]) + lx * sin_yaw + ly * cos_yaw
                corners.append(point(wx, wy))
            color = object_colors[index % len(object_colors)]
            draw.polygon(corners, fill=color + ("28" if draft else "45"), outline=color)
            if draft and index < 4:
                origin = point(float(obj["x"]), float(obj["y"]))
                heading = point(float(obj["x"]) + 0.34 * math.cos(yaw), float(obj["y"]) + 0.34 * math.sin(yaw))
                signal_arrow(origin, heading, color, width=1)

        if validated:
            path_points = (
                point(-length * 0.42, -room_width * 0.30),
                point(-length * 0.10, -room_width * 0.05),
                point(length * 0.18, room_width * 0.02),
                point(length * 0.42, room_width * 0.28),
            )
            draw.line(path_points, fill=CYAN + "aa", width=4, joint="curve")
            for px, py in path_points:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=CYAN)

    def stage_marker(number: str, cy: int, title: str, subtitle: str, color: str) -> None:
        draw.ellipse((53, cy - 17, 87, cy + 17), fill=color)
        centered_text(70, cy - 9, number, font(10, True), WHITE)
        draw.text((110, cy - 28), title, font=font(20, True), fill=INK)
        draw.text((110, cy + 7), subtitle, font=font(10), fill=MUTED)

    # Figure title and one-sentence reading key.
    draw.text((56, 170), "可审计纯净场景生成", font=font(34, True), fill=INK)
    draw.text((57, 218), "MLLM 草案 → 确定性落实 → 几何校验 → 语义校验 → 完成生成", font=font(17), fill=MUTED)
    draw.line((56, 258, 1144, 258), fill="#cbd6e5", width=2)

    stage_centers = (350, 500, 650, 800, 985)
    stage_colors = (MAGENTA, BLUE, CYAN, VIOLET, GREEN)
    for index in range(len(stage_centers) - 1):
        signal_arrow((70, stage_centers[index] + 24), (70, stage_centers[index + 1] - 24), "#aebdce", width=2)

    stage_marker("01", 350, "MLLM 给出草案", "理解房间条件并提出家具位置与朝向", MAGENTA)
    stage_marker("02", 500, "草案落实", "将自然语言意图编译为确定性三维几何", BLUE)
    stage_marker("03", 650, "几何校验", "检查边界、碰撞、开口净空与通行路径", CYAN)
    stage_marker("04", 800, "语义校验", "检查房间用途、家具组合与布局多样性", VIOLET)
    stage_marker("05", 985, "完成生成", "输出通过双重校验的纯净三维场景", GREEN)

    # 01: condition tokens are condensed by the MLLM into a visible layout draft.
    prompt_items = (("ROOM TYPE", MAGENTA), ("OPENINGS", CYAN), ("HSSD ASSETS", AMBER))
    for index, (label, color) in enumerate(prompt_items):
        ty = 315 + index * 28
        draw.line((350, ty + 7, 366, ty + 7), fill=color, width=4)
        draw.text((375, ty), label, font=font(9, True), fill=MUTED)
        draw.line((467, ty + 7, 505, 350), fill="#c3cfdd", width=1)
    signal_arrow((505, 350), (522, 350), MAGENTA)
    draw.ellipse((526, 306, 614, 394), fill=MAGENTA + "0d", outline=MAGENTA + "77", width=2)
    draw.ellipse((538, 318, 602, 382), fill=WHITE, outline=MAGENTA, width=2)
    draw.ellipse((548, 328, 592, 372), fill=MAGENTA)
    centered_text(570, 337, "MLLM", font(11, True), WHITE)
    signal_arrow((616, 350), (690, 350), MAGENTA)
    draw_topdown((705, 292, 1110, 408), draft=True)
    centered_text(908, 411, "layout draft", font(9, True), MAGENTA)

    # 02: the draft parameters are materialized into typed 3D geometry.
    draw.text((350, 450), "DRAFT PARAMETERS", font=font(10, True), fill=BLUE)
    parameter_rows = (("sofa", "x  y  yaw"), ("table", "x  y  yaw"), ("chair", "x  y  yaw"))
    for index, (name, values) in enumerate(parameter_rows):
        ty = 478 + index * 24
        draw.text((350, ty), name, font=font(9, True), fill=INK)
        draw.text((408, ty), values, font=font(8), fill=MUTED)
        draw.line((350, ty + 18, 486, ty + 18), fill="#dbe4ee", width=1)
    signal_arrow((496, 500), (535, 500), BLUE)
    draw.line((544, 458, 544, 542), fill=BLUE, width=2)
    draw.line((544, 458, 554, 458), fill=BLUE, width=2)
    draw.line((544, 542, 554, 542), fill=BLUE, width=2)
    draw.line((682, 458, 682, 542), fill=BLUE, width=2)
    draw.line((672, 458, 682, 458), fill=BLUE, width=2)
    draw.line((672, 542, 682, 542), fill=BLUE, width=2)
    centered_text(613, 475, "COMPILE", font(12, True), BLUE)
    centered_text(613, 505, "intent → geometry", font(9), MUTED)
    signal_arrow((691, 500), (744, 500), BLUE)
    draw_3d_scene(draw, scene, (735, 400, 1120, 535), show_labels=False, scale_factor=0.82, show_contract_label=False)

    # 03: the same scene is checked directly in geometric space.
    draw_topdown((350, 585, 660, 710), validated=True)
    geometry_checks = ("BOUNDARY", "COLLISION", "CLEARANCE", "PATH")
    for index, label in enumerate(geometry_checks):
        ty = 600 + index * 29
        draw.text((720, ty), label, font=font(9, True), fill=MUTED)
        draw.line((818, ty + 7, 1035, ty + 7), fill="#c9d5e2", width=3)
        draw.ellipse((1028, ty, 1042, ty + 14), fill=GREEN, outline=WHITE)
        draw.text((1052, ty - 2), "PASS", font=font(9, True), fill=GREEN)

    # 04: a compact relation graph and score traces make semantic review explicit.
    graph_center = (515, 800)
    graph_nodes = (
        ((515, 748), "ROOM", BLUE),
        ((430, 765), "SOFA", GREEN),
        ((420, 832), "TABLE", AMBER),
        ((515, 850), "SEAT", VIOLET),
        ((600, 830), "LAMP", MAGENTA),
        ((610, 764), "OPEN", CYAN),
    )
    for (nx, ny), _, color in graph_nodes[1:]:
        draw.line((graph_center, (nx, ny)), fill=color + "66", width=2)
    for left_index in range(1, len(graph_nodes)):
        for right_index in range(left_index + 1, len(graph_nodes)):
            if (left_index + right_index) % 2 == 0:
                draw.line((graph_nodes[left_index][0], graph_nodes[right_index][0]), fill="#cbd6e566", width=1)
    for (nx, ny), label, color in graph_nodes:
        radius = 21 if label == "ROOM" else 17
        draw.ellipse((nx - radius, ny - radius, nx + radius, ny + radius), fill=WHITE, outline=color, width=2)
        centered_text(nx, ny - 6, label, font(7, True), color)

    semantic_scores = (("room-function", 755, 0.91), ("furniture set", 795, 0.86), ("diversity", 835, 0.79))
    for label, ty, score in semantic_scores:
        draw.text((710, ty - 7), label, font=font(9, True), fill=MUTED)
        start_x, end_x, threshold_x = 820, 1040, 970
        score_x = int(start_x + (end_x - start_x) * score)
        draw.line((start_x, ty, end_x, ty), fill="#c9d5e2", width=3)
        draw.line((threshold_x, ty - 8, threshold_x, ty + 8), fill=VIOLET + "88", width=1)
        draw.ellipse((score_x - 6, ty - 6, score_x + 6, ty + 6), fill=VIOLET, outline=WHITE)
    draw.text((1052, 793), "PASS", font=font(10, True), fill=GREEN)

    # 05: a larger real sample is the visual endpoint of the five-stage pipeline.
    draw_3d_scene(draw, scene, (330, 855, 1140, 1090), show_labels=False, scale_factor=1.02, show_contract_label=False)

    figure_body = image.crop((0, 139, 1200, 1200))
    target_width = 870
    target_height = round(figure_body.height * target_width / figure_body.width)
    save(figure_body.resize((target_width, target_height), Image.Resampling.LANCZOS), "02_data_engine_pipeline.png")


def figure_engine_compact() -> None:
    """Render the five essential engine stages as a compact two-row academic schematic."""
    image = Image.new("RGB", (1400, 760), WHITE)
    draw = ImageDraw.Draw(image, "RGBA")
    scene_path = ROOT / "data/clean_layout_production_v2/accepted/scenes/clean_v2_prod_0017_living_room_0001.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))

    def centered_text(cx: int, y: int, text: str, fnt: ImageFont.FreeTypeFont, fill: str) -> None:
        draw.text((cx - draw.textlength(text, font=fnt) / 2, y), text, font=fnt, fill=fill)

    def signal_arrow(start: tuple[int, int], end: tuple[int, int], color: str) -> None:
        x1, y1 = start
        x2, y2 = end
        draw.line((x1, y1, x2, y2), fill=color, width=2)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 9
        left = (x2 - size * math.cos(angle - 0.48), y2 - size * math.sin(angle - 0.48))
        right = (x2 - size * math.cos(angle + 0.48), y2 - size * math.sin(angle + 0.48))
        draw.polygon(((x2, y2), left, right), fill=color)

    def stage_heading(x: int, y: int, width: int, number: str, title: str, subtitle: str, color: str) -> None:
        draw.ellipse((x, y, x + 30, y + 30), fill=color)
        centered_text(x + 15, y + 6, number, font(9, True), WHITE)
        draw.text((x + 43, y - 2), title, font=font(18, True), fill=INK)
        draw.text((x + 43, y + 29), subtitle, font=font(9), fill=MUTED)
        draw.line((x, y + 58, x + width, y + 58), fill=color + "55", width=2)

    def draw_topdown(box: tuple[int, int, int, int], *, validated: bool = False) -> None:
        x1, y1, x2, y2 = box
        length, room_width = float(scene["length"]), float(scene["width"])
        scale = min((x2 - x1 - 14) / length, (y2 - y1 - 14) / room_width)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        def point(x: float, y: float) -> tuple[int, int]:
            return int(cx + x * scale), int(cy - y * scale)

        floor = (*point(-length / 2, room_width / 2), *point(length / 2, -room_width / 2))
        draw.rectangle(floor, fill="#f7f9fc", outline=GREEN if validated else "#56708f", width=3 if validated else 2)
        for opening in scene["openings"]:
            ox, oy, _ = opening["clearance_center_xyz_m"]
            ow, od, _ = opening["clearance_size_xyz_m"]
            a = point(float(ox) - float(ow) / 2, float(oy) + float(od) / 2)
            b = point(float(ox) + float(ow) / 2, float(oy) - float(od) / 2)
            color = RED if str(opening["kind"]) == "door" else CYAN
            draw.rectangle((*a, *b), fill=color + "1c", outline=color + "aa", width=1)

        object_colors = (GREEN, AMBER, VIOLET, CYAN, MAGENTA, "#8aa34a", "#a87945")
        for index, obj in enumerate(scene["furniture"]):
            width_m = float(obj["bbox"]["width"])
            depth_m = float(obj["bbox"]["depth"])
            yaw = float(obj["yaw_rad"])
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            corners: list[tuple[int, int]] = []
            for lx, ly in ((-width_m / 2, -depth_m / 2), (width_m / 2, -depth_m / 2), (width_m / 2, depth_m / 2), (-width_m / 2, depth_m / 2)):
                wx = float(obj["x"]) + lx * cos_yaw - ly * sin_yaw
                wy = float(obj["y"]) + lx * sin_yaw + ly * cos_yaw
                corners.append(point(wx, wy))
            color = object_colors[index % len(object_colors)]
            draw.polygon(corners, fill=color + "35", outline=color)

        if validated:
            route = (
                point(-length * 0.40, -room_width * 0.28),
                point(-length * 0.12, -room_width * 0.04),
                point(length * 0.18, room_width * 0.03),
                point(length * 0.40, room_width * 0.28),
            )
            draw.line(route, fill=CYAN + "bb", width=4, joint="curve")
            for px, py in route:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=CYAN)

    draw.text((48, 28), "可审计纯净场景生成", font=font(32, True), fill=INK)
    draw.text((49, 72), "MLLM 草案 → 草案落实 → 几何校验 → 语义校验 → 最终场景", font=font(14), fill=MUTED)
    draw.line((48, 108, 1352, 108), fill=LINE, width=2)

    # Top row: proposal, deterministic realization, then geometry validation.
    stage_heading(55, 135, 380, "01", "MLLM 给出草案", "房间条件 → 布局草图", MAGENTA)
    stage_heading(500, 135, 350, "02", "草案落实", "语义意图 → 三维几何", BLUE)
    stage_heading(915, 135, 430, "03", "几何校验", "边界 · 碰撞 · 净空 · 路径", CYAN)
    signal_arrow((446, 280), (489, 280), NAVY_2)
    signal_arrow((861, 280), (904, 280), NAVY_2)

    # 01: three compact conditions enter the MLLM and become a layout draft.
    for index, (label, color) in enumerate((("ROOM", MAGENTA), ("OPENING", CYAN), ("ASSETS", AMBER))):
        ty = 213 + index * 25
        draw.line((72, ty + 6, 86, ty + 6), fill=color, width=3)
        draw.text((93, ty), label, font=font(8, True), fill=MUTED)
        draw.line((150, ty + 6, 174, 270), fill="#c6d2df", width=1)
    draw.ellipse((179, 232, 255, 308), fill=MAGENTA + "0d", outline=MAGENTA + "88", width=2)
    draw.ellipse((191, 244, 243, 296), fill=WHITE, outline=MAGENTA, width=2)
    draw.ellipse((199, 252, 235, 288), fill=MAGENTA)
    centered_text(217, 261, "MLLM", font(9, True), WHITE)
    signal_arrow((258, 270), (282, 270), MAGENTA)
    draw_topdown((287, 211, 426, 329))

    # 02: a small draft is compiled into a typed 3D scene.
    draw_topdown((515, 216, 624, 326))
    signal_arrow((632, 270), (653, 270), BLUE)
    draw.line((661, 224, 661, 316), fill=BLUE, width=2)
    draw.line((661, 224, 669, 224), fill=BLUE, width=2)
    draw.line((661, 316, 669, 316), fill=BLUE, width=2)
    draw.line((730, 224, 730, 316), fill=BLUE, width=2)
    draw.line((722, 224, 730, 224), fill=BLUE, width=2)
    draw.line((722, 316, 730, 316), fill=BLUE, width=2)
    centered_text(696, 248, "COMPILE", font(10, True), BLUE)
    centered_text(696, 278, "3D", font(17, True), INK)
    signal_arrow((738, 270), (756, 270), BLUE)
    draw_3d_scene(draw, scene, (748, 202, 850, 334), show_labels=False, scale_factor=0.82, show_contract_label=False)

    # 03: one plan and four pass channels summarize deterministic geometry checks.
    draw_topdown((930, 210, 1092, 338), validated=True)
    for index, label in enumerate(("BND", "COL", "CLR", "PATH")):
        ty = 220 + index * 30
        draw.text((1112, ty), label, font=font(8, True), fill=MUTED)
        draw.line((1153, ty + 7, 1300, ty + 7), fill="#c9d5e2", width=3)
        draw.ellipse((1294, ty + 1, 1306, ty + 13), fill=GREEN, outline=WHITE)
        draw.text((1314, ty - 1), "PASS", font=font(8, True), fill=GREEN)

    # The reading path turns once, then finishes right-to-left on the second row.
    signal_arrow((1130, 365), (1130, 410), NAVY_2)
    stage_heading(915, 420, 430, "04", "语义校验", "关系一致性 · 布局多样性", VIOLET)
    stage_heading(55, 420, 795, "05", "最终场景", "通过双重校验的纯净三维布局", GREEN)
    signal_arrow((904, 575), (861, 575), NAVY_2)

    # 04: relation structure plus two concise reviewer scores.
    center = (1020, 572)
    graph_nodes = (((1020, 525), "ROOM", BLUE), ((962, 553), "SOFA", GREEN), ((970, 620), "TABLE", AMBER), ((1078, 552), "OPEN", CYAN), ((1070, 620), "LAMP", MAGENTA))
    for (nx, ny), _, color in graph_nodes[1:]:
        draw.line((center, (nx, ny)), fill=color + "77", width=2)
    for (nx, ny), label, color in graph_nodes:
        radius = 18 if label == "ROOM" else 15
        draw.ellipse((nx - radius, ny - radius, nx + radius, ny + radius), fill=WHITE, outline=color, width=2)
        centered_text(nx, ny - 5, label, font(6, True), color)
    for label, ty, score in (("coherence", 548, 0.90), ("diversity", 600, 0.80)):
        draw.text((1110, ty - 6), label, font=font(8, True), fill=MUTED)
        start_x, end_x, threshold_x = 1175, 1320, 1275
        score_x = int(start_x + (end_x - start_x) * score)
        draw.line((start_x, ty, end_x, ty), fill="#c9d5e2", width=3)
        draw.line((threshold_x, ty - 7, threshold_x, ty + 7), fill=VIOLET + "88", width=1)
        draw.ellipse((score_x - 5, ty - 5, score_x + 5, ty + 5), fill=VIOLET, outline=WHITE)
    draw.text((1278, 632), "PASS", font=font(9, True), fill=GREEN)

    # 05: the final scene receives the largest visual area, with no extra bookkeeping.
    draw_3d_scene(draw, scene, (170, 450, 850, 650), show_labels=False, scale_factor=0.85, show_contract_label=False)

    save(image, "02_data_engine_pipeline.png")


def figure_engine_vertical_modules() -> None:
    """Render the engine as a narrow vertical companion to the dataset overview."""
    image = Image.new("RGB", (650, 660), WHITE)
    draw = ImageDraw.Draw(image, "RGBA")

    def centered_text(cx: int, y: int, text: str, fnt: ImageFont.FreeTypeFont, fill: str) -> None:
        draw.text((cx - draw.textlength(text, font=fnt) / 2, y), text, font=fnt, fill=fill)

    def signal_arrow(start: tuple[int, int], end: tuple[int, int], color: str) -> None:
        x1, y1 = start
        x2, y2 = end
        draw.line((x1, y1, x2, y2), fill=color, width=2)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 9
        left = (x2 - size * math.cos(angle - 0.48), y2 - size * math.sin(angle - 0.48))
        right = (x2 - size * math.cos(angle + 0.48), y2 - size * math.sin(angle + 0.48))
        draw.polygon(((x2, y2), left, right), fill=color)

    def compact_box(box: tuple[int, int, int, int], number: str, title: str, subtitle: str, color: str) -> None:
        x1, y1, x2, y2 = box
        draw.rounded_rectangle(box, radius=5, fill=color + "08", outline=color + "aa", width=2)
        draw.text((x1 + 16, y1 + 12), number, font=font(11, True), fill=color)
        draw.text((x1 + 50, y1 + 9), title, font=font(20, True), fill=INK)
        draw.text((x1 + 50, y1 + 45), subtitle, font=font(11), fill=MUTED)

    draw.text((34, 22), "可审计纯净布局数据引擎", font=font(30, True), fill=INK)
    draw.text((35, 64), "MLLM 布局提案 → 确定性几何编译 → 双重验证", font=font(13), fill=MUTED)
    draw.line((34, 98, 616, 98), fill=LINE, width=2)

    # All top-level modules share the same vertical axis and outer width.
    compact_box((40, 120, 610, 200), "01", "MLLM 布局提案", "房间条件 → 家具位置与朝向", MAGENTA)
    signal_arrow((325, 208), (325, 230), NAVY_2)
    compact_box((40, 240, 610, 320), "02", "确定性几何编译", "布局意图 → 可校验三维几何", BLUE)
    signal_arrow((325, 328), (325, 350), NAVY_2)

    # Validation is one module containing two required, separately legible checks.
    validation_box = (40, 360, 610, 640)
    draw.rounded_rectangle(validation_box, radius=6, fill="#f9fbfd", outline="#8fb4d3", width=2)
    draw.text((62, 379), "03", font=font(11, True), fill=CYAN)
    draw.text((96, 374), "双重验证", font=font(22, True), fill=INK)
    draw.text((96, 411), "两项共同构成场景接收判据", font=font(11), fill=MUTED)

    geometry_box = (75, 445, 575, 515)
    semantic_box = (75, 560, 575, 630)
    draw.rounded_rectangle(geometry_box, radius=4, fill=CYAN + "08", outline=CYAN + "aa", width=2)
    draw.text((96, 455), "几何约束校验", font=font(18, True), fill=INK)
    draw.text((96, 488), "边界 · 碰撞 · 开口净空 · 通行路径", font=font(10), fill=MUTED)
    draw.rounded_rectangle(semantic_box, radius=4, fill=VIOLET + "08", outline=VIOLET + "aa", width=2)
    draw.text((96, 570), "语义一致性审核", font=font(18, True), fill=INK)
    draw.text((96, 603), "房间用途 · 家具组合 · 布局多样性", font=font(10), fill=MUTED)
    draw.ellipse((312, 525, 338, 551), fill=GREEN, outline=WHITE)
    centered_text(325, 528, "+", font(14, True), WHITE)

    save(image, "02_data_engine_pipeline.png")


def figure_training() -> None:
    figure_width, figure_height = 1800, 580
    image = Image.new("RGB", (figure_width, figure_height), WHITE)
    draw = ImageDraw.Draw(image, "RGBA")

    paper_ink = "#182230"
    paper_arrow = "#364152"
    panel_line = "#cfd9e5"
    data_color = "#168a7a"
    aug_color = "#c47a18"
    model_color = "#2d6da8"
    target_color = "#6b58a5"
    loss_color = "#b65f3c"

    def panel(box: tuple[int, int, int, int], title: str, color: str, number: str) -> None:
        x1, y1, x2, y2 = box
        draw.rounded_rectangle(box, radius=6, fill=WHITE, outline=panel_line, width=2)
        draw.rectangle((x1, y1, x1 + 5, y2), fill=color)
        draw.ellipse((x1 + 20, y1 + 16, x1 + 48, y1 + 44), fill=color)
        number_font = font(13, True)
        number_box = draw.textbbox((0, 0), number, font=number_font)
        draw.text(
            (
                x1 + 34 - (number_box[2] - number_box[0]) / 2,
                y1 + 30 - (number_box[3] - number_box[1]) / 2 - number_box[1],
            ),
            number,
            font=number_font,
            fill=WHITE,
        )
        draw.text((x1 + 62, y1 + 17), title, font=font(21, True), fill=paper_ink)
        draw.line((x1 + 18, y1 + 58, x2 - 18, y1 + 58), fill=panel_line, width=1)

    def dashed_segment(
        start: tuple[int, int], end: tuple[int, int], color: str, width: int = 2
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        distance = math.hypot(x2 - x1, y2 - y1)
        if distance == 0:
            return
        position = 0.0
        while position < distance:
            segment_end = min(position + 10, distance)
            a, b = position / distance, segment_end / distance
            draw.line(
                (
                    x1 + (x2 - x1) * a,
                    y1 + (y2 - y1) * a,
                    x1 + (x2 - x1) * b,
                    y1 + (y2 - y1) * b,
                ),
                fill=color,
                width=width,
            )
            position += 17

    def route(
        points: list[tuple[int, int]], color: str, width: int = 2, *, dashed: bool = False
    ) -> None:
        for start, end in zip(points[:-2], points[1:-1]):
            if dashed:
                dashed_segment(start, end, color, width)
            else:
                draw.line((*start, *end), fill=color, width=width)
        arrow(draw, points[-2], points[-1], color, width=width, dashed=dashed)

    clean_box = (30, 115, 415, 470)
    corrupted_box = (460, 115, 900, 470)
    network_box = (945, 115, 1340, 470)
    output_box = (1385, 115, 1770, 470)

    panel(clean_box, "Clean layouts", data_color, "1")
    panel(corrupted_box, "Corrupted scene graph", aug_color, "2")
    panel(network_box, "SceneRepair Network", model_color, "3")
    panel(output_box, "Prediction & Loss", loss_color, "4")

    scene_path = ROOT / "data/clean_layout_production_v2/accepted/scenes/clean_v2_prod_0013_home_office_0007.json"
    clean_scene = json.loads(scene_path.read_text(encoding="utf-8"))
    corrupted_scene = copy.deepcopy(clean_scene)
    perturbations = {
        "desk_01": (0.55, -0.62, -2.50),
        "chair_01": (0.72, -0.48, 1.10),
        "armchair_01": (-0.22, -0.78, 0.32),
        "cabinet_01": (0.72, 1.42, 0.42),
    }
    for obj in corrupted_scene["furniture"]:
        object_id = str(obj["object_id"])
        if object_id in perturbations:
            obj["x"], obj["y"], obj["yaw_rad"] = perturbations[object_id]

    clean_scene_box = (42, 150, 403, 382)
    corrupted_scene_box = (472, 150, 888, 382)
    draw_3d_scene(
        draw,
        clean_scene,
        clean_scene_box,
        show_labels=False,
        scale_factor=0.90,
        show_contract_label=False,
    )
    furniture_pos, opening_pos, _ = draw_3d_scene(
        draw,
        corrupted_scene,
        corrupted_scene_box,
        show_labels=False,
        scale_factor=0.90,
        show_contract_label=False,
        highlight_ids=set(perturbations),
        highlight_color=RED,
    )

    # A small, representative subset of typed relations keeps the graph legible.
    relation_edges = (
        ("desk_01", "chair_01", target_color),
        ("armchair_01", "side_table_01", target_color),
        ("desk_01", "armchair_01", "#63758a"),
        ("cabinet_01", "bookcase_02", "#63758a"),
        ("chair_01", "bookcase_01", "#63758a"),
    )
    for source, target, color in relation_edges:
        if source not in furniture_pos or target not in furniture_pos:
            continue
        start, end = furniture_pos[source], furniture_pos[target]
        draw.line((*start, *end), fill=color + "9e", width=2)
    for object_id in {node for edge in relation_edges for node in edge[:2]}:
        if object_id in furniture_pos:
            x, y = furniture_pos[object_id]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=WHITE, outline="#30455d", width=2)
    for opening_id, (x, y) in opening_pos.items():
        if opening_id == "door_00":
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=WHITE, outline=RED, width=2)

    legend_font = font(12)
    draw.ellipse((508, 437, 518, 447), fill=RED)
    draw.text((525, 432), "perturbed furniture", font=legend_font, fill=MUTED)
    draw.line((675, 442, 697, 442), fill=target_color, width=2)
    draw.text((705, 432), "typed relation", font=legend_font, fill=MUTED)

    # Scene graph -> relation-aware Graph Transformer -> furniture node states.
    graph_nodes = (
        (990, 247, data_color),
        (1032, 220, aug_color),
        (1043, 278, target_color),
        (996, 318, model_color),
        (1052, 350, loss_color),
    )
    for first, second in ((0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)):
        draw.line((graph_nodes[first][0], graph_nodes[first][1], graph_nodes[second][0], graph_nodes[second][1]), fill="#8b9aab", width=2)
    for x, y, color in graph_nodes:
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=WHITE, outline=color, width=3)
    draw.text((972, 383), "typed graph", font=font(13), fill=MUTED)

    arrow(draw, (1070, 285), (1100, 285), paper_arrow, width=2)
    for offset in (10, 0):
        layer_box = (1102 + offset, 219 - offset, 1252 + offset, 355 - offset)
        draw.rounded_rectangle(layer_box, radius=6, fill="#f7faff", outline=model_color + "a0", width=2)
    transformer_font = font(16, True)
    for index, text_value in enumerate(("Relation-aware", "Graph", "Transformer")):
        text_width = draw.textlength(text_value, font=transformer_font)
        draw.text((1177 - text_width / 2, 246 + index * 24), text_value, font=transformer_font, fill=paper_ink)
    draw.rounded_rectangle((1208, 326, 1240, 349), radius=10, fill=model_color)
    draw.text((1217, 328), "x2", font=font(11, True), fill=WHITE)

    arrow(draw, (1262, 285), (1282, 285), paper_arrow, width=2)
    for index, color in enumerate((data_color, aug_color, target_color, model_color)):
        y = 235 + index * 39
        draw.rounded_rectangle((1287, y, 1318, y + 22), radius=4, fill=color + "24", outline=color, width=2)
    states_font = font(12)
    for index, line in enumerate(("furniture", "states")):
        line_width = draw.textlength(line, font=states_font)
        draw.text((1303 - line_width / 2, 391 + index * 17), line, font=states_font, fill=MUTED)

    # Two explicit prediction heads converge into the joint training objective.
    action_head = (1415, 214, 1550, 332)
    pose_head = (1582, 214, 1738, 332)
    for box, title in ((action_head, "Action"), (pose_head, "Pose delta")):
        draw.rounded_rectangle(box, radius=6, fill="#f8fafc", outline=panel_line, width=2)
        title_width = draw.textlength(title, font=font(16, True))
        draw.text(((box[0] + box[2] - title_width) / 2, box[1] + 13), title, font=font(16, True), fill=paper_ink)
    for index, height in enumerate((13, 25, 18, 33)):
        x = 1440 + index * 24
        draw.rounded_rectangle((x, 306 - height, x + 11, 306), radius=2, fill=model_color + "c8")
    for index, label in enumerate(("dx", "dy", "yaw")):
        x = 1600 + index * 42
        draw.text((x, 272), label, font=font(12), fill=MUTED)
        draw.line((x, 303, x + 24, 303), fill=model_color, width=3)
        draw.ellipse((x + 9, 294 - index * 6, x + 17, 302 - index * 6), fill=model_color)

    merge_x, merge_y = 1578, 382
    action_mid = (action_head[0] + action_head[2]) // 2
    pose_mid = (pose_head[0] + pose_head[2]) // 2
    draw.line((action_mid, action_head[3], action_mid, 348), fill="#8b9aab", width=2)
    draw.line((pose_mid, pose_head[3], pose_mid, 348), fill="#8b9aab", width=2)
    draw.line((action_mid, 348, merge_x, merge_y - 22), fill="#8b9aab", width=2)
    draw.line((pose_mid, 348, merge_x, merge_y - 22), fill="#8b9aab", width=2)
    draw.ellipse((merge_x - 22, merge_y - 22, merge_x + 22, merge_y + 22), fill="#fff5f1", outline=loss_color, width=3)
    draw.text((merge_x - 7, merge_y - 16), "L", font=font(24, True), fill=loss_color)
    draw.text((1513, 424), "joint objective", font=font(13), fill=MUTED)

    for start, end in (
        ((clean_box[2], 292), (corrupted_box[0], 292)),
        ((corrupted_box[2], 292), (network_box[0], 292)),
        ((network_box[2], 292), (output_box[0], 292)),
    ):
        arrow(draw, start, end, paper_arrow, width=3)

    supervision_route = "#8a80b1"
    route([(222, 115), (222, 48), (1578, 48), (1578, 115)], supervision_route, width=2, dashed=True)
    supervision_font = font(15, True)
    supervision_text = "Supervision target"
    supervision_width = draw.textlength(supervision_text, font=supervision_font)
    supervision_x = 900 - supervision_width / 2
    draw.rectangle((supervision_x - 12, 34, supervision_x + supervision_width + 12, 62), fill=WHITE)
    draw.text((supervision_x, 36), supervision_text, font=supervision_font, fill=supervision_route)

    backprop_route = "#a97460"
    route([(1578, 470), (1578, 535), (1142, 535), (1142, 470)], backprop_route, width=2, dashed=True)
    backprop_font = font(15, True)
    backprop_text = "Backpropagation"
    backprop_width = draw.textlength(backprop_text, font=backprop_font)
    backprop_x = 1360 - backprop_width / 2
    draw.rectangle((backprop_x - 12, 521, backprop_x + backprop_width + 12, 549), fill=WHITE)
    draw.text((backprop_x, 523), backprop_text, font=backprop_font, fill=backprop_route)

    save(image, "03_training_workflow.png")


def _manifest_stats() -> tuple[int, dict[str, int]]:
    path = ROOT / "data/clean_layout_production_v2/accepted/manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["room_type"]] = counts.get(row["room_type"], 0) + 1
    return len(rows), counts


def figure_dataset() -> None:
    total, counts = _manifest_stats()
    image, draw = canvas("合成布局数据集：统计与样例", "512 audited clean layouts spanning four room types, HSSD assets and diverse spatial arrangements", "04")
    colors = {"bedroom": BLUE, "living_room": MAGENTA, "dining_room": AMBER, "home_office": CYAN}

    # Left column: a compact dataset description and the statistics needed to read the samples.
    draw.text((64, 174), "DATASET SUMMARY", font=font(17, True), fill=GREEN)
    draw.text((64, 205), str(total), font=font(80, True), fill=NAVY)
    draw.text((66, 296), "accepted clean scenes", font=font(24, True), fill=BLUE)

    intro = "面向家具阶段场景修复的合成 clean target corpus。每个样本包含房间边界、门窗净空、HSSD 家具语义及三维位姿。"
    intro_font = font(18)
    intro_y = 355
    for line in wrap_lines(draw, intro, intro_font, 405):
        draw.text((66, intro_y), line, font=intro_font, fill=MUTED)
        intro_y += 29

    draw.line((66, 480, 485, 480), fill=LINE, width=2)
    draw.text((66, 505), "房型分布", font=font(23, True), fill=INK)
    labels = (("卧室", "bedroom"), ("客厅", "living_room"), ("餐厅", "dining_room"), ("家庭办公室", "home_office"))
    max_count = max(counts.values())
    bar_y = 554
    for cn, key in labels:
        value = counts.get(key, 0)
        draw.text((66, bar_y), cn, font=font(16), fill=INK)
        draw.rounded_rectangle((180, bar_y + 5, 405, bar_y + 23), radius=6, fill="#e7edf4")
        draw.rounded_rectangle((180, bar_y + 5, 180 + int(225 * value / max_count), bar_y + 23), radius=6, fill=colors[key])
        draw.text((423, bar_y - 3), str(value), font=font(17, True), fill=INK)
        bar_y += 49

    draw.line((66, 763, 485, 763), fill=LINE, width=2)
    draw.text((66, 789), "随场景保存的质量证据", font=font(18, True), fill=INK)
    draw.text((66, 823), "几何硬校验 · 独立盲审 · 多样性检查", font=font(15), fill=MUTED)
    draw.text((66, 852), "Scene JSON · Review · SHA-256 manifest", font=font(15), fill=MUTED)

    # Right: four real accepted scenes spanning room type, density and room-scale diversity.
    draw.line((520, 166, 520, 1032), fill=LINE, width=2)
    draw.text((560, 174), "合成场景样例", font=font(30, True), fill=INK)
    draw.text((560, 218), "覆盖不同房型、空间尺度、家具密度与开口配置的代表性三维布局", font=font(17), fill=MUTED)
    draw.line((560, 255, 1738, 255), fill=LINE, width=2)

    scene_root = ROOT / "data/clean_layout_production_v2/accepted/scenes"
    samples = (
        ("clean_v2_prod_0002_bedroom_0003.json", "卧室", "bedroom"),
        ("clean_v2_prod_0013_living_room_0025.json", "客厅", "living_room"),
        ("clean_v2_prod_0017_dining_room_0007.json", "餐厅", "dining_room"),
        ("clean_v2_prod_0013_home_office_0007.json", "家庭办公室", "home_office"),
    )
    sample_x = (560, 1145)
    sample_y = (282, 656)
    panel_letters = "abcd"
    draw.line((1122, 270, 1122, 1020), fill="#e2e8f0", width=1)
    draw.line((550, 635, 1738, 635), fill="#e2e8f0", width=1)

    for index, (filename, room_label, room_key) in enumerate(samples):
        scene = json.loads((scene_root / filename).read_text(encoding="utf-8"))
        x = sample_x[index % 2]
        y = sample_y[index // 2]
        accent = colors[room_key]
        draw.ellipse((x, y, x + 30, y + 30), fill=accent)
        letter_font = font(15, True)
        letter = panel_letters[index]
        letter_box = draw.textbbox((0, 0), letter, font=letter_font)
        draw.text((x + 15 - (letter_box[2] - letter_box[0]) / 2, y + 15 - (letter_box[3] - letter_box[1]) / 2 - letter_box[1]), letter, font=letter_font, fill=WHITE)
        draw.text((x + 43, y - 2), room_label, font=font(23, True), fill=INK)
        meta = f"{float(scene['length']):.1f} × {float(scene['width']):.1f} m  ·  {len(scene['furniture'])} 件家具  ·  {len(scene['openings'])} 个开口"
        draw.text((x + 43, y + 30), meta, font=font(13), fill=MUTED)
        draw_3d_scene(draw, scene, (x - 8, y + 54, x + 540, y + 285), show_labels=False, scale_factor=0.82, show_contract_label=False)

    save(image, "04_dataset_overview.png")


def _object_corners(obj: dict[str, object]) -> list[tuple[float, float]]:
    x, y = float(obj["x"]), float(obj["y"])
    yaw = float(obj["yaw_rad"])
    bbox = obj["bbox"]
    width, depth = float(bbox["width"]), float(bbox["depth"])
    c, s = math.cos(yaw), math.sin(yaw)
    result = []
    for lx, ly in ((-width / 2, -depth / 2), (width / 2, -depth / 2), (width / 2, depth / 2), (-width / 2, depth / 2)):
        result.append((x + c * lx - s * ly, y + s * lx + c * ly))
    return result


def draw_scene(
    draw: ImageDraw.ImageDraw,
    scene: dict[str, object],
    box: tuple[int, int, int, int],
    *,
    corrupted: bool,
    show_labels: bool = True,
) -> None:
    x1, y1, x2, y2 = box
    length, width = float(scene["length"]), float(scene["width"])
    margin = 40
    scale = min((x2 - x1 - margin * 2) / length, (y2 - y1 - margin * 2) / width)
    room_w, room_h = length * scale, width * scale
    left = x1 + (x2 - x1 - room_w) / 2
    top = y1 + (y2 - y1 - room_h) / 2

    def px(p: tuple[float, float]) -> tuple[float, float]:
        return left + (p[0] + length / 2) * scale, top + (width / 2 - p[1]) * scale

    draw.rectangle((left, top, left + room_w, top + room_h), fill=WHITE, outline=INK, width=5)
    for opening in scene["openings"]:
        cx, cy, _ = opening["clearance_center_xyz_m"]
        sx, sy, _ = opening["clearance_size_xyz_m"]
        a = px((cx - sx / 2, cy + sy / 2))
        b = px((cx + sx / 2, cy - sy / 2))
        color = RED if opening["kind"] == "door" else BLUE
        draw.rectangle((a[0], a[1], b[0], b[1]), fill=color + "30", outline=color, width=3)
    palette = {"sleep_surface": "#5b8fd1", "seating": "#68a96b", "support_surface": "#d1a257", "storage": "#8d6cab", "lighting": "#d9cf4c"}
    for obj in scene["furniture"]:
        points = [px(p) for p in _object_corners(obj)]
        violation = corrupted and obj["object_id"] in {"coffee_table_01", "armchair_01", "cabinet_01"}
        fill = "#e65b63" if violation else palette.get(obj.get("family"), "#75859b")
        draw.polygon(points, fill=fill + "cc", outline=RED if violation else INK)
        if violation:
            draw.line(points + [points[0]], fill=RED, width=5)
        center = px((float(obj["x"]), float(obj["y"])))
        if show_labels:
            short = str(obj["category"]).replace("_", " ")
            fnt = font(14, True)
            tw = draw.textlength(short, font=fnt)
            draw.text((center[0] - tw / 2, center[1] - 9), short, font=fnt, fill=WHITE if violation else INK)


def figure_application() -> None:
    prediction_path = ROOT / "docs/prediction_examples_30_sfur_v10.json"
    prediction_payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    candidates = [
        row
        for row in prediction_payload["examples"]
        if row["scene_id"] == "clean_v2_prod_0018_home_office_0042"
        and row.get("geometry_cleared")
        and row.get("sfur_after", {}).get("scene_functional_pass")
    ]
    example = max(
        candidates,
        key=lambda row: (
            int(row["violations_before"]["total"]),
            float(row.get("sfur_after", {}).get("functional_score", 0.0)),
            -len(row["furniture"]),
        ),
    )
    accepted_scene_path = ROOT / "data/clean_layout_production_v2/accepted/scenes" / f"{example['scene_id']}.json"
    accepted_scene = json.loads(accepted_scene_path.read_text(encoding="utf-8"))
    accepted_by_object_id = {str(obj["object_id"]): obj for obj in accepted_scene["furniture"]}
    room_length = float(example["room"]["length"])
    room_width = float(example["room"]["width"])

    def object_family(category: str) -> str:
        if category in {"chair", "armchair", "sofa", "bench", "ottoman"}:
            return "seating"
        if category in {"desk", "table", "coffee_table", "side_table", "console_table"}:
            return "support_surface"
        if category in {"cabinet", "wardrobe", "bookcase", "tv_bench"}:
            return "storage"
        if "lamp" in category:
            return "lighting"
        return "other"

    def prediction_scene(poses: list[dict[str, object]]) -> dict[str, object]:
        openings: list[dict[str, object]] = []
        for index, opening in enumerate(example["openings"]):
            x, y = float(opening["x"]), float(opening["y"])
            distance_x = abs(abs(x) - room_length / 2)
            distance_y = abs(abs(y) - room_width / 2)
            wall = ("east" if x >= 0 else "west") if distance_x <= distance_y else ("north" if y >= 0 else "south")
            openings.append(
                {
                    "opening_id": str(opening.get("id", f"opening_{index:02d}")),
                    "kind": str(opening["kind"]),
                    "wall": wall,
                    "clearance_center_xyz_m": [x, y, float(opening["z"])],
                    "clearance_size_xyz_m": [float(opening["width"]), float(opening["depth"]), float(opening["height"])],
                }
            )

        furniture: list[dict[str, object]] = []
        for item, pose in zip(example["furniture"], poses):
            category = str(item["category"])
            accepted_obj = accepted_by_object_id.get(str(item["id"]), {})
            furniture.append(
                {
                    "object_id": str(item["id"]),
                    "hssd_id": str(accepted_obj.get("hssd_id", "")),
                    "hssd_source_category": str(accepted_obj.get("hssd_source_category", category)),
                    "category": category,
                    "family": object_family(category),
                    "x": float(pose["x"]),
                    "y": float(pose["y"]),
                    "yaw_rad": float(pose["yaw"]),
                    "bbox": {
                        "width": float(pose["width"]),
                        "depth": float(pose["depth"]),
                        "height": float(pose["height"]),
                    },
                }
            )
        return {"length": room_length, "width": room_width, "openings": openings, "furniture": furniture}

    disturbed = prediction_scene(example["current_poses"])
    repaired = prediction_scene(example["predicted_poses"])
    moved_ids = {
        str(item["id"])
        for item, current, predicted in zip(example["furniture"], example["current_poses"], example["predicted_poses"])
        if abs(float(predicted["x"]) - float(current["x"])) > 0.025
        or abs(float(predicted["y"]) - float(current["y"])) > 0.025
        or abs(float(predicted["yaw"]) - float(current["yaw"])) > 0.025
    }
    image, draw = canvas(
        "三维场景修复：v10真实模型推理",
        "SFUR v10 final · disturbed 3D input → verifier-confirmed repaired scene",
        "05",
    )
    # This panel intentionally removes the earlier footer and metric strip so the two 3D scenes dominate the poster space.
    draw.rectangle((0, H - 84, W, H), fill=WHITE)

    left_panel = (62, 164, 812, 1046)
    right_panel = (988, 164, 1738, 1046)
    draw_realistic_3d_scene(
        image,
        draw,
        disturbed,
        left_panel,
        title="扰动输入",
        accent=RED,
        highlight_ids=moved_ids,
    )
    draw_realistic_3d_scene(
        image,
        draw,
        repaired,
        right_panel,
        title="v10修复输出",
        accent=GREEN,
        highlight_ids=moved_ids,
    )

    arrow_y = 602
    draw.line((826, arrow_y, 974, arrow_y), fill=NAVY_2, width=8)
    draw.polygon(((974, arrow_y), (940, arrow_y - 20), (940, arrow_y + 20)), fill=NAVY_2)
    draw.text((900 - draw.textlength("v10模型修复", font=font(24, True)) / 2, arrow_y - 70), "v10模型修复", font=font(24, True), fill=NAVY_2)
    draw.text((900 - draw.textlength("4-pass SFUR · dense Δpose", font=font(16)) / 2, arrow_y + 35), "4-pass SFUR · dense Δpose", font=font(16), fill=MUTED)

    save(image, "05_application_concept_demo.png")


def main() -> None:
    figure_model()
    figure_engine_vertical_modules()
    figure_training()
    figure_dataset()
    figure_application()
    print(f"Rendered 5 poster figures to {OUT}")


if __name__ == "__main__":
    main()
