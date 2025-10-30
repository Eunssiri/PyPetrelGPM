#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
시각화 실행 스크립트 (서브커맨드: sections | geometry3d) — 단일 파일

- sections : (sand/silt/clay) × (Z-mean XY, X-slice YZ, Y-slice XZ, Z-slice XY) 3×4 패널 저장
- geometry3d : sand/silt/clay 각각 3D 산점도 저장 (Plotly + kaleido 필요)

Usage
-----
    python gpm_viz.py sections ../params/params_viz_sections.yml
    python gpm_viz.py geometry3d  ../params/params_viz_geometry3d.yml

YAML (예시)
-----------
# sections
in_npz: outputs/out_petrel.npz
out_png: outputs/figures_v3/sections_original.png
x_slice: null
y_slice: null
z_slice: null

# geometry3d
in_npz: outputs/out_petrel.npz
outdir: outputs/figures_v3
case_name: original
cmaps: {silt: cividis, clay: Oranges, sand: inferno_r}
width: 1000
height: 800
scale: 2
marker_size: 4
show_colorbar: false
cmin: 0.0
cmax: 1.0
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import yaml


# =========================
# 공통 유틸
# =========================
def _find_key(props: Dict[str, np.ndarray], target: str) -> str:
    """
    props 딕셔너리에서 target 키(Sand/Silt/Clay)를 유연하게 탐색하여 반환.
    """
    if target in props:
        return target
    tl = target.lower()
    for k in props.keys():
        kl = k.lower()
        if kl == tl or kl.endswith("_" + tl) or kl.endswith(tl):
            return k
    for k in props.keys():
        if tl in k.lower():
            return k
    raise KeyError(f"'{target}' key not found in props: {list(props.keys())[:8]} ...")


def get_xyz_props(out_dict: Dict[str, Dict[str, np.ndarray]]) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
    """
    out 딕셔너리에서 xyz 좌표와 sand/silt/clay 속성 3D 배열을 꺼내 반환.
    """
    x = out_dict["xyz"]["x"]
    y = out_dict["xyz"]["y"]
    z = out_dict["xyz"]["z"]
    props = out_dict["props"]
    k_silt = _find_key(props, "Silt")
    k_clay = _find_key(props, "Clay")
    k_sand = _find_key(props, "Sand")
    return (x, y, z), {"sand": props[k_sand], "silt": props[k_silt], "clay": props[k_clay]}


def _cmap_nan_white(name: str):
    """
    matplotlib colormap의 NaN 색상을 흰색으로 설정하여 반환.
    """
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad((1, 1, 1, 1))
    return cmap


def _is_rectilinear(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> bool:
    """
    X/Y/Z 좌표 그리드가 직교(rectilinear)인지 판별.
    """
    try:
        okx = np.allclose(X[0, 0, :][None, None, :], X, equal_nan=True)
        oky = np.allclose(Y[0, :, 0][None, :, None], Y, equal_nan=True)
        okz = np.allclose(Z[:, 0, 0][:, None, None], Z, equal_nan=True)
        return bool(okx and oky and okz)
    except Exception:
        return False


def _edges1d(a: np.ndarray) -> np.ndarray:
    """
    셀 중심 좌표 1D → 코너 좌표 1D (pcolormesh용).
    """
    a = np.asarray(a, float)
    if a.size == 1:
        d = 1.0
        return np.array([a[0] - d / 2, a[0] + d / 2])
    mid = (a[:-1] + a[1:]) / 2.0
    first = a[0] - (mid[0] - a[0])
    last = a[-1] + (a[-1] - mid[-1])
    return np.r_[first, mid, last]


def _edges2d(cx: np.ndarray, cy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    셀 중심 좌표 2D → 코너 좌표 2D (pcolormesh용).

    Returns
    -------
    (Cx, Cy) : (M+1, N+1)
    """
    M, N = cx.shape
    cx_e_rows = np.vstack([_edges1d(cx[i, :]) for i in range(M)])
    cy_e_rows = np.vstack([_edges1d(cy[i, :]) for i in range(M)])
    Cx = np.column_stack([_edges1d(cx_e_rows[:, j]) for j in range(N + 1)])
    Cy = np.column_stack([_edges1d(cy_e_rows[:, j]) for j in range(N + 1)])
    return Cx, Cy


def nanmean_safe(a: np.ndarray, axis: Optional[int | Tuple[int, ...]] = None, keepdims: bool = False, fill_value: float = np.nan) -> np.ndarray:
    """
    np.nanmean 대체(빈 슬라이스 안전 처리).
    """
    a = np.asarray(a, float)
    mask = np.isfinite(a)
    total = np.where(mask, a, 0.0).sum(axis=axis, keepdims=keepdims)
    count = mask.sum(axis=axis, keepdims=keepdims)
    out = np.divide(total, count, out=np.full_like(total, fill_value, dtype=float), where=(count > 0))
    return out


# =========================
# 3D 산점도
# =========================
def save_prop_geometry3d(
    out: Dict[str, Dict[str, np.ndarray]],
    outdir: str | Path,
    case_name: str = "original",
    cmaps: Optional[Dict[str, str]] = None,
    width: int = 1000,
    height: int = 800,
    scale: int = 2,
    marker_size: int = 4,
    show_colorbar: bool = False,
    cmin: float = 0.0,
    cmax: float = 1.0,
) -> None:
    """
    sand/silt/clay 각각 3D 산점도 PNG 저장 (Plotly + kaleido 필요).
    """
    if cmaps is None:
        cmaps = {"sand": "inferno_r", "silt": "cividis", "clay": "Oranges"}

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (x, y, z), prop_dict = get_xyz_props(out)

    for prop_name in ["sand", "silt", "clay"]:
        scalar = prop_dict[prop_name]
        mask = np.isfinite(scalar)
        x_flat = x[mask]
        y_flat = y[mask]
        z_flat = z[mask]
        s_flat = scalar[mask]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x_flat,
                    y=y_flat,
                    z=z_flat,
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        color=s_flat,
                        colorscale=cmaps[prop_name],
                        showscale=show_colorbar,
                        opacity=0.9,
                        cmin=cmin,
                        cmax=cmax,
                    ),
                )
            ]
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=True, backgroundcolor="lightgray", gridcolor="white", zerolinecolor="black", showticklabels=True, tickformat=".0f"),
                yaxis=dict(showbackground=True, backgroundcolor="lightgray", gridcolor="white", zerolinecolor="black", showticklabels=True, tickformat=".0f"),
                zaxis=dict(showbackground=True, backgroundcolor="lightgray", gridcolor="white", zerolinecolor="black", showticklabels=True, tickformat=".0f"),
                bgcolor="rgba(0,0,0,0)",
                aspectmode="manual",
                aspectratio=dict(x=2.0, y=1.0, z=1.0),
            ),
            scene_camera=dict(eye=dict(x=1.5, y=-1.5, z=1.5)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        out_png = outdir / f"{case_name}_{prop_name}.png"
        pio.write_image(fig, str(out_png), format="png", width=width, height=height, scale=scale)
        print(f"[SAVE] {out_png}")


# =========================
# 3×4 단면 패널
# =========================
def save_gpm_sections_panel(
    out: Dict[str, Dict[str, np.ndarray]],
    x_slice: Optional[int] = None,
    y_slice: Optional[int] = None,
    z_slice: Optional[int] = None,
    out_png: str | Path = "outputs/figures_v3/sections.png",
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Sand / Silt / Clay — mean, X-slice, Y-slice, Z-slice (physical coords)",
) -> None:
    """
    (sand/silt/clay) × (Z-mean XY, X-slice YZ, Y-slice XZ, Z-slice XY) 3×4 패널 저장.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    props = out["props"]
    X = out["xyz"]["x"]
    Y = out["xyz"]["y"]
    Z = out["xyz"]["z"]

    SAND = props[_find_key(props, "Sand")]
    SILT = props[_find_key(props, "Silt")]
    CLAY = props[_find_key(props, "Clay")]

    Zn, Yn, Xn = SAND.shape
    if x_slice is None:
        x_slice = Xn // 2
    if y_slice is None:
        y_slice = Yn // 2
    if z_slice is None:
        z_slice = Zn // 2

    rect = _is_rectilinear(X, Y, Z)
    chans = [("sand", SAND, "inferno_r"), ("silt", SILT, "cividis"), ("clay", CLAY, "Oranges")]

    if figsize is None:
        xrange = float(np.nanmax(X) - np.nanmin(X))
        yrange = float(np.nanmax(Y) - np.nanmin(Y))
        zrange = float(np.nanmax(Z) - np.nanmin(Z))
        scale = 5.0 / max(1e-9 + max(xrange, yrange, zrange), 1.0)
        fig_w = 4 * max(xrange, 1.0) * scale
        fig_h = 3 * max(yrange, 1.0) * scale
        figsize = (fig_w, fig_h)

    fig, axes = plt.subplots(3, 4, figsize=figsize)
    titles = ["Z-mean (XY)", f"X-slice @ x={x_slice} (YZ)", f"Y-slice @ y={y_slice} (XZ)", f"Z-slice @ z={z_slice} (XY)"]

    for r, (name, VOL, cmap_name) in enumerate(chans):
        cmap = _cmap_nan_white(cmap_name)
        vmin, vmax = float(np.nanmin(VOL)), float(np.nanmax(VOL))

        # 1) Z-mean → XY
        ax = axes[r, 0]
        img = nanmean_safe(VOL, axis=0)
        if rect:
            x_axis = X[0, 0, :]
            y_axis = Y[0, :, 0]
            ext = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
            im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, aspect="auto")
        else:
            z_ref = z_slice
            Xc, Yc = X[z_ref, :, :], Y[z_ref, :, :]
            Xv, Yv = _edges2d(Xc, Yc)
            im = ax.pcolormesh(Xv, Yv, img, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        if r == 0:
            ax.set_title(titles[0])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)

        # 2) X-slice → YZ
        ax = axes[r, 1]
        img = VOL[:, :, x_slice]
        if rect:
            y_axis = Y[0, :, 0]
            z_axis = Z[:, 0, 0]
            ext = [y_axis.min(), y_axis.max(), z_axis.max(), z_axis.min()]
            im = ax.imshow(img, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, aspect="auto")
        else:
            Yc, Zc = Y[:, :, x_slice], Z[:, :, x_slice]
            Yv, Zv = _edges2d(Yc, Zc)
            im = ax.pcolormesh(Yv, Zv, img, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        if r == 0:
            ax.set_title(titles[1])
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)

        # 3) Y-slice → XZ
        ax = axes[r, 2]
        img = VOL[:, y_slice, :]
        if rect:
            x_axis = X[0, 0, :]
            z_axis = Z[:, 0, 0]
            ext = [x_axis.min(), x_axis.max(), z_axis.max(), z_axis.min()]
            im = ax.imshow(img, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, aspect="auto")
        else:
            Xc, Zc = X[:, y_slice, :], Z[:, y_slice, :]
            Xv, Zv = _edges2d(Xc, Zc)
            im = ax.pcolormesh(Xv, Zv, img, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        if r == 0:
            ax.set_title(titles[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)

        # 4) Z-slice → XY
        ax = axes[r, 3]
        img = VOL[z_slice, :, :]
        if rect:
            x_axis = X[0, 0, :]
            y_axis = Y[0, :, 0]
            ext = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
            im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, aspect="auto")
        else:
            Xc, Yc = X[z_slice, :, :], Y[z_slice, :, :]
            Xv, Yv = _edges2d(Xc, Yc)
            im = ax.pcolormesh(Xv, Yv, img, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        if r == 0:
            ax.set_title(titles[3])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_png}")


# =========================
# NPZ 로더
# =========================
def load_out_from_npz(npz_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    petrel_to_numpy.py가 생성한 .npz를 로드하여 원래의 딕셔너리 구조로 복원.
    """
    d = np.load(npz_path, allow_pickle=True)
    props = {k.split("prop::", 1)[1]: d[k] for k in d.files if k.startswith("prop::")}
    return {
        "meta": {
            "names": list(d["meta_names"]),
            "nx": int(d["meta_nx"]),
            "ny": int(d["meta_ny"]),
            "nz": int(d["meta_nz"]),
        },
        "ijk": {"i": d["ijk_i"], "j": d["ijk_j"], "k": d["ijk_k"]},
        "xyz": {"x": d["xyz_x"], "y": d["xyz_y"], "z": d["xyz_z"]},
        "props": props,
    }


# =========================
# CLI
# =========================
def main() -> None:
    """
    서브커맨드(sections | geometry3d)와 YAML 설정을 받아 시각화 실행.
    """
    if len(sys.argv) != 3 or sys.argv[1] not in ("sections", "geometry3d"):
        print("Usage: python gpm_viz.py [sections|geometry3d] ../params/params_viz_*.yml")
        sys.exit(1)

    mode, cfg_path = sys.argv[1], sys.argv[2]
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out = load_out_from_npz(cfg["in_npz"])

    if mode == "sections":
        save_gpm_sections_panel(
            out,
            x_slice=cfg.get("x_slice"),
            y_slice=cfg.get("y_slice"),
            z_slice=cfg.get("z_slice"),
            out_png=cfg.get("out_png", "outputs/figures_v3/sections.png"),
        )
    else:
        save_prop_geometry3d(
            out,
            outdir=cfg.get("outdir", "outputs/figures_v3"),
            case_name=cfg.get("case_name", "original"),
            cmaps=cfg.get("cmaps", {"sand": "inferno_r", "silt": "cividis", "clay": "Oranges"}),
            width=int(cfg.get("width", 1000)),
            height=int(cfg.get("height", 800)),
            scale=int(cfg.get("scale", 2)),
            marker_size=int(cfg.get("marker_size", 4)),
            show_colorbar=bool(cfg.get("show_colorbar", False)),
            cmin=float(cfg.get("cmin", 0.0)),
            cmax=float(cfg.get("cmax", 1.0)),
        )


if __name__ == "__main__":
    main()
