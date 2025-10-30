"""
자동 크롭 → 보간(Z→XY, ffill) → 스케일 → 스테이지 PNG + 최종 NPY 저장 (단일 파일)

- 입력: petrel_to_numpy.py에서 만든 out_petrel.npz
- 프로세스:
    1) NaN 최소 윈도우 자동 크롭(2D: j×i)
    2) 보간:
       - Z축: backward fill → forward fill
       - XY: 각 슬라이스 행마다 좌→우 forward fill (선두 유효값 전파 포함)
    3) 스케일링: minmax | standard
    4) 단계별 시각화(XY/XZ) PNG 저장
    5) 최종 결과(vol_scaled)만 .npy로 저장

Usage
-----
    python gpm_preprocessing.py ../params/params_preproc.yml

YAML (예시)
-----------
in_npz: outputs/out_petrel.npz
out_filled: outputs/filled.npy
crop_h: 50
crop_w: 36
view_z: 0
view_y: 0
scale_mode: minmax
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle


# =========================
# 보간/스케일 유틸
# =========================
def _ffill1d(arr: np.ndarray) -> np.ndarray:
    """
    1D 배열에서 NaN을 좌측(이전 값)으로 채움.

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    np.ndarray
        새 배열(원본 보존).
    """
    out = arr.copy()
    if np.isnan(out).all():
        return out
    for i in range(1, len(out)):
        if np.isnan(out[i]) and not np.isnan(out[i - 1]):
            out[i] = out[i - 1]
    return out


def _bfill1d(arr: np.ndarray) -> np.ndarray:
    """
    1D 배열에서 NaN을 우측(다음 값)으로 채움.

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    np.ndarray
        새 배열(원본 보존).
    """
    out = arr.copy()
    if np.isnan(out).all():
        return out
    for i in range(len(out) - 2, -1, -1):
        if np.isnan(out[i]) and not np.isnan(out[i + 1]):
            out[i] = out[i + 1]
    return out


def fill_z_axis_backward_then_forward(vol: np.ndarray) -> np.ndarray:
    """
    각 (j, i) 칼럼에 대해 z축으로 backward → forward 전파.

    Parameters
    ----------
    vol : np.ndarray
        (Z,Y,X) 또는 (C,Z,Y,X)

    Returns
    -------
    np.ndarray
        동일 shape.
    """
    vol = np.asarray(vol, float)
    if vol.ndim == 3:
        Z, Y, X = vol.shape
        out = vol.copy()
        for j in range(Y):
            for i in range(X):
                col = out[:, j, i]
                col = _bfill1d(col)
                col = _ffill1d(col)
                out[:, j, i] = col
        return out
    elif vol.ndim == 4:
        out = vol.copy()
        for c in range(vol.shape[0]):
            out[c] = fill_z_axis_backward_then_forward(vol[c])
        return out
    else:
        raise ValueError("Expected 3D or 4D volume.")


def fill_xy_per_slice_ffill_x0(vol: np.ndarray) -> np.ndarray:
    """
    각 z-slice의 각 행(j)에 대해 좌→우 ffill 수행.
    선두 유효값이 있으면 선두까지 값 전파 후 ffill.

    Parameters
    ----------
    vol : np.ndarray
        (Z,Y,X) 또는 (C,Z,Y,X)

    Returns
    -------
    np.ndarray
        동일 shape.
    """
    vol = np.asarray(vol, float)
    if vol.ndim == 3:
        Z, Y, X = vol.shape
        out = vol.copy()
        for k in range(Z):
            for j in range(Y):
                row = out[k, j, :]
                valid = np.where(~np.isnan(row))[0]
                if valid.size > 0:
                    first = valid[0]
                    if first > 0:
                        row[:first] = row[first]
                    row[:] = _ffill1d(row)
        return out
    elif vol.ndim == 4:
        out = vol.copy()
        for c in range(vol.shape[0]):
            out[c] = fill_xy_per_slice_ffill_x0(vol[c])
        return out
    else:
        raise ValueError("Expected 3D or 4D volume.")


def scale_volume(vol: np.ndarray, mode: str = "minmax") -> np.ndarray:
    """
    볼륨 스케일링.

    Parameters
    ----------
    vol : np.ndarray
    mode : {"minmax","standard"}

    Returns
    -------
    np.ndarray
        스케일된 볼륨.
    """
    v = vol.astype(float)
    if mode == "minmax":
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin < 1e-12):
            return np.zeros_like(v)
        return (v - vmin) / (vmax - vmin)
    elif mode == "standard":
        mean, std = np.nanmean(v), np.nanstd(v)
        if not np.isfinite(std) or std < 1e-12:
            return np.zeros_like(v)
        return (v - mean) / std
    else:
        raise ValueError("mode must be 'minmax' or 'standard'")


# =========================
# 크롭(최소 NaN 창)
# =========================
def _window_sum_2d(int_img: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    적분영상(int_img)에서 h×w 윈도우 합계를 O(1)로 계산.

    Returns
    -------
    np.ndarray
        (ny-h+1, nx-w+1)
    """
    S = int_img
    return S[h:, w:] - S[:-h, w:] - S[h:, :-w] + S[:-h, :-w]


def auto_crop_min_nan(
    total_ratio: np.ndarray,
    crop_h: int,
    crop_w: int,
    visualize: bool = False,
    crop_png_path: Optional[str] = None,
) -> Dict[str, int | float | np.ndarray]:
    """
    3D 배열 total_ratio (nz, ny, nx)에서 NaN이 가장 적은 (crop_h, crop_w) 2D 영역(j,i) 탐색.

    Parameters
    ----------
    total_ratio : np.ndarray
        (nz, ny, nx)
    crop_h : int
        창 높이 (j 방향)
    crop_w : int
        창 너비 (i 방향)
    visualize : bool
        True일 경우 탐색 결과를 PNG로 저장
    crop_png_path : str | None
        시각화 결과 저장 경로. None이면 저장 안 함.

    Returns
    -------
    dict
        {"i_min","i_max","j_min","j_max","nan_count","nan_ratio","nan_count_map"}
    """
    if total_ratio.ndim != 3:
        raise ValueError("total_ratio는 (nz, ny, nx) 3D 배열이어야 합니다.")
    nz, ny, nx = total_ratio.shape
    if crop_h <= 0 or crop_w <= 0 or crop_h > ny or crop_w > nx:
        raise ValueError(f"crop 크기 오류: crop_h={crop_h}, crop_w={crop_w}, ny={ny}, nx={nx}")

    nan_count_map = np.sum(np.isnan(total_ratio), axis=0)  # (ny, nx)
    int_img = np.pad(nan_count_map, ((1, 0), (1, 0)), mode="constant", constant_values=0).cumsum(0).cumsum(1)
    win_sums = _window_sum_2d(int_img, crop_h, crop_w)
    idx = np.argmin(win_sums)
    best_j, best_i = np.unravel_index(idx, win_sums.shape)
    min_nan = int(win_sums[best_j, best_i])
    j_min, j_max = int(best_j), int(best_j + crop_h - 1)
    i_min, i_max = int(best_i), int(best_i + crop_w - 1)
    nan_ratio = min_nan / (crop_h * crop_w * nz)

    if visualize and crop_png_path:
        Path(crop_png_path).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        valid_mask_any = np.any(~np.isnan(total_ratio), axis=0)
        ax.imshow(valid_mask_any, cmap="Greys", origin="upper")
        ax.set_title("Valid Cells with Auto Crop Box")
        ax.set_xlabel("i index")
        ax.set_ylabel("j index")
        rect = Rectangle((i_min, j_min), crop_w, crop_h, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        fig.savefig(crop_png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return {
        "i_min": i_min,
        "i_max": i_max,
        "j_min": j_min,
        "j_max": j_max,
        "nan_count": min_nan,
        "nan_ratio": nan_ratio,
        "nan_count_map": nan_count_map,
    }


# =========================
# 스테이지 플롯
# =========================
def _plot_stage(
    vol_orig: np.ndarray,
    vol_z: np.ndarray,
    vol_xy: np.ndarray,
    idx: int,
    mode: str = "xy",
    figsize: Tuple[float, float] = (9, 9),
    prefix: str | Path = "outputs/stages",
) -> None:
    """
    단계별(Original / Z-fill / XY-fill) 패널 저장 (Sand/Silt/Clay 3행 × 3열).

    Parameters
    ----------
    vol_orig, vol_z, vol_xy : np.ndarray
        (C=3, Z, Y, X) 볼륨
    idx : int
        mode='xy' → z index, mode='xz' → y index
    mode : {"xy","xz"}
        xy: XY 슬라이스 / xz: XZ 단면
    figsize : tuple[float, float]
        matplotlib figure size
    prefix : str | Path
        저장 파일 접두사 (파일명은 {prefix}_{mode}.png)
    """
    names = ["Sand", "Silt", "Clay"]
    cmaps = ["YlOrBr", "cividis", "Oranges"]
    stages = [("Original", vol_orig), ("Z-fill", vol_z), ("XY-fill", vol_xy)]

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=figsize, constrained_layout=True)
    for r in range(3):
        for c, (title, V) in enumerate(stages):
            ax = axes[r, c]
            if mode == "xy":
                im = ax.imshow(np.ma.masked_invalid(V[r, idx]), cmap=cmaps[r], origin="upper")
                ax.set_xlabel("x (i)")
                ax.set_ylabel("y (j)")
            else:
                im = ax.imshow(
                    np.ma.masked_invalid(V[r, :, idx, :]),
                    cmap=cmaps[r],
                    origin="upper",
                    aspect="auto",
                )
                ax.set_xlabel("x (i)")
                ax.set_ylabel("z (k)")
            if r == 0:
                ax.set_title(title)
        fig.colorbar(im, ax=axes[r, :].ravel().tolist(), fraction=0.03, pad=0.02)
        axes[r, 0].set_ylabel(names[r])

    out_png = prefix.with_name(prefix.name + f"_{mode}.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================
# 메인 파이프라인
# =========================
def preprocess_and_fill(
    out: Dict[str, Dict[str, np.ndarray]],
    crop_h: int = 50,
    crop_w: int = 36,
    view_z: int = 0,
    view_y: int = 0,
    scale_mode: str = "minmax",
    save_path: str = "outputs/filled.npy",
) -> np.ndarray:
    """
    전체 전처리 파이프라인을 실행하고 최종 결과(.npy)를 저장.

    Parameters
    ----------
    out : dict
        petrel_to_numpy.py 결과 딕셔너리 구조.
    crop_h, crop_w : int
        자동 크롭 창 크기(j×i).
    view_z : int
        XY 시각화용 z index.
    view_y : int
        XZ 시각화용 y index.
    scale_mode : {"minmax","standard"}
    save_path : str
        결과 .npy 저장 경로.

    Returns
    -------
    np.ndarray
        최종 스케일된 볼륨 (C=3, Z, Y, X).
    """
    save_path = str(save_path)
    base_dir = Path(save_path).parent
    base_dir.mkdir(parents=True, exist_ok=True)

    Silt = out["props"]["Silt"]
    Clay = out["props"]["Clay"]
    Sand = out["props"]["Sand"]

    # 1) 크롭
    total = Silt + Clay + Sand
    crop_png = str(base_dir / "crop_region.png")
    res = auto_crop_min_nan(total, crop_h=crop_h, crop_w=crop_w, visualize=True, crop_png_path=crop_png)

    j0, j1, i0, i1 = res["j_min"], res["j_max"], res["i_min"], res["i_max"]
    Silt_c = Silt[:, j0 : j1 + 1, i0 : i1 + 1]
    Clay_c = Clay[:, j0 : j1 + 1, i0 : i1 + 1]
    Sand_c = Sand[:, j0 : j1 + 1, i0 : i1 + 1]

    vol = np.stack([Sand_c, Silt_c, Clay_c], axis=0).astype(float)  # (C=3,Z,Y,X)

    # 2) 보간
    vol_z = fill_z_axis_backward_then_forward(vol)
    vol_xy = fill_xy_per_slice_ffill_x0(vol_z)
    vol_scaled = scale_volume(vol_xy, scale_mode)

    # 3) 스테이지 플롯
    _plot_stage(vol, vol_z, vol_xy, view_z, mode="xy", prefix=base_dir / "stages_z")
    _plot_stage(vol, vol_z, vol_xy, view_y, mode="xz", prefix=base_dir / "stages_y")

    # 4) 저장
    np.save(save_path, vol_scaled)
    print(f"[Saved] 보간+스케일 결과: {save_path} | shape={vol_scaled.shape}")
    return vol_scaled


def load_out_from_npz(npz_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    petrel_to_numpy.py가 생성한 .npz를 로드하여 원래의 딕셔너리 구조로 복원.

    Parameters
    ----------
    npz_path : str

    Returns
    -------
    dict
        {"meta","ijk","xyz","props"}
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
def main(config_path: str) -> None:
    """
    YAML 설정을 읽어 전처리 파이프라인을 실행.

    Parameters
    ----------
    config_path : str
        YAML 경로.
    """
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    npz_in: str = cfg["in_npz"]
    save_npy: str = cfg.get("out_filled", "outputs/filled.npy")
    crop_h: int = int(cfg.get("crop_h", 50))
    crop_w: int = int(cfg.get("crop_w", 36))
    view_z: int = int(cfg.get("view_z", 0))
    view_y: int = int(cfg.get("view_y", 0))
    scale_mode: str = cfg.get("scale_mode", "minmax")

    out = load_out_from_npz(npz_in)
    _ = preprocess_and_fill(
        out,
        crop_h=crop_h,
        crop_w=crop_w,
        view_z=view_z,
        view_y=view_y,
        scale_mode=scale_mode,
        save_path=save_npy,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gpm_preprocessing.py ../params/params_preproc.yml")
        sys.exit(1)
    main(sys.argv[1])
