"""
Petrel 텍스트 → NumPy 3D 그리드(npz) 변환 (단일 파일 버전)

- Petrel에서 내보낸 텍스트 파일을 파싱하여 (nz, ny, nx) 3D 격자 딕셔너리로 변환
- 좌표(x, y, z) 및 IJK 인덱스, 속성들을 모두 3D 배열로 저장
- 결과는 .npz로 압축 저장
- 필요한 경우 Sand(coarse)+Sand(fine) 합산하여 "Sand" 생성
- Z 축 과장(z_exaggeration) 옵션 제공

Usage
-----
    python petrel_to_numpy.py ../params/params_petrel.yml

YAML (예시)
-----------
petrel_path: Petrel_Export.txt
out_npz: outputs/out_petrel.npz
nan_sentinels: [-99, -999]
z_exaggeration: null
make_sand_sum: true
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


# =========================
# 헤더/본문 파싱 유틸들
# =========================
def parse_petrel_header(path: str) -> Tuple[int, List[str]]:
    """
    Petrel 헤더를 파싱하여 데이터 시작 라인(skiprows)과 원본 컬럼명 리스트를 반환.

    Parameters
    ----------
    path : str
        Petrel 텍스트 파일 경로.

    Returns
    -------
    skiprows : int
        데이터 본문이 시작되는 라인 번호.
    raw_names : list[str]
        헤더에 정의된 원본 컬럼명 리스트(속성명 포함).

    Raises
    ------
    ValueError
        파일이 비어있거나, PETREL 헤더/컬럼 수/컬럼명이 유효하지 않을 때.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError("빈 파일입니다.")

    petrel_idx = None
    for i, L in enumerate(lines[:200]):
        if L.strip().upper().startswith("PETREL:"):
            petrel_idx = i
            break
    if petrel_idx is None:
        raise ValueError("PETREL 헤더를 찾을 수 없습니다.")

    j = petrel_idx + 1
    num_cols: Optional[int] = None
    while j < len(lines):
        s = lines[j].strip()
        if s:
            if re.fullmatch(r"[0-9]+", s):
                num_cols = int(s)
                break
            raise ValueError(f"컬럼 수 정수 라인을 기대했지만: '{s}'")
        j += 1
    if num_cols is None:
        raise ValueError("컬럼 수(num_cols)를 찾지 못했습니다.")

    name_lines_start = j + 1
    name_lines_end = name_lines_start + num_cols
    if name_lines_end > len(lines):
        raise ValueError("컬럼명 라인 수가 파일 길이를 초과합니다.")

    raw_names: List[str] = []
    for L in lines[name_lines_start:name_lines_end]:
        parts = L.strip().split()
        if not parts:
            raise ValueError("비어있는 컬럼 정의 라인 발견.")
        raw_names.append(parts[0])

    skiprows = name_lines_end
    return skiprows, raw_names


def normalize_names(raw_names: List[str]) -> List[str]:
    """
    원본 컬럼명을 표준 키(i, j, k, x, y, z)로 정규화.
    그 외 속성명은 원형 유지.

    Parameters
    ----------
    raw_names : list[str]
        헤더에서 읽은 원본 컬럼명.

    Returns
    -------
    list[str]
        정규화된 컬럼명 리스트.
    """
    out: List[str] = []
    for n in raw_names:
        base = n.strip()
        key = base.lower().replace(" ", "_")
        if key in ("i_index", "i"):
            out.append("i")
        elif key in ("j_index", "j"):
            out.append("j")
        elif key in ("k_index", "k"):
            out.append("k")
        elif key in ("x_coord", "x"):
            out.append("x")
        elif key in ("y_coord", "y"):
            out.append("y")
        elif key in ("z_coord", "z"):
            out.append("z")
        else:
            out.append(base)
    return out


def load_petrel_body_as_numpy(path: str, skiprows: int, ncols: int) -> np.ndarray:
    """
    본문 데이터를 공백 구분으로 읽어 NxC 2D 배열을 반환.

    Parameters
    ----------
    path : str
        파일 경로.
    skiprows : int
        본문 시작 라인 번호.
    ncols : int
        사용할 컬럼 수.

    Returns
    -------
    np.ndarray
        (N, C) float 배열.

    Raises
    ------
    ValueError
        읽힌 열 개수가 기대와 다를 때.
    """
    data = np.loadtxt(
        path,
        comments=None,
        skiprows=skiprows,
        dtype=float,
        ndmin=2,
        usecols=tuple(range(ncols)),
    )
    if data.shape[1] != ncols:
        raise ValueError(f"열 개수 불일치: 기대 {ncols}, 실제 {data.shape[1]}")
    return data


def replace_nan_sentinels(arr: np.ndarray, nan_sentinels: Optional[List[float]]) -> None:
    """
    -99, -999 같은 센티넬 값을 np.nan으로 치환 (제자리 연산).

    Parameters
    ----------
    arr : np.ndarray
        대상 배열.
    nan_sentinels : list[float] | None
        치환할 센티넬 값 목록. None/빈 리스트면 무시.
    """
    if not nan_sentinels:
        return
    for v in nan_sentinels:
        arr[arr == v] = np.nan


def infer_grid_shape_from_ijk(i_col: np.ndarray, j_col: np.ndarray, k_col: np.ndarray) -> Tuple[int, int, int]:
    """
    1-based I/J/K 인덱스에서 (nx, ny, nz) 추정.

    Returns
    -------
    (nx, ny, nz) : tuple[int, int, int]
    """
    nx = int(np.nanmax(i_col))
    ny = int(np.nanmax(j_col))
    nz = int(np.nanmax(k_col))
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"그리드 크기 추정 실패: nx={nx}, ny={ny}, nz={nz}")
    return nx, ny, nz


def scatter_to_grid(
    values_1d: np.ndarray,
    i_col: np.ndarray,
    j_col: np.ndarray,
    k_col: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> np.ndarray:
    """
    (N,) 산포 데이터를 (nz, ny, nx) 그리드로 뿌리기.

    - 중복 인덱스가 있을 경우 마지막 값으로 덮어씀.

    Returns
    -------
    np.ndarray
        (nz, ny, nx) float 배열.
    """
    grid = np.full((nz, ny, nx), np.nan, dtype=float)
    mask = ~np.isnan(i_col) & ~np.isnan(j_col) & ~np.isnan(k_col) & ~np.isnan(values_1d)
    i0 = (i_col[mask].astype(int) - 1)
    j0 = (j_col[mask].astype(int) - 1)
    k0 = (k_col[mask].astype(int) - 1)
    val = values_1d[mask].astype(float)
    ok = (i0 >= 0) & (i0 < nx) & (j0 >= 0) & (j0 < ny) & (k0 >= 0) & (k0 < nz)
    i0, j0, k0, val = i0[ok], j0[ok], k0[ok], val[ok]
    grid[k0, j0, i0] = val
    return grid


def petrel_to_numpy_dict(
    path: str,
    nan_sentinels: Optional[List[float]] = None,
    z_exaggeration: Optional[float] = None,
    make_sand_sum: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Petrel 텍스트 파일 → 넘파이 3D 그리드 딕셔너리.

    Returns
    -------
    dict
        {
          "meta": {"names": list[str], "nx": int, "ny": int, "nz": int},
          "ijk":  {"i": (nz,ny,nx), "j": ..., "k": ...},
          "xyz":  {"x": (nz,ny,nx), "y": ..., "z": ...},
          "props":{prop_name: (nz,ny,nx), ...}
        }
    """
    skiprows, raw_names = parse_petrel_header(path)
    names = normalize_names(raw_names)
    ncols = len(names)

    body = load_petrel_body_as_numpy(path, skiprows, ncols)
    replace_nan_sentinels(body, nan_sentinels)

    try:
        i_idx, j_idx, k_idx = names.index("i"), names.index("j"), names.index("k")
        x_idx, y_idx, z_idx = names.index("x"), names.index("y"), names.index("z")
    except ValueError as e:
        raise ValueError(f"필수 컬럼(i,j,k,x,y,z) 확인 필요: {e}")

    i_col, j_col, k_col = body[:, i_idx], body[:, j_idx], body[:, k_idx]
    nx, ny, nz = infer_grid_shape_from_ijk(i_col, j_col, k_col)

    x_grid = scatter_to_grid(body[:, x_idx], i_col, j_col, k_col, nx, ny, nz)
    y_grid = scatter_to_grid(body[:, y_idx], i_col, j_col, k_col, nx, ny, nz)
    z_grid = scatter_to_grid(body[:, z_idx], i_col, j_col, k_col, nx, ny, nz)
    if z_exaggeration is not None:
        z_grid = z_grid * float(z_exaggeration)

    i_grid = scatter_to_grid(i_col, i_col, j_col, k_col, nx, ny, nz)
    j_grid = scatter_to_grid(j_col, i_col, j_col, k_col, nx, ny, nz)
    k_grid = scatter_to_grid(k_col, i_col, j_col, k_col, nx, ny, nz)

    props: Dict[str, np.ndarray] = {}
    for nm in names:
        if nm in ("i", "j", "k", "x", "y", "z"):
            continue
        col = body[:, names.index(nm)]
        props[nm] = scatter_to_grid(col, i_col, j_col, k_col, nx, ny, nz)

    if make_sand_sum:
        def find_key(cands: List[str]) -> Optional[str]:
            for c in cands:
                if c in props:
                    return c
            return None

        coarse = find_key(["Sand(coarse)", "Sand_coarse", "sand_coarse", "coarse_sand"])
        fine = find_key(["Sand(fine)", "Sand_fine", "sand_fine", "fine_sand"])
        if coarse and fine:
            props["Sand"] = props[coarse].astype(float) + props[fine].astype(float)

    return {
        "meta": {"names": names, "nx": nx, "ny": ny, "nz": nz},
        "ijk": {"i": i_grid, "j": j_grid, "k": k_grid},
        "xyz": {"x": x_grid, "y": y_grid, "z": z_grid},
        "props": props,
    }


# =========================
# CLI
# =========================
def main(config_path: str) -> None:
    """
    YAML 설정을 읽어 Petrel 텍스트를 파싱하고 .npz로 저장.

    Parameters
    ----------
    config_path : str
        YAML 경로.
    """
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    path_txt: str = cfg["petrel_path"]
    out_npz: str = cfg.get("out_npz", "outputs/out_petrel.npz")
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)

    nan_sentinels: List[float] = cfg.get("nan_sentinels", [])
    z_exaggeration: Optional[float] = cfg.get("z_exaggeration", None)
    make_sand_sum: bool = cfg.get("make_sand_sum", True)

    out = petrel_to_numpy_dict(
        path=path_txt,
        nan_sentinels=nan_sentinels,
        z_exaggeration=z_exaggeration,
        make_sand_sum=make_sand_sum,
    )

    np.savez_compressed(
        out_npz,
        **{
            "meta_names": np.array(out["meta"]["names"], dtype=object),
            "meta_nx": out["meta"]["nx"],
            "meta_ny": out["meta"]["ny"],
            "meta_nz": out["meta"]["nz"],
            "ijk_i": out["ijk"]["i"],
            "ijk_j": out["ijk"]["j"],
            "ijk_k": out["ijk"]["k"],
            "xyz_x": out["xyz"]["x"],
            "xyz_y": out["xyz"]["y"],
            "xyz_z": out["xyz"]["z"],
            **{f"prop::{k}": v for k, v in out["props"].items()},
        },
    )
    print(f"[SAVE] {out_npz}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python petrel_to_numpy.py ../params/params_petrel.yml")
        sys.exit(1)
    main(sys.argv[1])
