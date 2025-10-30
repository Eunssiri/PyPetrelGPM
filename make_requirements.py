# tools/make_requirements.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
프로젝트 소스(.py)에서 import된 외부 패키지를 수집해
현재 환경에 설치된 버전을 기준으로 requirements.txt를 생성합니다.

사용:
    python make_requirements.py software params -o requirements.txt
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Iterable, Set, Dict, List, Optional

try:
    # Python 3.8+
    from importlib import metadata as importlib_metadata
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore


# 표준 라이브러리/내부 모듈(필요시 추가)
STDLIKE: Set[str] = set(
    """
    sys os re math json pathlib typing typing_extensions dataclasses enum functools itertools collections
    statistics time datetime logging subprocess shutil argparse textwrap traceback importlib contextlib
    tempfile io gzip zipfile socket base64 hashlib uuid copy bisect operator pprint weakref
    warnings types gc inspect concurrent multiprocessing asyncio threading queue signal
    """.split()
)

# 흔한 import명 ↔ PyPI 배포명 매핑(패키지명/배포명이 다른 경우)
SPECIAL_DIST_MAP: Dict[str, str] = {
    "yaml": "PyYAML",
    "PIL": "Pillow",
    "skimage": "scikit-image",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
    "mpl_toolkits": "matplotlib",  # 종속 서브패키지
}

def find_py_files(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        pth = Path(p)
        if pth.is_file() and pth.suffix == ".py":
            files.append(pth)
        elif pth.is_dir():
            files.extend([f for f in pth.rglob("*.py")])
    return files

def extract_imports(pyfile: Path) -> Set[str]:
    """파일에서 최상위 import된 모듈명을 수집 (top-level 이름만)."""
    text = pyfile.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text, filename=str(pyfile))
    mods: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                top = (n.name or "").split(".")[0]
                if top:
                    mods.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                mods.add(top)
    return mods

def map_module_to_distribution(mods: Set[str]) -> Dict[str, str]:
    """
    import된 top-level module명을 PyPI 배포 패키지명으로 매핑.
    우선순위: SPECIAL_DIST_MAP → packages_distributions() 역매핑 → fallback(그대로 사용)
    """
    dist_map: Dict[str, str] = {}
    # 역매핑: {package_name: [distribution_names]}
    pkg2dists = importlib_metadata.packages_distributions()  # type: ignore

    for m in sorted(mods):
        # stdlib 비슷한 건 제외
        if m in STDLIKE:
            continue
        dist_name: Optional[str] = None

        # 1) 스페셜 매핑
        if m in SPECIAL_DIST_MAP:
            dist_name = SPECIAL_DIST_MAP[m]
        # 2) 설치된 배포에서 역매핑
        elif m in pkg2dists:
            # 가장 앞의 배포명을 선택
            dist_name = pkg2dists[m][0]
        else:
            # 3) matplotlib.axes 같은 서브패키지 케이스 방어
            # 이미 top-level만 뽑았으므로 여기선 그대로 둠. (없다면 이후 버전 조회에서 실패할 수 있음)
            dist_name = m

        dist_map[m] = dist_name
    return dist_map

def next_major(version: str) -> str:
    """'X.Y.Z' → 'X+1.0.0' 형태로 다음 메이저 버전 상한을 구함."""
    parts = version.split(".")
    try:
        major = int(parts[0])
    except Exception:
        # 예외적 버전 표기(예: '5.0.0rc1')는 대략적인 상한으로 처리
        return "9999"
    return f"{major+1}.0.0"

def freeze_requirements(dist_map: Dict[str, str]) -> Dict[str, str]:
    """
    배포명 → 버전 스펙 문자열 생성 (예: 'numpy>=2.1.1,<3.0.0')
    설치되지 않은 경우엔 버전 없이 배포명만 남김.
    """
    reqs: Dict[str, str] = {}
    for mod, dist in dist_map.items():
        try:
            ver = importlib_metadata.version(dist)  # type: ignore
            upper = next_major(ver)
            spec = f"{dist}>={ver},<{upper}"
        except importlib_metadata.PackageNotFoundError:
            # 설치 안 된 경우: 일단 배포명만 기록
            spec = dist
        except Exception:
            spec = dist
        reqs[dist] = spec
    return reqs

def main():
    ap = argparse.ArgumentParser(description="Generate requirements.txt from project imports.")
    ap.add_argument("paths", nargs="+", help="Scan these files/directories for .py imports")
    ap.add_argument("-o", "--output", default="requirements.txt", help="Output requirements file")
    args = ap.parse_args()

    pyfiles = find_py_files(args.paths)
    if not pyfiles:
        print("No .py files found under given paths.", file=sys.stderr)
        sys.exit(1)

    all_mods: Set[str] = set()
    for f in pyfiles:
        try:
            all_mods |= extract_imports(f)
        except Exception as e:
            print(f"[WARN] Parse failed: {f} ({e})", file=sys.stderr)

    # 프로젝트에서 실제 사용하는 외부들만 추리기 (우리 코드에 맞춰 약간의 화이트리스트 가이드)
    # 소스에 import가 있다면 자동으로 포함되므로 굳이 강제는 필요 없지만,
    # 안전을 위해 'Pillow'(matplotlib 저장용) 같은 우회 종속을 보강하고 싶다면 아래 라인 사용:
    # if "matplotlib" in all_mods: all_mods.add("PIL")

    dist_map = map_module_to_distribution(all_mods)
    reqs = freeze_requirements(dist_map)

    # 프로젝트에 확실히 필요한 핵심이 누락되었는지 보강 (선택)
    for must in ("numpy", "matplotlib", "PyYAML", "plotly", "kaleido", "Pillow"):
        # 모듈명이 아니라 배포명 기준으로 체크
        has = any(must.lower() == k.lower() for k in reqs.keys())
        if not has:
            try:
                ver = importlib_metadata.version(must)
                upper = next_major(ver)
                reqs[must] = f"{must}>={ver},<{upper}"
            except Exception:
                reqs.setdefault(must, must)

    lines = [reqs[k] for k in sorted(reqs.keys(), key=lambda s: s.lower())]
    out_path = Path(args.output)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {out_path} with {len(lines)} packages.")
    for ln in lines:
        print("  -", ln)

if __name__ == "__main__":
    main()
