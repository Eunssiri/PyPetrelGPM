#!/bin/bash
# ============================================================
# GPM (Geological Property Modeling) 전체 실행 스크립트
# ============================================================

# 에러 발생 시 즉시 종료
set -e

# 현재 경로 출력
echo "현재 경로: $(pwd)"
echo "=============================="

# 1) Petrel 텍스트 → NPZ 변환
echo "[1/4] Petrel → NPZ 변환 중..."
python software/petrel_to_numpy.py params/params_petrel.yml
echo "완료 ✅"
echo "------------------------------"

# 2) 전처리 (크롭/보간/스케일 + PNG/NPY 저장)
echo "[2/4] 전처리 진행 중..."
python software/gpm_preprocessing.py params/params_preproc.yml
echo "완료 ✅"
echo "------------------------------"

# 3) 단면 패널 PNG 생성
echo "[3/4] 단면 패널 시각화..."
python software/gpm_viz.py sections params/params_viz_sections.yml
echo "완료 ✅"
echo "------------------------------"

# 4) 3D 산점도 PNG 생성
echo "[4/4] 3D geometry 시각화..."
python software/gpm_viz.py geometry3d params/params_viz_geometry3d.yml
echo "완료 ✅"
echo "=============================="

echo "전체 프로세스 완료 🎉"
