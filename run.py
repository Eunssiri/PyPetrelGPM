import os

if __name__ == "__main__":
    # 1) Petrel 텍스트 → NPZ
    os.system("python software/petrel_to_numpy.py params/params_petrel.yml")

    # 2) 전처리(크롭/보간/스케일 + PNG/NPY 저장)
    os.system("python software/gpm_preprocessing.py params/params_preproc.yml")

    # 3) 단면 패널 PNG
    os.system("python software/gpm_viz.py sections params/params_viz_sections.yml")

    # 4) 3D 산점도 PNG
    os.system("python software/gpm_viz.py geometry3d params/params_viz_geometry3d.yml")

