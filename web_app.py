import streamlit as st
import tempfile
import os
from processor import analyze_video
import plotly.graph_objects as go
import trimesh # OBJ 파일을 읽기 위해 필요

st.set_page_config(page_title="AI 3D 발 측정기", layout="centered")
st.title("👟 AI 3D 발 사이즈 측정기")

uploaded_file = st.file_uploader("영상 파일 (.mp4) 업로드", type=["mp4"])

if uploaded_file is not None:
    if st.button("측정 시작"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        try:
            # 로딩 중 표시 (Spinner)
            with st.spinner('AI가 딥러닝 분석 중입니다... 잠시만 기다려주세요!'):
                length, width, instep = analyze_video(tfile.name)
            
            st.success("✅ 분석 완료!")
            
            # 결과 표시
            col1, col2, col3 = st.columns(3)
            col1.metric("발 길이", f"{length:.1f} mm")
            col2.metric("발 볼", f"{width:.1f} mm")
            col3.metric("발등 높이", f"{instep:.1f} mm")
            
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
        finally:
            if os.path.exists(tfile.name):
                os.remove(tfile.name)
mesh = trimesh.load('11536_foot_V3.obj')

# 분석된 수치(l, w, i)를 기반으로 스케일 조정
# 모델의 기본 크기를 mm 단위로 스케일링하는 로직입니다
scale_x = length / 260.0  # 예: 모델의 기본 길이가 260mm일 때 비율 계산
scale_y = width / 100.0
scale_z = instep / 60.0
mesh.apply_scale([scale_x, scale_y, scale_z])

# 3D 시각화 생성
fig = go.Figure(data=[go.Mesh3d(
    x=mesh.vertices[:, 0],
    y=mesh.vertices[:, 1],
    z=mesh.vertices[:, 2],
    i=mesh.faces[:, 0],
    j=mesh.faces[:, 1],
    k=mesh.faces[:, 2],
    color='lightpink', opacity=0.8
)])

# 스트림릿에 표시
st.plotly_chart(fig)
            