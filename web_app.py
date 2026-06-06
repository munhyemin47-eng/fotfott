import streamlit as st
import tempfile, os, trimesh
import plotly.graph_objects as go
from processor import analyze_video

st.set_page_config(page_title="AI 3D 발 측정기", layout="wide")
st.title("👟 AI 3D 발 사이즈 측정기")

uploaded_file = st.file_uploader("영상을 업로드하세요 (.mp4)", type=["mp4"])

if uploaded_file and st.button("측정 시작"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read()); tfile.close()
    
    try:
        # 로딩 중 표시 (Spinner)
        with st.spinner('AI가 딥러닝 분석 중입니다... 잠시만 기다려주세요!'):
            l, w, i = analyze_video(tfile.name)
        
        st.success("✅ 분석 완료!")
        
        col1, col2 = st.columns([1, 2])
        col1.metric("발 길이", f"{l:.1f} mm"); col1.metric("발 볼", f"{w:.1f} mm"); col1.metric("발등 높이", f"{i:.1f} mm")
        
        # 3D 모델 로드 (경로: models/foot_V3/)
        obj_path = os.path.join("models", "foot_V3", "11536_foot_V3.obj")
        
        if os.path.exists(obj_path):
            mesh = trimesh.load(obj_path)
            mesh.apply_scale([l/260, w/100, i/60]) 
            fig = go.Figure(data=[go.Mesh3d(x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2], 
                                            i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2], 
                                            color='cyan', opacity=0.8)])
            col2.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"모델 파일을 찾을 수 없습니다: {obj_path}")
            
    finally:
        if os.path.exists(tfile.name): os.remove(tfile.name)