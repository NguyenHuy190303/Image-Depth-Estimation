import os
import time
import base64
import streamlit as st

from scripts.pixelwise_matching import *
from scripts.window_based_matching import *
from scripts.metrics import * 
from components.streamlit_footer import footer


@st.cache_data(max_entries=1000)
def inference_and_display_result(algo_type, 
                                 similiarity_type, 
                                 left_img_path,
                                 right_img_path, 
                                 disparity_range, 
                                 kernel_size,
                                 scale):
    if algo_type == 'Pixel-wise matching':
        depth, depth_color = pixel_wise_matching(left_img_path=left_img_path,
                                                 right_img_path=right_img_path,
                                                 similiarity_type=similiarity_type,
                                                 disparity_range=disparity_range,
                                                 scale=scale)
    elif algo_type == 'Window-based matching':
        depth, depth_color = window_based_matching(left_img_path=left_img_path,
                                                   right_img_path=right_img_path,
                                                   similiarity_type=similiarity_type,
                                                   disparity_range=disparity_range,
                                                   kernel_size=kernel_size,
                                                   scale=scale)
        
    return depth, depth_color


def main():
    st.set_page_config(
        page_title="AIO2024 Module02 Project Image Depth Estimation - AI VIETNAM",
        page_icon='static/aivn_favicon.png',
        layout="wide"
    )

    col1, col2 = st.columns([0.8, 0.2], gap='large')
    
    with col1:
        st.title('AIO2024 - Module02 - Image Project')
        st.title(':sparkles: :blue[Stereo Matching] Image Depth Estimation Demo')
        
    with col2:
        logo_img = open("static/aivn_logo.png", "rb").read()
        logo_base64 = base64.b64encode(logo_img).decode()
        st.markdown(
            f"""
            <a href="https://aivietnam.edu.vn/">
                <img src="data:image/png;base64,{logo_base64}" width="full">
            </a>
            """,
            unsafe_allow_html=True,
        )

    algo_type = st.selectbox('**Type of matching algorithm**',
                                ('Pixel-wise matching', 'Window-based matching'))
    if algo_type == 'Pixel-wise matching':
        similarity_type = st.selectbox('**Type of similarity function**',
                                    ('l1', 'l2'))
        kernel_size = 0
    else: 
        similarity_type = st.selectbox('**Type of similarity function**',
                                    ('l1', 'l2', 'cosine', 'correlation'))
        kernel_size = st.slider('**Kernel size**', 1, 15, 3, 2)
    img_content = st.selectbox('**Image content**',
                                ('Tsukuba', 'Aloe'))
    if img_content == 'Aloe':
        aloe_right_version = st.selectbox('**Aloe right image version**',
                                    ('Version 1', 'Version 2', 'Version 3'))
    if img_content == 'Tsukuba':
        left_img_path = 'img_contents/Tsukuba/left.png'
        right_img_path = 'img_contents/Tsukuba/right.png'
    elif img_content == 'Aloe':
        left_img_path = 'img_contents/Aloe/Aloe_left_1.png'
        if aloe_right_version == 'Version 1':
            right_img_path = 'img_contents/Aloe/Aloe_right_1.png'
        elif aloe_right_version == 'Version 2':
            right_img_path = 'img_contents/Aloe/Aloe_right_2.png'
        elif aloe_right_version == 'Version 3':
            right_img_path = 'img_contents/Aloe/Aloe_right_3.png'
    else:
        raise Exception('Image content not found!')

    if algo_type == 'Pixel-wise matching':
        disparity_range = st.slider('**Disparity range**', 1, 200, 16)
        scale = st.slider('**Scale factor**', 1, 200, 16)
    else:
        disparity_range = st.slider('**Disparity range**', 1, 200, 64)
        scale = st.slider('**Scale factor**', 1, 200, 3)

    submitted = st.button('Submit')

    st.divider()

    if submitted:
        st.write(f'Input: {img_content}')
        col1, col2 = st.columns(2, gap='small')
        col1.image(left_img_path, caption='Left image')
        col2.image(right_img_path, caption='Right image')
        start_time = time.time()
        depth, depth_color = inference_and_display_result(algo_type=algo_type,
                                                          similiarity_type=similarity_type,
                                                          left_img_path=left_img_path,
                                                          right_img_path=right_img_path,
                                                          disparity_range=disparity_range,
                                                          kernel_size=kernel_size,
                                                          scale=scale)
        
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
        end_time = time.time() - start_time
        st.write(f'Time processing: {end_time}s')
        col1.image(depth, caption='Disparity map')
        col2.image(depth_color, caption='Disparity map (in heatmap)')


    footer()


if __name__ == '__main__':
    main()