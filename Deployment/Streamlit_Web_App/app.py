from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms

from deployment_model_vit_h_14 import DeploymentModelViT_H_14
from deployment_model_dino_v2 import DeploymentModel_DINO_v2

# Load the ViT-H_14 model and the DINOv2 model
vit_h_model = DeploymentModelViT_H_14.getInstance()
dino_v2_model = DeploymentModel_DINO_v2.getInstance()

def load_images(img_list, limit=100):
    for i, image_file in enumerate(img_list):
        if i < limit:
            if i % 5 == 0:
                cols = st.columns(5)
            cols[i % 5].image('/home/group15/VPR/visual-product-recognition-2/Deployment/Streamlit_Web_App/' + image_file)


# Title of the app
st.title("Visual Product Recognition - ViT-H_14/DINOv2 Models - Group 15")

# Adding text
st.write("Visual Product Recognition systems let you find the product you are looking for very easily by replacing the"
         " traditional text queries with image queries. Let's get started...")
image_data = st.file_uploader(label='Upload the image you want to query on', type=['jpg', 'png'])

canvas_max_height = 500
canvas_max_width = 500

query_image_folder = '/home/group15/VPR/visual-product-recognition-2/Deployment/Streamlit_Web_App/query_images'

if image_data is not None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_file_name = str(timestamp) + '.jpg'
    uploaded_image = Image.open(image_data)
    uploaded_image.save('/home/group15/VPR/visual-product-recognition-2/Deployment/Streamlit_Web_App/query_images/' + image_file_name)
    width, height = uploaded_image.size

    canvas_height = canvas_max_height
    canvas_width = canvas_max_width

    image_pixel_by = 1
    if width >= height:
        image_pixel_by = canvas_max_width / width
        canvas_height = int(height * image_pixel_by)
    else:
        image_pixel_by = canvas_max_width / width
        canvas_width = int(width * image_pixel_by)

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # Fixed fill color with some opacity
        stroke_width=2,
        stroke_color="#FFFF00",
        background_color="#EEEEEE",
        background_image=uploaded_image,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="rect",
        key="canvas"
    )

    multiple_bounding_box_error = None
    bbox_x = 0
    bbox_y = 0
    bbox_w = 0
    bbox_h = 0

    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow

        number_of_bounding_boxes = len(objects)
        if number_of_bounding_boxes > 1:
            multiple_bounding_box_error = st.error('You cannot have multiple bounding boxes within a single image. '
                                                   'Please undo the previously drawn bounding box before drawing'
                                                   'another', icon="🚨")
        elif number_of_bounding_boxes == 1 and multiple_bounding_box_error is not None:
            multiple_bounding_box_error.empty()

        # Define a transform to convert PIL
        # image to a Torch tensor
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])

        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        if number_of_bounding_boxes == 1:
            bbox_x = int(int(objects['left']) / image_pixel_by)
            bbox_y = int(int(objects['top']) // image_pixel_by)
            bbox_w = int(int(objects['width']) // image_pixel_by)
            bbox_h = int(int(objects['height']) // image_pixel_by)
            st.write(f"Bounding Box : Left: {bbox_x}, Top: {bbox_y}, Width: {bbox_w}, Height: {bbox_h}")

            chosen_model = st.selectbox('Select a model', options=['Trained ViT-H_14', 'Zero Shot DINOv2'])
            model = None
            if chosen_model == 'Trained ViT-H_14':
                model = vit_h_model
            else:
                model = dino_v2_model

            if st.button("Search Similar Products", type="primary"):
                query_tensor = model.get_query_embedding(query_image_folder + '/' + image_file_name, bbox_x, bbox_y, bbox_w, bbox_h)
                list_of_images = model.get_matching_gallery_images(query_tensor)
                load_images(list_of_images)
