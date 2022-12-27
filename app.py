import sys
import streamlit as st
sys.path.append('src/')
from src import pipeline

pipe = pipeline.MLPipeline()

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.write("""
            # Low Dose CT Scan Image Denoising
            This is an image reconstruction web application to remove noise from Low Dose CT scans using the [Multi-scale Dilation with Residual Fused Attention (MD-RFA) Network](https://github.com/kevinmfreire/MD-RFA)\n

            ### MD-RFA Demo\n
            You can see how the model works by trying out four different demos.  Each demo is a low dose CT scan, and takes about 5 seconds to denoise due to the size of the network.
            By clicking the denoise button for the desired image, the DICOM file goes through some pre processing in order to have the correct input data for the network, and the MD-RFA
            removes all noise artifacts and reconstructs the image into a higher quality image.
            """)

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        thoracic_img = 'data/sample/thoracic-low.dcm'
        st.header('Thoracic')
        st.image(pipe.display(thoracic_img))
        if st.button("denoise thoracic"):
            img = pipe.pre_process(thoracic_img)
            original = pipe.post_process(img)
            pred = pipe.predict(img)
            result = pipe.post_process(pred)

            st.write("Your denoise DICOM file:")
            st.image(result, caption="Denoised Image")
    
    with col2:
        head_img = 'data/sample/head-low.dcm'
        st.header('Head')
        st.image(pipe.display(head_img))
        if st.button("denoise head"):
            img = pipe.pre_process(head_img)
            original = pipe.post_process(img)
            pred = pipe.predict(img)
            result = pipe.post_process(pred)

            st.write("Your denoise DICOM file:")
            st.image(result, caption="Denoised Image")

    with col3:
        chest_img = 'data/sample/chest-low.dcm'
        st.header('Chest')
        st.image(pipe.display(chest_img))
        if st.button("denoise chest"):
            img = pipe.pre_process(chest_img)
            original = pipe.post_process(img)
            pred = pipe.predict(img)
            result = pipe.post_process(pred)

            st.write("Your denoise DICOM file:")
            st.image(result, caption="Denoised Image")

    with col4:
        abdomen_img = 'data/sample/abdomen-low.dcm'
        st.header('Abdomen')
        st.image(pipe.display(abdomen_img))
        if st.button("denoise abdomen"):
            img = pipe.pre_process(abdomen_img)
            original = pipe.post_process(img)
            pred = pipe.predict(img)
            result = pipe.post_process(pred)

            st.write("Your denoise DICOM file:")
            st.image(result, caption="Denoised Image")

    st.write("""
            ### Upload your own DICOM file!\n
            You can alsp upload your own DICOM file and the model will remove all noise artifacts and recosntruct the image.  If you wish to be able to download the denoised DICOM image then feel free to 
            reach out to me at kmfayora@gmail.com.
            """)

    file = st.file_uploader("Please upload an CT image", type=["dcm"])

    if file is None:
        st.text("Please upload a DICOM file")
    else:
        img = pipe.pre_process(file)
        original = pipe.display(file)
        pred = pipe.predict(img)
        result = pipe.post_process(pred)

        st.write("Your denoised DICOM file:")
        st.image([original, result], caption=["Before", "After"])