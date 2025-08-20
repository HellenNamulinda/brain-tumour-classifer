import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .normal { background-color: #6dcf84; border-left: 10px solid #28a745; }
    .tumor { background-color: #fa8c96; border-left: 10px solid #dc3545; }
    .stMetric { background-color: #f8f9fa; padding: 1rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/brain_tumor_tensorflow_model.h5', compile=False)
    # Build the model by running a dummy prediction with uint8 type
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    _ = model.predict(dummy_input, verbose=0)
    return model

class_names = ['Glioma', 'Meningioma', 'NORMAL', 'Neurocitoma', 'Outros', 'Schwannoma']
class_info = {
    'NORMAL': "No tumor detected.",
    'Glioma': "Most common primary brain tumor.",
    'Meningioma': "Usually benign tumor of brain membranes.", 
    'Neurocitoma': "Rare neuronal tumor, often benign.",
    'Schwannoma': "Benign nerve sheath tumor.",
    'Outros': "Other wounds. Further analysis recommended."
}

def import_and_predict(image_data, model):
    size = (224,224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction, img_reshape

def create_confidence_chart(predictions, class_names):
    fig = go.Figure(data=go.Bar(
        x=predictions[0] * 100,
        y=class_names,
        orientation='h',
        marker_color=['#28a745' if cls == 'NORMAL' else '#dc3545' for cls in class_names],
        text=[f'{p*100:.1f}%' for p in predictions[0]],
        textposition='inside'
    ))
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Confidence (%)",
        yaxis_title="Tumor Type",
        height=400,
        showlegend=False
    )
    return fig

def generate_gradcam(model, img_array, layer_name=None):
    try:
        _ = model.predict(img_array, verbose=0)

        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
            if layer_name is None:
                st.warning("No convolutional layer found for Grad-CAM.")
                return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.outputs]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, tf.argmax(predictions[0])]

        grads = tape.gradient(loss, conv_outputs)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * conv_outputs[0, :, :, i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam
    except Exception as e:
        st.warning("Grad-CAM generation failed: ")
        print(f"Grad-CAM generation failed: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header"> Brain Tumor Classification from MRI</h1>', unsafe_allow_html=True)
    model = load_model()

    with st.sidebar:
        st.image("test_data/NORMAL/T2_normal (240).jpeg", caption="MRI Brain Scan")
        st.markdown("### Instructions")
        st.markdown("""
        1. Upload an MRI brain scan image
        2. Wait for AI analysis
        3. Review prediction results
        4. Check model explanation
        """)
        st.markdown("---")
        st.markdown("**Warning:** This is a research demo. Not for medical diagnosis.")

    uploaded_file = st.file_uploader(
        "Upload MRI Scan Image", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear MRI brain scan image"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
        with col2:
            with st.spinner("Analyzing image..."):
                predictions, img_reshape = import_and_predict(image, model)
                predicted_idx = np.argmax(predictions)
                predicted_class = class_names[predicted_idx]
                confidence = predictions[0][predicted_idx] * 100

                if predicted_class == 'NORMAL':
                    st.markdown(f"""
                    <div class="prediction-box normal">
                        <h3>{predicted_class}</h3>
                        <p>{class_info[predicted_class]}</p>
                        <p><strong>Confidence: {confidence:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                    <div class="prediction-box tumor">
                        <h3>{predicted_class}</h3>
                        <p>{class_info.get(predicted_class, "Tumor detected.")}</p>
                        <p><strong>Confidence: {confidence:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        col3, col4 = st.columns([1, 1])
        with col3:
            st.subheader("Detailed Predictions")
            fig = create_confidence_chart(predictions, class_names)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.subheader("Model Explanation")
            gradcam = generate_gradcam(model, img_reshape)
            if gradcam is not None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.imshow(img_reshape[0])
                ax1.set_title("Original")
                ax1.axis('off')
                ax2.imshow(img_reshape[0])
                ax2.imshow(gradcam, alpha=0.6, cmap='jet')
                ax2.set_title("Model Focus Areas")
                ax2.axis('off')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Grad-CAM visualization not available")
        st.markdown("---") 
    else:
        st.info("Please upload an MRI scan image to begin analysis")

if __name__ == "__main__":
    main()