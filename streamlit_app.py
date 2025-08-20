import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
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
    .nonmed { background-color: #ffe08a; border-left: 10px solid #ff9900; }
    .stMetric { background-color: #f8f9fa; padding: 1rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Load classification models
@st.cache_resource
def load_med_filter_model():
    model = tf.keras.models.load_model('models/med_vs_non_med_model.h5', compile=False)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    return model

@st.cache_resource
def load_tumor_model():
    model = tf.keras.models.load_model('models/brain_tumor_tensorflow_model.h5', compile=False)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    return model

# Helper functions
def classify_med_vs_nonmed(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    img = np.asarray(image)  # <-- float32
    img_reshape = img[np.newaxis, ...]
    
    prediction = model.predict(img_reshape, verbose=0) 
    print("Raw prediction:", prediction)
    confidence = float(prediction[0][0]) * 100

    # 1 = medical, 0 = non-medical
    predicted_class = 1 if prediction[0][0] >= 0.5 else 0
    return predicted_class, confidence


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
        x=predictions * 100,
        y=class_names,
        orientation='h',
        marker_color=['#28a745' if cls == 'NORMAL' else '#dc3545' for cls in class_names],
        text=[f'{p*100:.1f}%' for p in predictions],
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

# Tumor classes
class_names = ['Glioma', 'Meningioma', 'NORMAL', 'Neurocitoma', 'Outros', 'Schwannoma']
class_info = {
    'NORMAL': "No tumor detected.",
    'Glioma': "Most common primary brain tumor.",
    'Meningioma': "Usually benign tumor of brain membranes.", 
    'Neurocitoma': "Rare neuronal tumor, often benign.",
    'Schwannoma': "Benign nerve sheath tumor.",
    'Outros': "Other wounds. Further analysis recommended."
}

def main():
    st.markdown('<h1 class="main-header"> Brain Tumor Classification from MRI</h1>', unsafe_allow_html=True)

    # Load models
    med_filter_model = load_med_filter_model()
    tumor_model = load_tumor_model()
    weights = med_filter_model.get_weights()
    print([w.mean() for w in weights])
    with st.sidebar:
        st.image("test_data/NORMAL/T2_normal (240).jpeg", caption="MRI Brain Scan")
        st.markdown("### Instructions")
        st.markdown("""
        1. Upload an MRI brain scan image
        2. Wait for AI analysis
        3. Review prediction results
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
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
             # Step 1: Filter medical vs non-medical 
            med_class, med_conf = classify_med_vs_nonmed(image, med_filter_model)

            if med_class == 1:  # Non-medical
                st.warning(f"This does not look like a medical scan. Confidence: {med_conf:.1f}%. Please upload a valid MRI scan.")
            else:
                # Step 2: If medical, run tumor model
                with st.spinner("Analyzing MRI scan for brain tumor..."):
                    predictions, img_reshape = import_and_predict(image, tumor_model)
                    predicted_idx = np.argmax(predictions)
                    predicted_class = class_names[predicted_idx]
                    confidence = predictions[0][predicted_idx] * 100

                    # Show main prediction box
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

                    # Detailed predictions right below the box
                    st.subheader("Detailed Predictions")
                    fig = create_confidence_chart(predictions[0], class_names)
                    st.plotly_chart(fig, use_container_width=True) 

    else:
        st.info("Please upload an MRI scan image to begin analysis")

if __name__ == "__main__":
    main()
