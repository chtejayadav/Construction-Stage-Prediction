import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# App title and layout
st.set_page_config(page_title="Construction Stage Classifier", layout="centered")
st.title("🏗️ Construction Stage Prediction App")

# Add gray background styling
def add_gray_backgrounds():
    css = """
    <style>
    /* Main app area - light gray gradient */
    .stApp {
        background: linear-gradient(to right, #F0F0F0, #D9D9D9);
        color: #222222;
        background-attachment: fixed;
        background-size: cover;
        background-repeat: no-repeat;
    }

    /* Sidebar - dark gray background */
    [data-testid="stSidebar"] {
        background-color: #333333 !important;
        color: #EEEEEE !important;
    }

    /* Sidebar headers, text, and links */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] a {
        color: #EEEEEE !important;
    }

    /* Sidebar inputs and buttons */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] button {
        color: #333333 !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply gray background theme
add_gray_backgrounds()

# Sidebar
st.sidebar.title("📘 About This App")
st.sidebar.info(
    """
This web application predicts the construction stage of a building from an uploaded image.

🔧 **Technologies Used** & 🧠 **Model Overview**
- TensorFlow & Keras for deep learning
- MobileNetV2 as the feature extraction backbone
- Streamlit for the interactive web interface
- Transfer learning to achieve high accuracy with minimal data
- Pretrained MobileNetV2 model (on ImageNet)
- Fine-tuned on 12 construction site images
- Supports 6 unique stages of construction

🏗️ **Recognized Construction Stages**
1. Site Preparation
2. Foundation Work
3. Structural Framework
4. Roofing & Exterior
5. Interior Work
6. Final Touches

📂 **Input Requirement**
- Accepts JPG or PNG images
- Images must be clear and focused on the site
    """
)


# Constants
image_dir = "dataset"
image_size = (128, 128)
num_classes = 6
model_path = "construction_stage_model.h5"

# Stage descriptions
stage_labels = {
    0: (
        "### Stage 0 – Site Preparation\n"
        "🟡 Work Done: 0% – 16.6% \n\n"
        "This stage marks the beginning of the project. The construction site is cleared of vegetation, debris, and any obstructions. "
        "The ground is leveled and compacted. Layouts are marked, utilities might be connected, and access paths are created.\n\n"
        "✅ Work Completed: Land cleared, leveled, and basic layout done.\n"
        "\n Estimated Time to Completion"
    ),
    1: (
        "### Stage 1 – Foundation Work\n"
        "🟡 Work Done: 16.7% – 33.3% \n\n"
        "This involves excavation and construction of the foundation. Concrete footings, steel reinforcements, and slabs are laid. "
        "Waterproofing layers are often added to prevent future moisture problems.\n\n"
        "✅ Work Completed: Excavation done, foundation and waterproofing laid.\n"
        "\n Estimated Time to Completion :- 6 months"
    ),
    2: (
        "### Stage 2 – Structural Framework\n"
        "🟡 Work Done: 33.4% – 50% \n\n"
        "The skeleton of the building is erected here. Beams, columns, floors, and slabs are added using concrete or steel. "
        "This gives the building its shape and support.\n\n"
        "✅ Work Completed: Load-bearing framework and floors completed.\n"
        "\n Estimated Time to Completion :- 4.5 to 5 months"
    ),
    3: (
        "### Stage 3 – Roofing & Exterior\n"
        "🟢 Work Done: 50.1% – 66.6% \n\n"
        "The roof is built and sealed, making the structure weather-resistant. External walls are built and plastered. "
        "Doors and window frames may be installed at this stage.\n\n"
        "✅ Work Completed: Roof constructed, exterior enclosure complete.\n"
        "\n Estimated Time to Completion :- 3.5 to 4 months"
    ),
    4: (
        "### Stage 4 – Interior Work\n"
        "🟢 Work Done: 66.7% – 83.3% \n\n"
        "Includes plumbing, electrical wiring, wall partitioning, and initial finishes. The core services are routed and rooms take form. "
        "Putty and primer may be applied as preparation for painting.\n\n"
        "✅ Work Completed: Plumbing, wiring, and internal walls completed.\n"
        "\n Estimated Time to Completion :- 2 months"
    ),
    5: (
        "### Stage 5 – Final Touches\n"
        "✅ Work Done: 83.4% – 100% \n\n"
        "This final stage involves painting, flooring, electrical fittings, and quality checks. The site is cleaned and made ready for use.\n\n"
        "✅ Work Completed: Finishes applied, quality tested, site ready for handover.\n"
        "\n Estimated Time to Completion :- 0.75 months (~3 weeks)"
    )
}


# Load or train model
@st.cache_resource
def load_or_train_model():
    if os.path.exists(model_path):
        return load_model(model_path)

    st.warning("Training new model from scratch...")

    images, labels = [], []
    for i in range(1, 13):
        path = os.path.join(image_dir, f"{i}.jpg")
        img = load_img(path, target_size=image_size)
        img_array = preprocess_input(img_to_array(img))
        images.append(img_array)
        labels.append((i - 1) // 2)

    X = np.array(images)
    y = to_categorical(labels, num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=4, verbose=0)

    model.save(model_path)
    return model

# Load model
model = load_or_train_model()

# Upload image
st.header("📤 Upload a Construction Site Image")
uploaded_file = st.file_uploader("Select a JPG or PNG image", type=["jpg", "jpeg", "png"])

# Prediction
if uploaded_file:
    try:
        img = load_img(uploaded_file, target_size=image_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        predicted_class = int(np.argmax(pred))
        confidence = float(np.max(pred))

        st.success(f"✅ Predicted Stage: **Stage {predicted_class}** (Confidence: {confidence:.2f})")
        st.markdown(stage_labels[predicted_class])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
