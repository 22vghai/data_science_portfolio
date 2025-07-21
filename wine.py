import streamlit as st
import numpy as np
import pandas as pd
import pickle

# App title 
st.title("Wine Quality Predictor")
st.markdown("Adjust the sliders to set wine features and get a predicted quality score")

#Load trained model and preprocessing pipeline
@st.cache_resource

def load_artifacts():
    with open('wine_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Define featue list manually for save and load if needed 
    features = ['fixed acidity', 'volatile acidity', 'citric acid',
                'residual sugar', 'chlorides', 'free sulfur dioxide',
                'total sulfur dioxide', 'density', 'pH', 'sulphates',
                'alcohol']
    return model, features

model, features = load_artifacts()

#Sliders for user input 
def get_user_input():
    inputs = {}
    for feature in features:
        step = 0.001 if "sulfur" in feature or "density" in feature else 0.1
        max_val = 20.0 if feature == 'alcohol' else 15.0
        default_val = 10.0 if feature == 'alcohol' else 7.0
        inputs[feature] = st.slider(
            feature.replace("_", " ").capitalize(),
            min_value=0.0,
            max_value=max_val,
            value=default_val,
            step=step
            )
    return pd.DataFrame([inputs])

user_df = get_user_input()

# Show user input
st.subheader("Your Wine's Features")
st.write(user_df)

#Make predictions
predictions = model.predict(user_df)[0]

st.subheader("Predicted Wine Quality")
st.metric("Prediction", f"{predictions:.2f}", 10)

#Optional: Show feature importances 

if st.checkbox("Show Feature Importances"):
    importances = np.abs(model.coef_) # use absolute value importance
    
    feat_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(feat_df.set_index("Feature"))

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit + Pickle")