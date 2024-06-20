import streamlit as st
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import torch

ENERGY_THRESHOLD = -3.038491129875183

class_idx_to_name_dict = {
    0: "Anthracnose_Fungi",
    1: "Bacterial_Wilt_Bacteria",
    2: "Belly_Rot_Fungi",
    3: "Downy_Mildew_Fungi",
    4: "Gummy_Stem_Blight_Fungi",
    5: "Healthy_Crop_Cucumber",
    6: "Healthy_Crop_Leaf",
    7: "Pythium_Fruit_Rot_Fungi"
}

def load_model(path):
    model = torch.load(path)
    model.eval()
    return model

def transform_image(image, device):
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2
    ])
    image = transform(image=np.array(image))['image']
    image_tensor = torch.tensor(image).unsqueeze(0).to(device)
    return image_tensor

def calculate_probability_and_predicted_class(output):
    p = torch.softmax(output, dim=1)
    probability, predicted_class = torch.max(p, 1)
    return probability, predicted_class

def calculate_energy(output):
    return -torch.logsumexp(output, dim=1)

def inference(model, image, device):

    image_tensor = transform_image(image, device)

    with torch.inference_mode():
        output = model(image_tensor)
        energy = calculate_energy(output)
        energy_value = energy.item()
        probability, predicted_class = calculate_probability_and_predicted_class(output)
        probability_value = probability.item()
        predicted_class_value = predicted_class.item()
        if energy_value > ENERGY_THRESHOLD:            
            
            return {
                'result': 'unknown',
                'energy': energy_value,
                'probability': probability_value,
                'predicted_class': predicted_class_value
            }

        return {
            'result': 'known',
            'probability': probability_value,
            'predicted_class': predicted_class_value
        }

def main():
    st.title("Image Classification for Cucumber")

    uploaded_file = st.file_uploader("Choose an image of cucumber...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        model_path = "resnet50_cucumber.pth"
        model = load_model(path=model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        model.to(device)
        result = inference(model, image, device)

        if result['result'] == 'unknown':
            st.write("The class is of unknown origin")
            st.write(f"The energy is {result['energy']:.4f}, but the energy threshold is {ENERGY_THRESHOLD:.4f}")
            st.write(f"Probability: {result['probability']:.4f}")
            st.write(f"Predicted class: {class_idx_to_name_dict[result['predicted_class']]}")
        else:
            st.write(f"Probability: {result['probability']:.4f}")
            st.write(f"Predicted class: {class_idx_to_name_dict[result['predicted_class']]}")

if __name__ == "__main__":
    main()