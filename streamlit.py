import streamlit as st
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import pandas as pd
import torch
import os

# The energy threshold calculated on all images from the training dataset
ENERGY_THRESHOLD = -3.038491129875183

# Mapping of prediction classes to their names
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

# Stores general recommendations in case the model decided an image is out-of-distribution
general_recommendations = """
- **Soil Preparation**: Ensure well-drained, fertile soil with a pH of 6.0 to 6.8.
- **Watering**: Water cucumbers consistently, aiming for 1-2 inches of water per week. Use drip irrigation or soaker hoses to keep foliage dry and reduce disease risk.
- **Fertilization**: Apply balanced fertilizers (such as 10-10-10) before planting and side-dress with nitrogen during the growing season.
- **Mulching**: Use organic mulch to conserve moisture, reduce weeds, and keep fruits clean.
- **Pest and Disease Monitoring**: Regularly inspect plants for signs of pests and diseases. Early detection is key to managing issues effectively.
- **Support and Spacing**: Use trellises or cages to support cucumber vines and ensure good air circulation by spacing plants appropriately.
- **Harvesting**: Pick cucumbers regularly to encourage continuous production. Harvest when fruits are firm and of desired size.
"""

def load_model(path: str) -> torch.nn.Module:
    '''
    Load a trained Pytorch model from a specified file path.

    Args:
        path (str): The file path to saved model.

    Returns:
        torch.nn.Module: The loaded Pytorch model
    '''
    model = torch.load(path)
    model.eval()
    return model

def load_recommendations(path: str) -> pd.DataFrame:
    '''
    Load a csv file containing recommendations from a specified file path.

    Args:
        path (str): The file path to the recommendations.

    Returns:
        pd.DataFrame: The loaded recommendations.
    '''
    recommendations = pd.read_csv(path)
    return recommendations

def transform_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    '''
    Transform an image for model inference.

    Args:
        image (Image.Image): The input image to be transformed.
        device (torch.device): The device to which image tensor will be moved.
    
    Returns:
        torch.Tensor: Transformed image tensor.
    '''
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    image = transform(image=np.array(image))['image']
    image_tensor = torch.tensor(image).unsqueeze(0).to(device)
    return image_tensor

def calculate_probability_and_predicted_class(output: torch.Tensor) -> tuple:
    '''
    Calculate probability and predicted class from the model.

    Args:
        output (torch.Tensor): The model's output tensor.

    Returns:
        A tuple, containing probability and preicted class.
    '''
    p = torch.softmax(output, dim=1)
    probability, predicted_class = torch.max(p, 1)
    return probability, predicted_class

def calculate_energy(output: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the energy of the model's output for out-of-distribution detection.

    Args:
        output (torch.Tensor): The model's output tensor.

    Returns:
        torch.Tensor: The energy of the current output.
    '''
    return -torch.logsumexp(output, dim=1)

def inference(model: torch.nn.Module, image: Image.Image, device: torch.device) -> dict:
    '''
    Perfomrs inference on an input image using the specified model.

    Args:
        model (torch.nn.Module): The trained Pytorch model for making predictions.
        image (Image.Image): The input image for inference.
        device (torch.device): The device for computation.

    Returns:
        The dictionary containing the inference result.
    '''

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
    
def classify_image(image: Image.Image):
    '''
    Classify the input image and display the results using Streamlit.

    Args:
        image (Image.Image): The image to be classified.
    '''
    model_path = "resnet50_cucumber.pth"
    model = load_model(path=model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model.to(device)
    result = inference(model, image, device)
    rec_path = "recommendations.csv"
    recommendations = load_recommendations(rec_path)

    if result['result'] == 'unknown':
        st.write("The class is of unknown origin")
        st.write(f"The energy is {result['energy']:.4f}, but the energy threshold is {ENERGY_THRESHOLD:.4f}")
        st.write(f"Potentially it could be: {class_idx_to_name_dict[result['predicted_class']]} with the probability {result['probability']:.4f}")
        st.write("General Recommendations for Cucumber:")
        st.write(general_recommendations)
    else:
        st.write(f"Probability: {result['probability']:.4f}")
        st.write(f"Predicted class: {class_idx_to_name_dict[result['predicted_class']]}")
        if class_idx_to_name_dict[result['predicted_class']] not in ["Healthy_Crop_Cucumber", "Healthy_Crop_Leaf"]:
            st.write("Recommendations")
            st.write("Pesticide Methods:")
            # This code can be read like this:
            # recommendations['disease_name' == 'predicted_disease_name']['give_pesticide_recom']
            st.markdown(recommendations[recommendations['disease'] == class_idx_to_name_dict[result['predicted_class']]]['pesticide'].values[0])
            st.write("Non-pesticide Methods:")
            st.markdown(recommendations[recommendations['disease'] == class_idx_to_name_dict[result['predicted_class']]]['non-pesticide'].values[0])
        else:
            st.markdown(recommendations[recommendations['disease'] == class_idx_to_name_dict[result['predicted_class']]]['maintenance'].values[0])

def main():
    '''
    Main function to run the Streamlit app for cucumber dataset.
    '''
    st.title("Disease Detection for Cucumber")
    st.write("Upload your image of cucumber or choose one of the example images below:")

    example_images = os.listdir("examples")
    example_images = [os.path.join("examples", img) for img in example_images]
    selected_example = st.selectbox("Choose an example image:", ['None'] + example_images)
    if selected_example != 'None':
        image = Image.open(selected_example)
        st.image(image, caption="Selected Example Image", use_column_width=True)
        st.write("Detecting...")
        classify_image(image)

    uploaded_file = st.file_uploader("Choose an image of cucumber...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Detecting...")
        classify_image(image)  

if __name__ == "__main__":
    main()