import streamlit as st
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import pandas as pd
import torch
import os
import random
import matplotlib.pyplot as plt

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

# Stores general recommendations in case the model decides an image is out-of-distribution
general_recommendations = """
- **Soil Preparation**: Ensure well-drained, fertile soil with a pH of 6.0 to 6.8.
- **Watering**: Water cucumbers consistently, aiming for 1-2 inches of water per week. Use drip irrigation or soaker hoses to keep foliage dry and reduce disease risk.
- **Fertilization**: Apply balanced fertilizers (such as 10-10-10) before planting and side-dress with nitrogen during the growing season.
- **Mulching**: Use organic mulch to conserve moisture, reduce weeds, and keep fruits clean.
- **Pest and Disease Monitoring**: Regularly inspect plants for signs of pests and diseases. Early detection is key to managing issues effectively.
- **Support and Spacing**: Use trellises or cages to support cucumber vines and ensure good air circulation by spacing plants appropriately.
- **Harvesting**: Pick cucumbers regularly to encourage continuous production. Harvest when fruits are firm and of desired size.
"""

# Random cucumber facts
cucumber_facts = [
    "Cucumbers are 95% water.",
    "Cucumbers belong to the same plant family as melons, including watermelon and cantaloupe.",
    "Cucumber slices can fight bad breath when pressed to the roof of your mouth with your tongue for 30 seconds.",
    "Cucumbers can be used to soothe sunburns.",
    "Cucumbers contain vitamin K, which is important for bone health.",
    "The phrase ‘cool as a cucumber’ is actually derived from the cucumber’s ability to cool the temperature of the blood."
]

def load_model(path: str) -> torch.nn.Module:
    """
    Load a PyTorch model from the specified file path.

    Args:
        path (str): The file path to the saved model.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    model = torch.load(path)
    model.eval()
    return model

def load_recommendations(path: str) -> pd.DataFrame:
    """
    Load recommendations from a CSV file.

    Args:
        path (str): The file path to the CSV containing recommendations.

    Returns:
        pd.DataFrame: DataFrame containing the recommendations.
    """
    recommendations = pd.read_csv(path)
    return recommendations

def transform_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Transform an image for model inference.

    Args:
        image (Image.Image): The input image to be transformed.
        device (torch.device): The device to which the image tensor will be moved.

    Returns:
        torch.Tensor: The transformed image tensor.
    """
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    image = transform(image=np.array(image))['image']
    image_tensor = torch.tensor(image).unsqueeze(0).to(device)
    return image_tensor

def calculate_probability_and_predicted_class(output: torch.Tensor) -> tuple:
    """
    Calculate the probability and predicted class from model output.

    Args:
        output (torch.Tensor): The output tensor from the model.

    Returns:
        tuple: A tuple containing the probability and predicted class.
    """
    p = torch.softmax(output, dim=1)
    probability, predicted_class = torch.max(p, 1)
    return probability, predicted_class

def calculate_energy(output: torch.Tensor) -> torch.Tensor:
    """
    Calculate the energy of the model output.

    Args:
        output (torch.Tensor): The output tensor from the model.

    Returns:
        torch.Tensor: The calculated energy.
    """
    return -torch.logsumexp(output, dim=1)

def inference(model: torch.nn.Module, image: Image.Image, device: torch.device) -> dict:
    """
    Perform inference on an input image using the specified model.

    Args:
        model (torch.nn.Module): The loaded PyTorch model.
        image (Image.Image): The input image for inference.
        device (torch.device): The device for computation.

    Returns:
        dict: A dictionary containing the inference result.
    """
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
    """
    Classify an input image and display the results using Streamlit.

    Args:
        image (Image.Image): The input image for classification.
    """
    model_path = "resnet50_cucumber.pth"
    model = load_model(path=model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model.to(device)
    result = inference(model, image, device)
    rec_path = "recommendations.csv"
    recommendations = load_recommendations(rec_path)

    if result['result'] == 'unknown':
        st.write("### The class is of unknown origin")
        st.write(f"**Energy**: {result['energy']:.4f}, but the energy threshold is {ENERGY_THRESHOLD:.4f}")
        st.write(f"**Potential class**: {class_idx_to_name_dict[result['predicted_class']]} with the probability {result['probability']:.4f}")
        st.write("### General Recommendations for Cucumber:")
        st.write(general_recommendations)
    else:
        st.write(f"### Probability: {result['probability']:.4f}")
        st.write(f"### Predicted class: {class_idx_to_name_dict[result['predicted_class']]}")
        if class_idx_to_name_dict[result['predicted_class']] not in ["Healthy_Crop_Cucumber", "Healthy_Crop_Leaf"]:
            st.write("### Recommendations")
            st.write("**Pesticide Methods:**")
            st.markdown(recommendations[recommendations['disease'] == class_idx_to_name_dict[result['predicted_class']]]['pesticide'].values[0])
            st.write("**Non-pesticide Methods:**")
            st.markdown(recommendations[recommendations['disease'] == class_idx_to_name_dict[result['predicted_class']]]['non-pesticide'].values[0])
        else:
            st.markdown(recommendations[recommendations['disease'] == class_idx_to_name_dict[result['predicted_class']]]['maintenance'].values[0])

def display_random_fact():
    """
    Display a random cucumber fact.
    """
    fact = random.choice(cucumber_facts)
    st.info(f"**Did you know?** {fact}")

def display_disease_frequency_chart(disease_counts: dict):
    """
    Display a bar chart of disease frequencies.

    Args:
        disease_counts (dict): A dictionary with disease names as keys and their counts as values.
    """
    diseases = list(disease_counts.keys())
    counts = list(disease_counts.values())

    plt.figure(figsize=(10, 6))
    plt.barh(diseases, counts, color='skyblue')
    plt.xlabel('Count')
    plt.title('Disease Frequency Detected by the Model')
    st.pyplot(plt)

def main():
    """
    Main function to run the Streamlit app for cucumber disease detection.
    """
    st.title("Disease Detection for Cucumber")
    
    # Display the GIF
    gif_url = "https://i.gifer.com/embedded/download/edP.gif"
    st.markdown(
        f'<img src="{gif_url}" width="600" alt="Cucumber Disease Detection"/>',
        unsafe_allow_html=True
    )
    
    st.write("Upload your image of cucumber or choose one of the example images below:")

    example_images = os.listdir("examples")
    example_images = [os.path.join("examples", img) for img in example_images]
    selected_example = st.selectbox("Choose an example image:", ['None'] + example_images)
    if selected_example != 'None':
        image = Image.open(selected_example)
        st.image(image, caption="Selected Example Image", use_column_width=True)
        st.write("### Detecting...")
        classify_image(image)

    uploaded_file = st.file_uploader("Choose an image of cucumber...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("### Detecting...")
        classify_image(image)  

    # Display random cucumber fact
    display_random_fact()

    # Display disease frequency chart
    disease_counts = {
        "Anthracnose_Fungi": 15,
        "Bacterial_Wilt_Bacteria": 10,
        "Belly_Rot_Fungi": 5,
        "Downy_Mildew_Fungi": 7,
        "Gummy_Stem_Blight_Fungi": 12,
        "Healthy_Crop_Cucumber": 30,
        "Healthy_Crop_Leaf": 25,
        "Pythium_Fruit_Rot_Fungi": 3
    }
    display_disease_frequency_chart(disease_counts)

    # Additional resources
    st.write("### Additional Resources")
    st.markdown("""
    - [Cucumber Growing Guide](https://www.almanac.com/plant/cucumbers)
    - [Cucumber Disease Management](https://www.vegetables.cornell.edu/crops/cucumber/)
    - [Organic Cucumber Production](https://attra.ncat.org/attra-pub/viewhtml.php?id=48)
    """)

if __name__ == "__main__":
    main()