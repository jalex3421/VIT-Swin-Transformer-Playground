import timm
import torch
from torchvision import transforms
from PIL import Image
import gradio as gr

#define a dictionary to map the prediction with the available classes
class_names = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Bengal", "Basset Hound", "Beagle",
    "Birman", "Bombay", "British Shorthair", "Calico", "Chihuahua", "Chinese Crested", "Egyptian Mau",
    "English Bulldog", "English Setter", "German Shepherd", "Himalayan", "Jack Russell Terrier", "Japanese Chin",
    "Keeshond", "King Charles Spaniel", "Labrador Retriever", "Maine Coon", "Manchester Terrier",
    "Munchkin", "Newfoundland", "Persian", "Pomeranian", "Pug", "Ragdoll", "Saint Bernard", "Samoyed",
    "Scottish Fold", "Shiba Inu", "Siamese", "Sphynx", "Staffordshire Bull Terrier", "Wheaten Terrier"
]

# Load model
model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=37)
model.load_state_dict(torch.load("trained_model.pth", map_location="cpu"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Adjust to match your training setup
])

def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        predicted = torch.argmax(outputs, dim=1).item()
    className = class_names[predicted]
    return f"Predicted Class: {className}/ Predicted Class id {predicted}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Swin Transformer Oxford Pets Classifier",
    description="Upload a pet image to predict its class."
)

demo.launch()
