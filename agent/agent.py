import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
import torch
from torchvision import transforms
from PIL import Image
import requests
import timm
from io import BytesIO

#load Swin Transformer model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 37
model = timm.create_model('swin_base_patch4_window7_224',pretrained=False,num_classes=NUM_CLASSES)

model.load_state_dict(torch.load(r"demo\swin_transfomer_weight.pth", map_location=device))
model.to(device)
model.eval()

#map the predicted classes to the class names
class_names = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Bengal", "Basset Hound", "Beagle",
    "Birman", "Bombay", "British Shorthair", "Calico", "Chihuahua", "Chinese Crested", "Egyptian Mau",
    "English Bulldog", "English Setter", "German Shepherd", "Himalayan", "Jack Russell Terrier", "Japanese Chin",
    "Keeshond", "King Charles Spaniel", "Labrador Retriever", "Maine Coon", "Manchester Terrier",
    "Munchkin", "Newfoundland", "Persian", "Pomeranian", "Pug", "Ragdoll", "Saint Bernard", "Samoyed",
    "Scottish Fold", "Shiba Inu", "Siamese", "Sphynx", "Staffordshire Bull Terrier", "Wheaten Terrier"
]

# Define preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

def classify_image_from_url() -> dict:
    """Classifies a hardcoded image from a URL using the pretrained Swin Transformer."""
    try:
        # Hardcoded image URL
        image_url = "https://github.com/jalex3421/VisionTransformerImplementation/blob/main/demoImages/image_2.jpg?raw=true"
        
        # Fetch the image from the URL
        response = requests.get(image_url)
        if response.status_code != 200: return {"status": "error", "error_message": "Failed to download image from the URL."}

        # Open the image
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Preprocess the image (assuming transform is defined elsewhere)
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Perform inference with the pretrained model
        with torch.no_grad():
            output = model(input_tensor)
            pred_index = torch.argmax(output, 1).item()
        
        predicted_class = class_names[pred_index]

        return {"status": "success","prediction": f"Class: {predicted_class} (Index: {pred_index})"}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.
    Args:
        city (str): The name of the city for which to retrieve the weather report.
    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "okinawa":
        return {
            "status": "success",
            "report": (
                "The weather in Okinawa is sunny with a temperature of 27 degrees Celsius."
            ),
        }
    else:
        return {"status": "error","error_message": f"Weather information for '{city}' is not available."}

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.
    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "okinawa":
        tz_identifier = "Asia/Tokyo"  # Okinawa shares the same timezone as Tokyo (JST)
    else:
        return {
            "status": "error",
            "error_message": f"Sorry, I don't have timezone information for {city}.",
        }
    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}')
    return {"status": "success", "report": report}


root_agent = Agent(
    name="image_classifier_agent",
    model="gemini-2.0-flash",
    description="Agent that classifies images given an image URL (hardcoded).",
    instruction=(
        "You are a helpful agent that classifies an image. To trigger the classification, simply say 'classify image'. Also, you can get the weather and the time zone information "
    ),
    tools=[get_weather, get_current_time, classify_image_from_url],  
)
