import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import CarSafetyModel

def run_inference(target_folder, model_path="car_safety_model.pth"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model
    model = CarSafetyModel().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Weights loaded from {model_path}")
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Get images from the target directory
    img_files = sorted([os.path.join(target_folder, f) for f in os.listdir(target_folder)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    frame_buffer = []

    print("--- Inference Started ---")
    for img_path in img_files:
        img = Image.open(img_path).convert('RGB')
        frame_buffer.append(transform(img))

        # Perform inference when buffer has N frames
        if len(frame_buffer) == 10:
            input_tensor = torch.stack(frame_buffer).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                prediction = torch.argmax(probs).item()
                confidence = probs[0][prediction].item()

            status = "EMERGENCY" if prediction == 1 else "NORMAL"
            print(f"Frame: {os.path.basename(img_path)} | Result: {status} | Conf: {confidence:.4f}")
            
            # Sliding window: remove oldest frame
            frame_buffer.pop(0)

if __name__ == "__main__":
    # Path to the folder containing frames to test
    test_folder = "./test_frames" 
    run_inference(test_folder)