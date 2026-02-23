import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from model import CarSafetyModel

class FrameSequenceDataset(Dataset):
    def __init__(self, root_dir, n_frames=10, transform=None):
        self.root_dir = root_dir
        self.n_frames = n_frames
        self.transform = transform
        self.samples = []
        
        for label_name in ['normal', 'emergency']:
            label = 0 if label_name == 'normal' else 1
            label_dir = os.path.join(root_dir, label_name)
            
            if not os.path.exists(label_dir): continue

            for seq_folder in os.listdir(label_dir):
                seq_path = os.path.join(label_dir, seq_folder)
                if not os.path.isdir(seq_path): continue
                
                # Get sorted list of frame images
                images = sorted([os.path.join(seq_path, f) for f in os.listdir(seq_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                # Sliding window to create sequences
                for i in range(0, len(images) - n_frames + 1, 5): 
                    self.samples.append((images[i:i+n_frames], label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths, label = self.samples[idx]
        frames = [self.transform(Image.open(p).convert('RGB')) for p in img_paths]
        return torch.stack(frames), label

def main():
    # Hardware Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = FrameSequenceDataset(root_dir='./Dataset', n_frames=10, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    model = CarSafetyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training Loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "car_safety_model.pth")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()