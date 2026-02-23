import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class CarSafetyModel(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256):
        super(CarSafetyModel, self).__init__()
        # Use EfficientNet-B0 as the feature extractor
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.feature_size = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity() 

        # LSTM to process the temporal sequence of frames
        self.lstm = nn.LSTM(input_size=self.feature_size, 
                            hidden_size=hidden_size, 
                            num_layers=2, 
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, Frames, C, H, W)
        batch_size, n_frames, c, h, w = x.size()
        
        # Flatten batch and frames for CNN processing
        x = x.view(batch_size * n_frames, c, h, w)
        features = self.backbone(x) 
        
        # Reshape back for LSTM (Batch, Seq_Length, Feature_Size)
        features = features.view(batch_size, n_frames, -1)
        
        # Use the hidden state of the last time step
        lstm_out, _ = self.lstm(features)
        return self.fc(lstm_out[:, -1, :])