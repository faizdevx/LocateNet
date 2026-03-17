import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
import numpy as np
import cv2

class ReIDService:
    def __init__(self):
        """
        Initialize the ReID model (OSNet or MobileNet-based ReID).
        We use a 512-d output layer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # In a production environment, you'd load a weights file here.
        # For this setup, we'll initialize a feature extractor.
        self.model = mobilenet_v2(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.last_channel, 512)
        )
        self.model.to(self.device)
        self.model.eval()

        # ReID standard preprocessing: Resize to 256x128 (standard for bodies)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ ReID Service loaded on {self.device}")

    def extract_feature(self, person_crop):
        """
        Takes a cropped image of a person (from YOLO) 
        and returns a 512-d embedding.
        """
        if person_crop is None or person_crop.size == 0:
            return None

        # Preprocess the crop
        input_tensor = self.transform(person_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(input_tensor)
            
        # Move to CPU and normalize
        embedding = embedding.cpu().detach().numpy().flatten()
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
            
        return embedding

    def get_batch_embeddings(self, image, bboxes):
        """
        Takes a full frame and a list of YOLO bounding boxes [[x1, y1, x2, y2], ...]
        Returns a list of 512-d embeddings for each person.
        """
        body_embeddings = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            # Crop the person from the original image
            person_crop = image[y1:y2, x1:x2]
            
            if person_crop.size > 0:
                emb = self.extract_feature(person_crop)
                body_embeddings.append(emb)
            else:
                body_embeddings.append(None)
                
        return body_embeddings