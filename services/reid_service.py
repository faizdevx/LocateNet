"""
Placeholder — untrained projection head; not a valid ReID model.
Planned replacement: OSNet (torchreid) trained on Market-1501.

This service is disabled by default (ENABLE_REID=False in config.py). torch and torchvision
are imported lazily, so they are not needed unless ReID is switched on.
"""
import numpy as np


class ReIDService:
    def __init__(self):
        """
        Initialize the placeholder feature extractor (ImageNet MobileNetV2 + random 512-d head).
        """
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
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

        print(f"ReID placeholder loaded on {self.device}")

    def extract_feature(self, person_crop):
        """
        Takes a cropped image of a person (from YOLO)
        and returns a 512-d embedding.
        """
        if person_crop is None or person_crop.size == 0:
            return None

        # Preprocess the crop
        input_tensor = self.transform(person_crop).unsqueeze(0).to(self.device)

        with self._torch.no_grad():
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
