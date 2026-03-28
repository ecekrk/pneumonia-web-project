import torch.nn as nn
from torchvision import models


def build_efficientnet_b3(num_classes=2):
    """
    EfficientNet-B3 seçme nedeni:
    - Transfer learning ile güçlü özellik çıkarımı sağlar
    - Sağlık görüntülerinde klasik basit CNN'lere göre daha iyi genelleme sunabilir
    - ImageNet pretrained ağırlıkları sayesinde daha kısa sürede iyi performans alınabilir
    """

    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

    # EfficientNet'in son sınıflandırma katmanını bizim problemimize uyarlıyoruz
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )

    return model