import torch
import torch.nn as nn
from torchvision import models, transforms

CLASS_LABELS = [
    'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle',
    'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau',
    'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees',
    'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher',
    'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard',
    'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx',
    'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier',
]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Same eval transforms used in the notebook (In [2] val_transform, In [27] custom_val_transform)
TRANSFORMS = {
    "resnet18": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    "cnn": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
}


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def load_model(arch, weights_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arch == "cnn":
        model = CNN(len(CLASS_LABELS))
    elif arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_LABELS))
    else:
        raise ValueError(f"Unknown arch: {arch}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval(), device


@torch.no_grad()
def predict_top5(model, image, arch, device, k=5):
    x = TRANSFORMS[arch](image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0]
    top = probs.topk(k)
    return [(CLASS_LABELS[i], p.item()) for p, i in zip(top.values, top.indices)]