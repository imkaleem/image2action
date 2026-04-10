import os
import rdflib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

############################################
# 1️⃣ Load RDF Graph
############################################

print("Loading RDF graphs...")

g = rdflib.Graph()
g.parse("schema.ttl", format="turtle")
g.parse("bdd_100k_val_data.ttl", format="turtle")
g.parse("coco_traffic_val_data.ttl", format="turtle")

print(f"Graph loaded with {len(g)} triples")

############################################
# 2️⃣ SPARQL Query
############################################

query = """
PREFIX cv: <http://vision.semkg.org/onto/v0.1/>

SELECT ?path ?action
WHERE {
  ?img cv:filePath ?path ;
       cv:containsObject ?vehicle .
  ?vehicle cv:action ?action .

  FILTER(?action != cv:UnknownAction)
}
"""

results = g.query(query)

data = []
for row in results:
    path = str(row.path)
    action = str(row.action)
    data.append((path, action))

print(f"Total samples from KG: {len(data)}")

############################################
# 3️⃣ Convert URIs → Numeric Labels
############################################

# Extract unique actions
unique_actions = sorted(list(set([a for _, a in data])))

label_map = {uri: idx for idx, uri in enumerate(unique_actions)}
idx_to_label = {v: k for k, v in label_map.items()}

print("\nLabel Mapping:")
for k, v in label_map.items():
    print(f"{k.split('/')[-1]} → {v}")

# Convert dataset
dataset = [(path, label_map[action]) for path, action in data]

############################################
# 4️⃣ Train/Test Split
############################################

train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

############################################
# 5️⃣ PyTorch Dataset
############################################

class KGImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = KGImageDataset(train_data, transform)
val_dataset = KGImageDataset(val_data, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

############################################
# 6️⃣ Model Setup (Transfer Learning)
############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(unique_actions))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

############################################
# 7️⃣ Training Loop
############################################

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    val_acc = evaluate(model, val_loader)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Training Loss: {running_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

############################################
# 8️⃣ Save Model
############################################

torch.save(model.state_dict(), "kg_action_classifier.pth")
print("\nModel saved as kg_action_classifier.pth")