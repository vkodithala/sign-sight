import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import onnxruntime as ort

# -------------------------
# Device Selection: prefer CUDA, else CPU
# -------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# -------------------------
# 1. Load and Prepare the Data
# -------------------------
with open("collected_asl_data.pkl", "rb") as f:
    data_dict = pickle.load(f)

# Create a sorted mapping from class names to integer labels.
class_names = sorted(data_dict.keys())
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
print("Class mapping:", class_to_idx)

# Prepare the dataset: flatten each landmark sample (assumed to be a list of (x, y) pairs)
data_list = []
labels_list = []
expected_length = None

for class_name, samples in data_dict.items():
    for landmarks in samples:
        flat_sample = np.array(landmarks, dtype=np.float32).flatten()
        if expected_length is None:
            expected_length = flat_sample.shape[0]
            print("Expected feature length:", expected_length)
        if flat_sample.shape[0] == expected_length:
            data_list.append(flat_sample)
            labels_list.append(class_to_idx[class_name])
        else:
            print(f"Skipping sample from {class_name} with shape {flat_sample.shape}")

data_array = np.array(data_list)
labels_array = np.array(labels_list)
print("Total samples used:", data_array.shape[0])
print("Observation shape:", data_array[0].shape)
num_classes = len(class_names)


# -------------------------
# 2. Define a Data Augmentation Transform for Landmarks
# -------------------------
def augment_landmarks(sample):
    """
    Given a 1D torch tensor (flattened landmarks), reshape to (num_points, 2),
    then apply random rotation (±10°), scaling (0.9–1.1×), and additive Gaussian noise.
    Return the augmented flattened tensor.
    """
    # Reshape to (num_points, 2)
    num_points = sample.shape[0] // 2
    landmarks = sample.view(num_points, 2).clone().cpu().numpy()

    # Compute the center of the landmarks.
    center = np.mean(landmarks, axis=0)

    # Random rotation angle (in radians, ±10°)
    angle = np.deg2rad(np.random.uniform(-10, 10))
    cos_val, sin_val = np.cos(angle), np.sin(angle)
    R = np.array([[cos_val, -sin_val], [sin_val, cos_val]])

    # Random scaling factor between 0.9 and 1.1.
    scale = np.random.uniform(0.9, 1.1)

    # Subtract center, apply rotation and scaling, then add center back.
    transformed = (landmarks - center) @ R.T * scale + center

    # Add Gaussian noise (mean=0, std=0.01)
    noise = np.random.normal(0, 0.01, size=transformed.shape)
    transformed += noise

    # Flatten back to 1D and convert to tensor.
    augmented = torch.tensor(transformed.flatten(), dtype=torch.float32)
    return augmented.to(sample.device)


# -------------------------
# 3. Create a PyTorch Dataset
# -------------------------
class ASLDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data  # numpy array of shape (N, input_dim)
        self.labels = labels  # numpy array of shape (N,)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        if self.transform:
            sample = self.transform(sample)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label


# -------------------------
# 4. Split into Training and Testing Sets
# -------------------------
num_samples = data_array.shape[0]
indices = np.random.permutation(num_samples)
test_ratio = 0.2
num_test = int(test_ratio * num_samples)
num_train = num_samples - num_test

train_indices = indices[:num_train]
test_indices = indices[num_train:]

train_data = data_array[train_indices]
train_labels = labels_array[train_indices]
test_data = data_array[test_indices]
test_labels = labels_array[test_indices]

# For training, apply augmentation; for testing, no augmentation.
train_dataset = ASLDataset(train_data, train_labels, transform=augment_landmarks)
test_dataset = ASLDataset(test_data, test_labels, transform=None)

print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# -------------------------
# 5. Define the Model
# -------------------------
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


input_dim = expected_length
model = SimpleClassifier(input_dim, num_classes)
model = model.to(device)
print(model)

# -------------------------
# 6. Training Setup
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 125

# -------------------------
# 7. Training Loop
# -------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# -------------------------
# 8. Export the Trained Model to ONNX
# -------------------------
dummy_input = torch.randn(1, input_dim, device=device)
onnx_model_path = "asl_classifier.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    export_params=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
print("Model exported to ONNX at:", onnx_model_path)

# -------------------------
# 9. Evaluate the ONNX Model using ONNX Runtime
# -------------------------
ort_session = ort.InferenceSession(onnx_model_path)

onnx_preds = []
onnx_targets = []

# Loop over the test dataset (using batch_size=1 for simplicity)
for sample, target in test_dataset:
    input_np = sample.unsqueeze(0).numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: input_np}
    ort_outs = ort_session.run(None, ort_inputs)
    # Assume output shape is (1, num_classes)
    pred = int(np.argmax(ort_outs[0], axis=1)[0])
    onnx_preds.append(pred)
    onnx_targets.append(int(target.item()))

accuracy = accuracy_score(onnx_targets, onnx_preds)
precision = precision_score(onnx_targets, onnx_preds, average="macro", zero_division=0)
recall = recall_score(onnx_targets, onnx_preds, average="macro", zero_division=0)
f1 = f1_score(onnx_targets, onnx_preds, average="macro", zero_division=0)

print("\nEvaluation Metrics using ONNX Model on Test Set:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1 Score (macro): {f1:.4f}")

# -------------------------
# 10. Optionally, Save the PyTorch Model State
# -------------------------
torch.save(model.state_dict(), "asl_classifier.pth")
print("PyTorch model saved as asl_classifier.pth")
