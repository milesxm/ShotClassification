import os 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from neuralnet import CricketShotClassifier
import matplotlib.pyplot as plt

train_fraction = 0.7
val_fraction = 0.2
test_fraction = 0.1

train_loss_record = []

class CricketShotDataset(Dataset):
    def __init__(self, shot_folder, shot_label):
        self.shot_folder = shot_folder
        self.shot_label = shot_label
        self.files = os.listdir(shot_folder)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        shot_file = os.path.join(self.shot_folder, self.files[idx])
        shot_data = np.load(shot_file)
        shot_data = shot_data.reshape(shot_data.shape[0], -1)
        return torch.tensor(shot_data,dtype=torch.float32), torch.tensor(self.shot_label, dtype=torch.long)

#Path to data
cover_drive_path = "Data\poselandmarks\coverdrives"
pull_shot_path = "Data\poselandmarks\pullshots"
cut_shot_path = "Data\poselandmarks\cutshots"
sweep_shot_path = "Data\poselandmarks\sweepshots"
augmented_cover_drive_path = "Data\poselandmarks\coverdrivesaugmented"
augmented_pull_shot_path = "Data\poselandmarks\pullshotsaugmented"
augmented_cut_shot_path = "Data\poselandmarks\cutshotsaugmented"
augmented_sweep_shot_path = "Data\poselandmarks\sweepshotsaugmented"
cover_drive_flipped_path = "Data\poselandmarks\coverdrivesflipped"
pull_shot_flipped_path = "Data\poselandmarks\pullshotsflipped"
cut_shot_flipped_path = "Data\poselandmarks\cutshotsflipped"
sweep_shot_flipped_path = "Data\poselandmarks\sweepshotsflipped"
augmented_cover_drive_flipped_path = "Data\poselandmarks\coverdrivesflippedaugmented"
augmented_pull_shot_flipped_path = "Data\poselandmarks\pullshotsflippedaugmented"
augmented_cut_shot_flipped_path = "Data\poselandmarks\cutshotsflippedaugmented"
augmented_sweep_shot_flipped_path = "Data\poselandmarks\sweepshotsflippedaugmented"


# Adding the original keypoints to the dataset
cover_drives_dataset = CricketShotDataset(cover_drive_path, 0)
pull_shots_dataset = CricketShotDataset(pull_shot_path, 1)
cut_shots_dataset = CricketShotDataset(cut_shot_path, 2)
sweep_shots_dataset = CricketShotDataset(sweep_shot_path, 3)

# Adding the augmented keypoints to the dataset
augmented_cover_drives_dataset = CricketShotDataset(augmented_cover_drive_path, 0)
augmented_pull_shots_dataset = CricketShotDataset(augmented_pull_shot_path, 1)
augmented_cut_shots_dataset = CricketShotDataset(augmented_cut_shot_path, 2)
augmented_sweep_shots_dataset = CricketShotDataset(augmented_sweep_shot_path, 3)

# Adding the flipped keypoints to the dataset
cover_drives_flipped_dataset = CricketShotDataset(cover_drive_flipped_path, 0)
pull_shots_flipped_dataset = CricketShotDataset(pull_shot_flipped_path, 1)
cut_shots_flipped_dataset = CricketShotDataset(cut_shot_flipped_path, 2)
sweep_shots_flipped_dataset = CricketShotDataset(sweep_shot_flipped_path, 3)

# Adding the augmented flipped keypoints to the dataset
augmented_cover_drives_flipped_dataset = CricketShotDataset(augmented_cover_drive_flipped_path, 0)
augmented_pull_shots_flipped_dataset = CricketShotDataset(augmented_pull_shot_flipped_path, 1)
augmented_cut_shots_flipped_dataset = CricketShotDataset(augmented_cut_shot_flipped_path, 2)
augmented_sweep_shots_flipped_dataset = CricketShotDataset(augmented_sweep_shot_flipped_path, 3)


combined_dataset = ConcatDataset([cover_drives_dataset, pull_shots_dataset, augmented_cover_drives_dataset, augmented_pull_shots_dataset, 
                                  cut_shots_dataset, augmented_cut_shots_dataset, sweep_shots_dataset, augmented_sweep_shots_dataset, cover_drives_flipped_dataset, 
                                  pull_shots_flipped_dataset, augmented_cover_drives_flipped_dataset, augmented_pull_shots_flipped_dataset, cut_shots_flipped_dataset, 
                                  augmented_cut_shots_flipped_dataset, sweep_shots_flipped_dataset, augmented_sweep_shots_flipped_dataset])



total_length = len(combined_dataset)

train_length = int(train_fraction * total_length)
val_length = int(val_fraction * total_length)
test_length = total_length - train_length - val_length

train_dataset, val_dataset, test_dataset = random_split(combined_dataset, [train_length, val_length, test_length])

# Examples used before the model relearns 
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Creating the model and using gpu
model = CricketShotClassifier().to("cuda")

# Weighting the classes for the small dataset
#class_weights = torch.tensor([0.24,0.28,0.30,0.18]).to("cuda")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 2000
max_grad_norm = 1.0

for epoch in range(num_epochs):
    #PyTorch training mode
    model.train()

    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_dataloader:
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")

        optimizer.zero_grad()

        # forward pass 
        outputs = model(inputs)
        # Loss function
        loss = criterion(outputs, labels)
        loss.backward()

        # Clips gradients to avoid gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Gradient Descent
        optimizer.step()


        # gets current loss
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_dataloader)
    train_loss_record.append(train_loss)
    train_accuracy = correct_train / total_train * 100

    # Validation
    model.eval()
    running_loss_val = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss_val += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_loss_val / len(val_dataloader)
    val_accuracy = correct_val / total_val * 100

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")


plt.plot(train_loss_record)
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
torch.save(model.state_dict(), "cricketshotclassifierv6.2noweights.pth")

#testing model

model.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_accuracy = correct_test / total_test * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")