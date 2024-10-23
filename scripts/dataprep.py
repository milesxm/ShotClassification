import os 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from neuralnet import CricketShotClassifier

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
augmented_cover_drive_path = "Data\poselandmarks\coverdrivesaugmented"
augmented_pull_shot_path = "Data\poselandmarks\pullshotsaugmented"
augmented_cut_shot_path = "Data\poselandmarks\cutshotsaugmented"
cover_drive_flipped_path = "Data\poselandmarks\coverdrivesflipped"
pull_shot_flipped_path = "Data\poselandmarks\pullshotsflipped"
cut_shot_flipped_path = "Data\poselandmarks\cutshotsflipped"

# Adding the original keypoints to the dataset
cover_drives_dataset = CricketShotDataset(cover_drive_path, 0)
pull_shots_dataset = CricketShotDataset(pull_shot_path, 1)
cut_shots_dataset = CricketShotDataset(cut_shot_path, 2)

# Adding the augmented keypoints to the dataset
augmented_cover_drives_dataset = CricketShotDataset(augmented_cover_drive_path, 0)
augmented_pull_shots_dataset = CricketShotDataset(augmented_pull_shot_path, 1)
augmented_cut_shots_dataset = CricketShotDataset(augmented_cut_shot_path, 2)

# Adding the flipped keypoints to the dataset
cover_drives_flipped_dataset = CricketShotDataset(cover_drive_flipped_path, 0)
pull_shots_flipped_dataset = CricketShotDataset(pull_shot_flipped_path, 1)
cut_shots_flipped_dataset = CricketShotDataset(cut_shot_flipped_path, 2)


from torch.utils.data import ConcatDataset
combined_dataset = ConcatDataset([cover_drives_dataset, pull_shots_dataset, augmented_cover_drives_dataset, augmented_pull_shots_dataset, 
                                  cut_shots_dataset, augmented_cut_shots_dataset])

batch_size = 16
dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

model = CricketShotClassifier().to("cuda")


class_weights = torch.tensor([0.31,0.36,0.33]).to("cuda")
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000

for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:

        inputs = inputs.to("cuda")
        labels = labels.to("cuda")

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total * 100
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


torch.save(model.state_dict(), "cricketshotclassifierv4.2.pth")