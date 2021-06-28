import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import torchaudio.transforms as T
from torch.utils.data import DataLoader
import torch.utils.data.dataset as dataset
import pandas as pd
import os
import torchaudio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AICovidVNDataset(dataset.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.aicovidvn_data = pd.read_csv(csv_file)
        self.file_path = self.aicovidvn_data['file_path'].values
        self.assessment_result = self.aicovidvn_data['assessment_result'].values
        self.root_dir = root_dir

        self.transform = transform

    def __len__(self):
        return len(self.aicovidvn_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        SAMPLE_WAV_PATH = os.path.join(self.root_dir, self.file_path[idx])
        waveform, sample_rate = torchaudio.load(SAMPLE_WAV_PATH)
        waveform = waveform.to(device)
        if self.transform:
            waveform = self.transform(waveform)
        target = torch.tensor(self.assessment_result[idx], dtype=torch.float32, device=device)
        sample = (waveform, target)
        return sample


def train_and_validate(model, loss_criterion, optimizer, scheduler, epochs=25):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)

    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''

    start = time.time()
    history = []
    best_loss = 100000.0
    best_epoch = None

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        # Set to training mode
        model.train()

        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clean existing gradients
            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            outputs = torch.squeeze(outputs, dim=1)
            sigmoid = nn.Sigmoid()
            outputs = sigmoid(outputs)

            # Compute losss
            loss = loss_criterion(outputs, labels)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()
            scheduler.step()

            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)

            # Compute the accuracy
            predictions = outputs >= 0.5
            # ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)

            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(test_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                # ret, predictions = torch.max(outputs.data, 1)
                predictions = outputs >= 0.5
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)
                # if not j % 100:
                print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j,
                                                                                                           loss.item(),
                                                                                                           acc.item()))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch

        # Find average training loss and training accuracy
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss / test_data_size
        avg_valid_acc = valid_acc / test_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        epoch_end = time.time()

        print(
            "Epoch : {:03d}, Training: Loss - {:.4f}, Accuracy - {:.4f}%, \n\t\tValidation : Loss - {:.4f}, Accuracy - {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))

        # Save if the model has best accuracy till now
        if (epoch + 1) % 50:
            torch.save(model, 'Models/model_' + str(epoch) + '.pt')
    return model, history, best_epoch


if __name__ == '__main__':
    # Applying Transforms to the Data
    mfcc_transform = T.MFCC(
        sample_rate=8000,
        n_mfcc=256,
        melkwargs={
            'n_fft': 2048,
            'n_mels': 256,
            'hop_length': 512,
            'mel_scale': 'htk',
        }
    )

    train_dataset = AICovidVNDataset(csv_file='./Data/aicv115m_public_train/metadata_train_challenge.csv',
                                     root_dir='./Data/aicv115m_public_train/train_audio_files_8k',
                                     transform=transforms.Compose([
                                         mfcc_transform.to(device),
                                         transforms.Resize(256).to(device),
                                         transforms.CenterCrop(224).to(device)
                                     ]))
    lengths = [int(len(train_dataset) * 0.8), len(train_dataset) - int(len(train_dataset) * 0.8)]
    train_data, test_data = torch.utils.data.random_split(dataset=train_dataset, lengths=lengths,
                                                          generator=torch.Generator().manual_seed(42))

    # aa = list(test_data)
    # ss = 0
    # for a in aa:
    #     ss += a[1]
    # print(ss)

    # test_data = AICovidVNDataset(csv_file='./Data/aicv115m_public_test/metadata_public_test.csv',
    #                              root_dir='./Data/aicv115m_public_test/public_test_audio_files_8k',
    #                              transform=transforms.Compose([
    #                                  mfcc_transform,
    #                                  transforms.Resize(256),
    #                                  transforms.CenterCrop(224)
    #                              ]))
    # batch size
    bs = 16
    train_data_loader = DataLoader(train_data, batch_size=bs, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(test_data, batch_size=bs, shuffle=True, drop_last=False)

    train_data_size = len(train_data)
    test_data_size = len(test_data)

    # Load pretrained ResNet50 Model
    resnet50 = models.resnet50(pretrained=False)
    resnet50 = resnet50.to(device)
    resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Change the final layer of ResNet50 Model for Transfer Learning
    fc_inputs = resnet50.fc.in_features

    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    # Convert model to be used on GPU
    resnet50 = resnet50.to(device)

    # Define Optimizer and Loss Function
    loss_func = nn.BCELoss()
    num_epochs = 500
    optimizer = optim.Adam(resnet50.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    trained_model, history, best_epoch = train_and_validate(resnet50, loss_func, optimizer, scheduler, num_epochs)
    torch.save(history, 'history.pt')
