import pandas as pd
import torch.utils.data.dataset as dataset
import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as T
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
        self.uuid = self.aicovidvn_data['uuid'].values
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
        sample = waveform
        return self.uuid[idx], sample


def predict(model):
    with torch.no_grad():
        model.eval()
        uuids = []
        outputs = []
        for (uuid, input) in test_data_loader:
            output = model(input)
            uuids.append(uuid[0])
            outputs.append(round(output.item(), 4))
    return uuids, outputs


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
    test_data = AICovidVNDataset(csv_file='./Data/aicv115m_public_test/metadata_public_test.csv',
                                 root_dir='./Data/aicv115m_public_test/public_test_audio_files_8k',
                                 transform=transforms.Compose([
                                     mfcc_transform.to(device),
                                     transforms.Resize(256).to(device),
                                     transforms.CenterCrop(224).to(device)
                                 ]))
    test_data_size = len(test_data)
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)
    resnet50 = torch.load('models/best_model.pt')
    uuids, outputs = predict(model=resnet50)
    data = {'uuid': uuids,
            'assessment_result': outputs}
    df = pd.DataFrame(data, columns=['uuid', 'assessment_result'])
    df.to_csv('results/results.csv', index=False, header=True)
