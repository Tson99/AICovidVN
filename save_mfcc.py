import librosa
import numpy as np
import pylab
import os
import pandas as pd
import librosa.display

custpath = './images/data_test'  # Where mel spec images will be stored
csv_file = './data/aicv115m_public_test/metadata_public_test.csv'
df = pd.read_csv(csv_file)
uuid = df['uuid']
# assessment_result = df['assessment_result']
file_path = df['file_path']


def feature_extractor(idx):
    if 1 == 1:
        path = os.path.join('./data/aicv115m_public_test/public_test_audio_files_8k', file_path[idx])
        audio, sr = librosa.load(path)

        # For MFCCS
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=39)
        mfccsscaled = np.mean(mfccs.T, axis=0)

        # Mel Spectogram
        pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        melspec = librosa.feature.melspectrogram(y=audio, sr=sr)
        s_db = librosa.power_to_db(melspec, ref=np.max)
        librosa.display.specshow(s_db)

        # savepath = os.path.join(custpath, str(uuid[idx]) + '_' + str(assessment_result[idx]) + '.png')
        savepath = os.path.join(custpath, str(uuid[idx]) + '.png')
        pylab.savefig(savepath, bbox_inches=None, pad_inches=0)
        pylab.close()
    return mfccsscaled, savepath


features = []
diagnoses = []
imgpaths = []

if __name__ == '__main__':
    for idx in range(len(df)):
        mfccs, savepath = feature_extractor(idx)
        # features.append(mfccs)
        # imgpaths.append(savepath)
        # diagnoses.append([row[3], row[4]])
