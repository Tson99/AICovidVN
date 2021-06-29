import pandas as pd
import os

file_path = []
assessment_result = []
for path in os.listdir('./images/data_train'):
    file_path.append(path)
    text = path.split('_')[-1]
    text = text.split('.')[0]
    assessment_result.append(text)
    print(file_path, assessment_result)
data = {
    'file_path': file_path,
    'assessment_result': assessment_result
}
df = pd.DataFrame(data, columns=['file_path', 'assessment_result'])
df.to_csv('images/train_annotation.csv', index=False, header=True)