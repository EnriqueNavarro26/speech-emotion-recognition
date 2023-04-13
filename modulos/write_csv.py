from utils import extract_process, export_process
from utils import add_noise, change_speed, shift_process, pitch_process

import os
import glob
import pandas as pd

# Define the paths to the data folders
path_tess = '../data/TESS/'
path_emodb = '../emotion-recognition-using-speech-master/data/emodb/wav/'
path_custom_1 = '../emotion-recognition-using-speech-master/data/train-custom/'
path_custom_2 = '../emotion-recognition-using-speech-master/data/test-custom/'
path_ravdess = '../data/ravdess/'
path_savee = '../data/SAVEE/'
path_mesd = '../data/MESD/'
path_ruso = '../data/younger_ruso/'

# DataAug
data_aug = False

# mapeo de emociones --> positivo, negativo, neutro
map_tess = {
    'fear':'negative',
    'pleasant': 'positive',
    'sad': 'negative',
    'angry': 'negative',
    'disgust':'negative',
    'happy':'positive',
    'neutral':'neutral'
    }

categories_emo = {
        "W": "angry",
        "L": "boredom",
        "E": "disgust",
        "A": "fear",
        "F": "happy",
        "T": "sad",
        "N": "neutral"
    }

map_emo = {
        "happy": "positive",
        "neutral": "neutral",
        "sad": "negative",
        "fear": "negative",
        "angry": "negative",
        "disgust": "negative",
        "boredom": "negative"
    }

map_custom = {
    'happy':'positive',
    'neutral': 'neutral',
    'sad': 'negative',
    }

map_ravdess = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
    }
map_sent_ravdess = {
    'neutral': 'neutral',
    'calm': 'positive',
    'happy': 'positive',
    'sad': 'negative',
    'angry': 'negative',
    'fearful': 'negative',
    'disgust': 'negative',
    'surprised': 'positive'
    }
map_savee = {
        'a': 'negative',
        'd': 'negative',
        'f': 'negative',
        'h': 'positive',
        'n': 'neutral',
        'sa': 'negative',
        'su': 'positive'
    }
map_mesd = {
    'Anger': 'negative',
    'Disgust': 'negative',
    'Fear': 'negative',
    'Happiness': 'positive',
    'Neutral': 'neutral',
    'Sadness': 'negative',

}
map_ruso = {
    'Anger': 'negative',
    'Joy': 'positive',
    'Neutral': 'neutral',
    'Sad': 'negative',
}



# funciones para escribir los csv
def write_tess_csv(data_aug=False):

    X, y = [],[]
    for folder in os.listdir(path_tess):
        if folder != '.DS_Store':
            target = folder.split('_')[1].lower()
            target = map_tess[target]
            for file in os.listdir(os.path.join(path_tess,folder)):
                audio_path = os.path.join(path_tess, folder, file)
                if data_aug == False:
                    result = export_process(audio_path)
                    X.append(result)
                    y.append(target)
                else:
                    results = export_process(audio_path, data_augmentation=True)
                    for result in results:
                        X.append(result)
                        y.append(target)
    df = pd.DataFrame(X)
    df['target'] = y
    df.to_csv('df_tess.csv', index=False)

def write_emodb_csv(data_aug=False):
    X, y = [],[]
    for file in glob.glob(path_emodb + "*.wav"):
        emotion = categories_emo[os.path.basename(file)[5]]
        if emotion != 'boredom':
            target = map_emo[emotion]
            if data_aug == False:
                result = export_process(file)
                X.append(result)
                y.append(target)
            else:
                results = export_process(file, data_augmentation=True)
                for result in results:
                    X.append(result)
                    y.append(target)
    df = pd.DataFrame(X)
    df['target'] = y
    df.to_csv('df_emodb.csv', index=False)

def write_custom_csv(data_aug=False):
    X, y = [],[]
    for file in glob.glob(path_custom_1 + "*.wav"):
        emotion = (file.split('.')[2].split('_')[1])
        emotion = map_custom[emotion]

        if data_aug == False:
            result = export_process(file)
            X.append(result)
            y.append(emotion)
        else:
            results = export_process(file, data_augmentation=True)
            for result in results:
                X.append(result)
                y.append(emotion)
                
    df_custom_train = pd.DataFrame(X)
    df_custom_train['target'] = y

    X_1, y_1 = [],[]
    for file in glob.glob(path_custom_2 + "*.wav"):
        emotion = (file.split('.')[2].split('_')[1])
        emotion = map_custom[emotion]

        if data_aug == False:
            result = export_process(file)
            X.append(result)
            y.append(emotion)
        else:
            results = export_process(file, data_augmentation=True)
            for result in results:
                X.append(result)
                y.append(emotion)

    df_custom_test = pd.DataFrame(X_1)
    df_custom_test['target'] = y_1

    df = pd.concat([df_custom_train, df_custom_test])
    df.to_csv('df_custom.csv', index=False)

def write_ravdess_csv(data_aug=False):
    X, y = [], []

    for actor in os.listdir(path_ravdess):
        if actor != '.DS_Store':
            for file in os.listdir(os.path.join('../data/ravdess/', actor)):
                emotion = map_ravdess[file.split('-')[2]]
                target = map_sent_ravdess[emotion]

                audio_path = os.path.join("../data/ravdess/", actor, file)
                
                if data_aug == False:
                    result = export_process(audio_path)
                    X.append(result)
                    y.append(target)
                else:
                    results = export_process(audio_path, data_augmentation=True)
                    for result in results:
                        X.append(result)
                        y.append(target)

    df_ravdess = pd.DataFrame(X)
    df_ravdess['target'] = y
    df_ravdess.to_csv('df_ravdess.csv', index=False)

def write_savee_csv(data_aug=False):
    X, y = [], []

    for file in os.listdir(path_savee):
        audio_path = os.path.join(path_savee, file)
        emotion = file.split('_')[1][:-6]
        target = map_savee[emotion]
        if data_aug == False:
            result = export_process(audio_path)
            X.append(result)
            y.append(target)
        else:
            results = export_process(audio_path, data_augmentation=True)
            for result in results:
                X.append(result)
                y.append(target)

    df_savee = pd.DataFrame(X)
    df_savee['target'] = y
    df_savee.to_csv('df_savee.csv', index=False)

def write_mesd_csv(data_aug=False):
    X, y = [], []
    for file in os.listdir(path_mesd):
        if file != '.DS_Store':
            if file.split('_')[1] == 'C':
                audio_path = os.path.join(path_mesd, file)
                emotion = file.split('_')[0]
                target = map_mesd[emotion]
                if data_aug == False:
                    result = export_process(audio_path)
                    X.append(result)
                    y.append(target)
                else:
                    results = export_process(audio_path, data_augmentation=True)
                    for result in results:
                        X.append(result)
                        y.append(target)
    df_mesd = pd.DataFrame(X)
    df_mesd['target'] = y
    df_mesd.to_csv('df_mesd.csv', index=False)

def write_ruso_csv(data_aug=False):
    X, y = [], []
    for target in os.listdir(path_ruso):
        if target != '.DS_Store':
            for file in os.listdir(os.path.join(path_ruso, target)):
                audio_path = os.path.join(path_ruso, target, file)
                emotion = map_ruso[target]
                if data_aug == False:
                    result = export_process(audio_path)
                    X.append(result)
                    y.append(emotion)
                else:
                    results = export_process(audio_path, data_augmentation=True)
                    for result in results:
                        X.append(result)
                        y.append(emotion)
            
    df_ruso = pd.DataFrame(X)
    df_ruso['target'] = y
    df_ruso.to_csv('df_ruso.csv', index=False)

def concat_csvs():
    print(os.getcwd())
    # buscar archivos csvs y concatenarlos
    
    csv_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.csv')]

    df_list = []
    for csv_file in csv_files:
        csv_path = os.path.join(os.getcwd(), csv_file)
        df = pd.read_csv(csv_path)
        df_list.append(df)
    df_final = pd.concat(df_list)
    df_final.to_csv('df_final.csv', index=False)


if __name__ == '__main__':
    write_tess_csv(data_aug=data_aug)
    write_emodb_csv(data_aug=data_aug)
    write_custom_csv(data_aug=data_aug)
    write_ravdess_csv(data_aug=data_aug)
    write_savee_csv(data_aug=data_aug)
    
    # write_mesd_csv(data_aug=data_aug)
    write_ruso_csv(data_aug=data_aug)
    # write_mesd_csv(data_aug=True)
    concat_csvs()



