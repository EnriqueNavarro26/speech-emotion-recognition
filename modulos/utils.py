import librosa
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def add_noise(data):
    noise_value = 0.015 * np.random.uniform() * np.amax(data)
    data = data + noise_value * np.random.normal(size=data.shape[0])
    return data

def shift_process(data):
    shift_range = int(np.random.uniform(low=-15,high=15) * 1000)
    return np.roll(data,shift_range)

def pitch_process(data,sampling_rate,pitch_factor=0.7):
    return librosa.effects.pitch_shift(data,sampling_rate,pitch_factor)

def change_speed(data, rate = 0.8):
    return librosa.effects.time_stretch(data, rate = rate)

def extract_process(data, sample_rate):
    
    output_result = np.array([])
    mean_zero = np.mean(librosa.feature.zero_crossing_rate(y=data).T,axis=0)
    output_result = np.hstack((output_result,mean_zero))
    
    stft_out = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft_out,sr=sample_rate).T,axis=0)
    output_result = np.hstack((output_result,chroma_stft))
    
    mfcc_out = np.mean(librosa.feature.mfcc(y=data,sr=sample_rate).T,axis=0)
    output_result = np.hstack((output_result,mfcc_out))
    
    root_mean_out = np.mean(librosa.feature.rms(y=data).T,axis=0)
    output_result = np.hstack((output_result,root_mean_out))
    
    mel_spectogram = np.mean(librosa.feature.melspectrogram(y=data,sr=sample_rate).T,axis=0)
    output_result = np.hstack((output_result,mel_spectogram))
    
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft_out, sr=sample_rate).T,axis=0)
    output_result = np.hstack((output_result, contrast))
    
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T,axis=0)
    output_result = np.hstack((output_result, tonnetz))
    
    return output_result

def export_process(path, data_augmentation = False):


    if data_augmentation == False:
        data,sr = librosa.load(path)
        output_1 = extract_process(data,sr)
        result = np.array(output_1)
        return result
    
    data,sr = librosa.load(path)
    
    output_1 = extract_process(data,sr)
    result = np.array(output_1)
    
    # noise_out = add_noise(data)
    # output_2 = extract_process(noise_out,sr)
    # result = np.vstack((result,output_2))
    
    noise_out = add_noise(data)
    speed_out = change_speed(noise_out)
    output_3 = extract_process(speed_out,sr)
    result = np.vstack((result,output_3))
    
    noise_out = add_noise(data)
    shift_out = shift_process(noise_out)
    output_4 = extract_process(shift_out,sr)
    result = np.vstack((result,output_4))
    
    # pitch_out= pitch_process(data, sr)
    # output_5 = extract_process(pitch_out,sr)
    # result = np.vstack((result,output_5))
    
    plus_out = change_speed(data)
    strectch_pitch = pitch_process(plus_out,sr)
    output_6 = extract_process(strectch_pitch,sr)
    result = np.vstack((result,output_6))
    
    return result



def data_preparation(data: pd.core.frame.DataFrame, Y_col: int=-1, 
                    standard_scaler: bool=True, one_hot_encoding: bool=False,
                    train_test: bool=True, split_rate: float=0.8,
                    cv: bool=False, n_fold: int=5,
                    shuffle: bool=True):

                    '''This function prepares the data in such a way that it is ready to train an ML algorithm.
                    Parameters:
                    > data: data as DataFrame
                    > Y_col: column index number of labels (default is -1)
                    > StandardScaler: if True (default) data has to be standardized through the StandardScaler
                    > OneHotEncoder: if True (default is False) labels have to be one-hot-encoded
                    > train_test: if True (default) the dataset has to be split in train and test set
                    > test_size: it determines the size of the train test with respect to the whole dataset (default is 0.8)
                    > cv: if True (default is False) K-Folds cross-validator object is created
                    > n_fold: number of splitting iterations in the cross-validator
                    > shuffle: Whether to shuffle the data during train_test_split and cross_validation (default is True)
                    '''
                    # Initialize variables
                    # Y = data.iloc[:,Y_col].values
                    # X = data.iloc[: ,:-1].values
                    X = data.iloc[:,:-1].values
                    Y = data["target"].values
                    kf = None
                    encoder = None

                    if standard_scaler==True:
                        scaler = StandardScaler()
                        X = scaler.fit_transform(X)

                    if one_hot_encoding==True:
                        encoder = OneHotEncoder()
                        Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
                        dump(encoder, 'checkpoints/encoder.joblib') 

                    if train_test==True:
                        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, 
                                                                            shuffle = shuffle, 
                                                                            train_size = split_rate)
                        if cv==True:
                            kf = KFold(n_splits=n_fold, shuffle=shuffle, random_state=None)
                        
                        pd.DataFrame(x_train).to_csv('checkpoints/x_train.csv', index=False)
                        pd.DataFrame(y_train).to_csv('checkpoints/y_train.csv', index=False)
                        pd.DataFrame(x_test).to_csv('checkpoints/x_test.csv', index=False)
                        pd.DataFrame(y_test).to_csv('checkpoints/y_test.csv', index=False)
                        return x_train, x_test, y_train, y_test, kf
                    
                    else:
                        if cv==True:
                            kf = KFold(n_splits=n_fold, shuffle=shuffle, random_state=None)

                        X.to_csv('out/X.csv', index=False)
                        Y.to_csv('out/Y.csv', index=False)
                        return X, Y, kf
                    
