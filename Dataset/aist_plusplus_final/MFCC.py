import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm, trange
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import speechpy


def getMFCCVector(fileName,path='./audio',startTime=0,duration=4,fps=60,sr=None,):
    '''
    获取论文中MFCC的26维变量，分别是13的mfcc和13的衍生变量
    :param fileName: 音频文件名
    :param path: 音频文件路径，不包括文件部分
    :param startTime: 开始时间点 秒
    :param duration:  持续时间 秒
    :param fps: 帧率
    :param sr: 采样率
    :return: mfccs array 每帧26
    '''
    for audiofile in os.listdir(path):
        if audiofile != fileName:
            continue
        else:
            start = startTime
            duration = duration
            fps = fps
            stop = start + duration
            y, sr = librosa.load(path + "/" + audiofile, sr=None, offset=start,
                                 duration=duration)  # sr是采样率,None指原生采样率

            '''
            分帧就是将原始语音信号分成大小固定的N段语音信号，这里每一段语音信号都被称为一帧，
            帧长一般取10到30ms。分帧一般采用交叠分段的方法，是为了使帧与帧之间平滑过渡
            保持其连续性。前一帧和后一帧的交叠部分称为帧移，帧移与帧长的比值一般取为0-1/2。
            '''
            y, sr
            audio_dist = y
            frame_shift = int(sr / (fps + 4))  # 帧移，保证60个都有正常数据 4代表着framelength和shift之间的比值
            frame_length = frame_shift * 2  # 帧长
            hop_length = int(frame_shift)  # frame_shift * sr窗口每部滑动距离（跳长）
            win_length = int(frame_length)  # 窗口样本
            '''
            plt.figure()
            librosa.display.waveplot(audio_dist,sr)
            #librosa.display.waveplot(audio_dist,sr)
            '''
            mfccs_flag = False
            mfccs = np.array([])
            for i in range(fps * duration):
                '''
                mfcc返回 (n_mfcc,a)的向量，a=ceil(autioframelength / 帧移)
                '''
                mfccs_tmp = librosa.feature.mfcc(y=audio_dist[i * frame_shift:i * frame_shift + frame_length],
                                                 sr=sr,
                                                 n_mfcc=13,
                                                 hop_length=frame_length + 1)  # 因为ceil的存在所以为了保证每次的计算纬度都为1所以就+1

                derivative = speechpy.feature.extract_derivative_feature(mfccs_tmp)
                derivative = derivative[:, :, 1]  # get 1 derivative
                mfccs_tmp = np.append(mfccs_tmp, derivative)

                if mfccs_flag == False:
                    mfccs = mfccs_tmp
                    mfccs_flag = True
                else:
                    mfccs = np.vstack((mfccs, mfccs_tmp))

            return mfccs


