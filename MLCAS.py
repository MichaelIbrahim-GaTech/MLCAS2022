import numpy as np
import pandas as pd
import imutils
from sklearn.linear_model import LinearRegression
import json
import cv2
import os

from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

if __name__ == '__main__':
    path = os.path.dirname(os.path.realpath(__file__)) + "\\" # This is the path of the project
    path_train_directory = path + "Dev_Phase\\training\\"
    path_test_directory = path + "Test_Phase\\"
    path_train_videos = path + "Dev_Phase\\training\\videos_train\\"
    path_test_videos = path + "Test_Phase\\videos_test\\"
    path_pod_data = path + "Dev_Phase\\training\\pod_annotations\\"
    # Prepare the training features and save them in file Train_with_features.csv, if you already produced this file,
    # you can set calculate_features to false
    calculate_features = True
    # Prepare the test set features and save them in file Test_with_features.csv, if you already produced this file,
    # you can set calculate_dev_features to false
    calculate_dev_features = True
    lower = np.array([6, 0, 0], dtype="uint8")
    upper = np.array([60, 255, 255], dtype="uint8")
    TrainFeatures = ['length','Contours','mask_average','Diverse','Elite','c1','c2']

    df = pd.read_csv(path_pod_data + 'pod_detection_annotations.csv')
    df['region_shape_attributes'] = df['region_shape_attributes'].str.replace('\"\"','\"')
    all_images = df['filename'].unique()
    df = df[df['region_shape_attributes'].map(len) > 5]
    used_images = df['filename'].unique()
    mdict = df.groupby('filename')['filename'].count().to_dict()
    NewDF = pd.DataFrame()
    NewDF['images'] = mdict.keys()
    NewDF['count'] = mdict.values()
    NewDF['video'] = 0
    NewDF['frame'] = 0
    for index, row in NewDF.iterrows():
        tokens = row['images'].replace(".png","").split('_')
        NewDF.at[index,'video'] = int(tokens[1])
        NewDF.at[index,'frame'] = int(tokens[2])

    minarea = 1000000
    maxarea = 0
    for index, row in df.iterrows():
        token = json.loads(row['region_shape_attributes'])
        area = token['width']*token['height']
        if area > maxarea:
            maxarea = area
        if area < minarea:
            minarea = area

    print("Preparing Train Features")
    if calculate_features:
        train_ancestery_df = pd.read_csv(path + "Train_Ancestery_Estimates.csv")
        train_ancestery = {}
        for index, row in train_ancestery_df.iterrows():
            tokens = row['Files'].replace(".png","").split('\\')
            frame = tokens[-1]
            train_ancestery[frame] = [row['Diverse'], row['Elite'], row['PI']]
        train_count_df = pd.read_csv(path + "Train_Count_Estimates.csv")
        train_count = {}
        for index, row in train_count_df.iterrows():
            tokens = row['Files'].replace(".png","").split('\\')
            frame = tokens[-1]
            train_count[frame] = [row['c1'], row['c2'], row['c3']]

        train = pd.read_csv(path_train_directory + "train_set.csv")
        train['length'] = train['Frame.stop'] - train['Frame.start']
        train['mask_average'] = 0.0
        train['Diverse'] = 0.0
        train['Elite'] = 0.0
        train['PI'] = 0.0
        train['Contours'] = 0.0
        train['c1'] = 0.0
        train['c2'] = 0.0
        train['c3'] = 0.0
        train['count'] = 0
        for index, row in train.iterrows():
            print("Train record " + str(index))
            video = row['Video']
            start = row['Frame.start']
            end = row['Frame.stop']
            for index2, row2 in NewDF.iterrows():
                if row2['video'] == video and row2['frame'] >= start and row2['frame'] <= end:
                    train.at[index, 'sample'] = row2['count']

            cam = cv2.VideoCapture(path_train_videos+str(video)+".mp4")
            images = []

            # frame
            currentframe = 0
            while (True):
                 # reading from frame
                ret, frame = cam.read()
                if ret:
                    # increasing counter so that it will
                    # show how many frames are created
                    currentframe += 1
                    if currentframe >= start and currentframe <= end:
                        scale_percent = 50  # percent of original size
                        width = int(frame.shape[1] * scale_percent / 100)
                        height = int(frame.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        images.append(cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
                    if currentframe > end:
                        break
                else:
                    break
            # Release all space and windows once done
            cam.release()
            cv2.destroyAllWindows()
            mask_average = 0.0
            Diverse = 0.0
            Elite = 0.0
            PI = 0.0
            c1 = 0.0
            c2 = 0.0
            c3 = 0.0
            Contours = 0.0
            for i in range(len(images)):
                Diverse += train_ancestery[str(video) + "_" + str(int(start + i))][0]
                Elite += train_ancestery[str(video) + "_" + str(int(start + i))][1]
                PI += train_ancestery[str(video) + "_" + str(int(start + i))][2]
                c1 += train_count[str(video) + "_" + str(int(start + i))][0]
                c2 += train_count[str(video) + "_" + str(int(start + i))][1]
                c3 += train_count[str(video) + "_" + str(int(start + i))][2]
                img = images[i]
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_img, lower, upper)
                mask_average += np.mean(mask)
                res = cv2.bitwise_and(img, img, mask=mask)
                # Grayscale
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                # Find Canny edges
                edged = cv2.Canny(gray, 30, 200)
                contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > minarea and area < maxarea:
                        Contours += 1
            mask_average /= len(images)
            Diverse /= len(images)
            Elite /= len(images)
            PI /= len(images)
            c1 /= len(images)
            c2 /= len(images)
            c3 /= len(images)
            Contours /= len(images)
            train.at[index,'Diverse'] = Diverse
            train.at[index,'Elite'] = Elite
            train.at[index,'PI'] = PI
            train.at[index,'c1'] = c1
            train.at[index,'c2'] = c2
            train.at[index,'c3'] = c3
            train.at[index,'mask_average'] = mask_average
            train.at[index,'Contours'] = Contours

        train.to_csv(path + "Train_with_features.csv")
    else:
        train = pd.read_csv(path + "Train_with_features.csv")

    # Train the model
    print("Training linear regression model")
    model = LinearRegression()
    model.fit(train[TrainFeatures],train['Pod_count'])

    print("Preparing Test Features")
    if calculate_dev_features:
        dev_ancestery_df = pd.read_csv(path + "Test_Ancestery_Estimates.csv")
        dev_ancestery = {}
        for index, row in dev_ancestery_df.iterrows():
            tokens = row['Files'].replace(".png","").split('\\')
            frame = tokens[-1]
            dev_ancestery[frame] = [row['Diverse'], row['Elite'], row['PI']]
        dev_count_df = pd.read_csv(path + "Test_Count_Estimates.csv")
        dev_count = {}
        for index, row in dev_count_df.iterrows():
            tokens = row['Files'].replace(".png","").split('\\')
            frame = tokens[-1]
            dev_count[frame] = [row['c1'], row['c2'], row['c3']]

        dev = pd.read_csv(path_test_directory + 'test_set.csv')
        dev['length'] = dev['Frame.stop'] - dev['Frame.start']
        dev['mask_average'] = 0.0
        dev['Diverse'] = 0.0
        dev['Elite'] = 0.0
        dev['PI'] = 0.0
        dev['c1'] = 0.0
        dev['c2'] = 0.0
        dev['c3'] = 0.0
        dev['Contours'] = 0.0
        dev['count'] = 0
        for index, row in dev.iterrows():
            print("Dev record " + str(index))
            video = row['Video']
            start = row['Frame.start']
            end = row['Frame.stop']
            for index2, row2 in NewDF.iterrows():
                if row2['video'] == video and row2['frame'] >= start and row2['frame'] <= end:
                    dev.at[index, 'sample'] = row2['count']

            cam = cv2.VideoCapture(path_test_videos+str(int(video))+".mp4")
            images = []
            # frame
            currentframe = 0
            while (True):
                # reading from frame
                ret, frame = cam.read()
                if ret:
                    # increasing counter so that it will
                    # show how many frames are created
                    currentframe += 1
                    if currentframe >= start and currentframe <= end:
                        images.append(cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
                    if currentframe > end:
                        break
                else:
                    break# Release all space and windows once done
            cam.release()
            cv2.destroyAllWindows()
            mask_average = 0.0
            Diverse = 0.0
            Elite = 0.0
            PI = 0.0
            c1 = 0.0
            c2 = 0.0
            c3 = 0.0
            Contours = 0.0
            for i in range(len(images)):
                Diverse += dev_ancestery[str(int(video)) + "_" + str(int(start + i))][0]
                Elite += dev_ancestery[str(int(video)) + "_" + str(int(start + i))][1]
                PI += dev_ancestery[str(int(video)) + "_" + str(int(start + i))][2]
                c1 += dev_count[str(int(video)) + "_" + str(int(start + i))][0]
                c2 += dev_count[str(int(video)) + "_" + str(int(start + i))][1]
                c3 += dev_count[str(int(video)) + "_" + str(int(start + i))][2]
                img = images[i]
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_img, lower, upper)
                mask_average += np.mean(mask)
                res = cv2.bitwise_and(img, img, mask=mask)
                # Grayscale
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                # Find Canny edges
                edged = cv2.Canny(gray, 30, 200)
                contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > minarea and area < maxarea:
                        Contours += 1
            mask_average /= len(images)
            Diverse /= len(images)
            Elite /= len(images)
            PI /= len(images)
            c1 /= len(images)
            c2 /= len(images)
            c3 /= len(images)
            Contours /= len(images)
            dev.at[index,'Diverse'] = Diverse
            dev.at[index,'Elite'] = Elite
            dev.at[index,'PI'] = PI
            dev.at[index,'c1'] = c1
            dev.at[index,'c2'] = c2
            dev.at[index,'c3'] = c3
            dev.at[index,'mask_average'] = mask_average
            dev.at[index,'Contours'] = Contours

        dev['sample'] = model.predict(dev[TrainFeatures])
        dev.to_csv(path + "Test_with_features.csv")
    else:
        dev = pd.read_csv(path + "Test_with_features.csv")

    dev['sample'] = model.predict(dev[TrainFeatures])


    for index, row in dev.iterrows():
        video = row['Video']
        start = row['Frame.start']
        end = row['Frame.stop']
        for index2, row2 in NewDF.iterrows():
            if row2['video'] == video and row2['frame'] >= start and row2['frame'] <= end:
                sub2 = 464.3059 + 1.2521 * row2['count']
                dev.at[index,'sample'] = sub2
    sub = np.full((45,1),0.0)
    for i in range(45):
        sub[i] = round((dev.loc[2*i,'sample'] + dev.loc[2*i+1,'sample']) / 2)

    sub.T.tofile(path + "submission.csv",sep='\n')
