import numpy as np
import pandas as pd
import cv2
import timm
from fastai.vision.all import *
import torch.nn.functional as F
import os

if __name__ == '__main__':
    path = os.path.dirname(os.path.realpath(__file__)) + "\\" # This is the path of the project
    path_train_directory = path + "Dev_Phase\\training\\"
    path_test_directory = path + "Test_Phase\\"
    path_train_count = path + "Train_Count\\"
    train_paths = [path + 'Train_Count\\1',
                   path + 'Train_Count\\2',
                   path + 'Train_Count\\3']
    path_test_count = path + "Test_Count\\"
    path_train_videos = path + "Dev_Phase\\training\\videos_train\\"
    path_test_videos = path + "Test_Phase\\videos_test\\"
    if not os.path.exists(path_train_count):
        os.makedirs(path_train_count)
    if not os.path.exists(train_paths[0]):
        os.makedirs(train_paths[0])
    if not os.path.exists(train_paths[1]):
        os.makedirs(train_paths[1])
    if not os.path.exists(train_paths[2]):
        os.makedirs(train_paths[2])
    if not os.path.exists(path_test_count):
        os.makedirs(path_test_count)

    lower = np.array([6, 0, 0], dtype="uint8")
    upper = np.array([60, 255, 255], dtype="uint8")
    # Prepare data use the videos to produce images to be used in training the model and save it in the path_train_count
    # and path_test_count, if you already has produced the images, you set the prepare_data to false
    prepare_data = True
    # Train the model, if you already trained it, you set the train to false
    train = True
    # Produce the output of the model, if you don't need it, set the test to false
    test = True

    if prepare_data:
        print("Preparing the data")

        train_df = pd.read_csv(path_train_directory+"train_set.csv")
        for index, row in train_df.iterrows():
            print("Train record " + str(index))
            video = row['Video']
            start = row['Frame.start']
            end = row['Frame.stop']
            count = row['Pod_count']
            cclass = 3
            if count <= 379:
                cclass = 1
            elif count <= 607:
                cclass = 2

            cam = cv2.VideoCapture(path_train_videos+ str(video) + ".mp4")
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
                        filename = path_train_count + str(cclass) + "\\" + str(
                            video) + "_" + str(currentframe) + ".png"
                        img = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        mask = cv2.inRange(hsv_img, lower, upper)
                        res = cv2.bitwise_and(img, img, mask=mask)
                        cv2.imwrite(filename, res)

                    if currentframe > end:
                        break
                else:
                    break
            # Release all space and windows once done
            cam.release()
            cv2.destroyAllWindows()

        dev = pd.read_csv(path_test_directory+'test_set.csv')
        for index, row in dev.iterrows():
            print("Dev record " + str(index))
            video = row['Video']
            start = row['Frame.start']
            end = row['Frame.stop']
            # print(video)
            cam = cv2.VideoCapture(path_test_videos + str(int(video)) + ".mp4")
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
                        filename = path_test_count + str(video) + "_" + str(currentframe) + ".png"
                        img = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        mask = cv2.inRange(hsv_img, lower, upper)
                        res = cv2.bitwise_and(img, img, mask=mask)
                        cv2.imwrite(filename, res)
                    if currentframe > end:
                        break
                else:
                    break  # Release all space and windows once done
            cam.release()
            cv2.destroyAllWindows()

    if train:
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            get_y=parent_label,
            splitter=RandomSplitter(),
            item_tfms=Resize(256),
            batch_tfms=[
                *aug_transforms(size=(224, 224)),
                Normalize.from_stats(*imagenet_stats)
            ]
        )

        dls = dblock.dataloaders(path_train_count, bs=128)

        learn = vision_learner(
            dls,
            "convnext_small_in22k",
            metrics=accuracy,
            cbs=[
                EarlyStoppingCallback(patience=3),
                SaveModelCallback()
            ]
        ).to_fp16()

        learn.fine_tune(50, freeze_epochs=5)
        learn.export(path + "model_count.pth")
        interp = ClassificationInterpretation.from_learner(learn)
        interp.plot_confusion_matrix()
    if test:
        learn = load_learner(path + "model_count.pth")
        test_files = os.listdir(path_test_count)
        test_image_filepaths = [path_test_count+f for f in test_files]
        test_dl = learn.dls.test_dl(test_image_filepaths)
        preds, _ = learn.get_preds(dl=test_dl)
        #print('Vocabulary for estimates')
        #print(dls.vocab[[0, 1, 2]])
        test_df = pd.DataFrame(test_image_filepaths, columns=['Files'])
        test_df['c1'] = preds[:, 0]
        test_df['c2'] = preds[:, 1]
        test_df['c3'] = preds[:, 2]
        test_df.to_csv(path + "Test_Count_Estimates.csv")

        train_image_filepaths = [train_paths[0] + "\\" + f for f in os.listdir(train_paths[0])] + [
            train_paths[1] + "\\" + f for f in os.listdir(train_paths[1])] + [train_paths[2] + "\\" + f for f in
                                                                              os.listdir(train_paths[2])]
        train_dl = learn.dls.test_dl(train_image_filepaths)
        preds, _ = learn.get_preds(dl=train_dl)
        # print(dls.vocab[[0, 1, 2]])

        train_df = pd.DataFrame(train_image_filepaths, columns=['Files'])
        train_df['c1'] = preds[:, 0]
        train_df['c2'] = preds[:, 1]
        train_df['c3'] = preds[:, 2]
        train_df.to_csv(path + "Train_Count_Estimates.csv")






