from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import shutil

PATH = "C:\\Users\\User\\Desktop\\Project\\Depression\\FER\\data\\fer2013plus_split\\fer2013subject\\all_images_test"  # Images dir
# Subject dir to save (CREATE FIRST)
SAVE_PATH = "C:\\Users\\User\\Desktop\\Project\\Depression\\FER\\data\\fer2013plus_split\\fer2013subject\\subject"
CLASSES = ["angry", "disgust", "sad", "fear", "surprise", "neutral", "happy"]

FILENAMES = [f for f in os.listdir(
    PATH) if os.path.isfile(os.path.join(PATH, f))]
FILENAMES = [f for f in FILENAMES if f.split(".")[1] == "png"]
print(FILENAMES)


def main():
    # CREATE TEMP DIR first then pass in
    find_face_subject(
        i=50, temp_path='C:\\Users\\User\\Desktop\\Project\\Depression\\FER\\data\\fer2013plus_split\\fer2013subject\\temp')


def find_in(test_path, target_path):

    files = [f for f in os.listdir(
        test_path) if os.path.isfile(os.path.join(test_path, f))]
    model = DeepFace.build_model('Facenet512')
    for num, file in enumerate(files):
        df = DeepFace.find(img_path=PATH+"\\"+file,
                           db_path=target_path, enforce_detection=False, model=model, prog_bar=False)

        array = df.to_numpy()
        if len(array) > 1:
            os.chdir(SAVE_PATH)
            try:
                os.mkdir("person_"+str(num))
            except Exception:
                print("Already exists")

            for i in array:
                print(i[0])  # found file name

                os.chdir(test_path)
                img = cv.imread(i[0])

                os.chdir(SAVE_PATH+"\\person_"+str(num))
                cv.imwrite(file, img)
                # plt.imshow(img)
                # plt.show()


def find_face_subject(i=0, temp_path=None):
    if temp_path == None:
        return print("Please provide temp path")
    model = DeepFace.build_model('Facenet512')
    i = i
    while True:
        print(FILENAMES[0])
        try:
            input_img = cv.imread(FILENAMES[0])  # fer0000053.png
            if not input_img:
                print(FILENAMES[0] + " NOT FOUND (Fixing prob)")
                shutil.copy(PATH+"\\"+FILENAMES[0], FILENAMES[0])
                input_img = cv.imread(FILENAMES[0])
                os.remove(FILENAMES[0])
            df = DeepFace.find(img_path=input_img,
                               db_path=PATH, enforce_detection=False, model_name="Facenet512", model=model, detector_backend="mtcnn", prog_bar=False)

            array = df.to_numpy()
            print(array)
            os.chdir(SAVE_PATH)
            try:
                os.mkdir("person_"+str(i))
            except Exception:
                print("Already exists")

            for x in array:

                print(x[0])  # found dir file name
                filename = x[0][-14:]
                print(filename)

                os.chdir(PATH)
                img = cv.imread(filename)

                os.chdir(SAVE_PATH+"\\person_"+str(i))
                try:
                    cv.imwrite(filename, img)
                    FILENAMES.remove(filename)
                    shutil.move(PATH+"\\"+filename, temp_path+"\\"+filename)
                except Exception:
                    print(
                        f"================= SOMETHING WRONG WHEN SAVING *{filename}* =========================")
        except EOFError:
            print(f"{FILENAMES[0]} IMAGE IS NONE")

        i += 1
        if len(FILENAMES) == 0:
            break


if __name__ == "__main__":
    main()
