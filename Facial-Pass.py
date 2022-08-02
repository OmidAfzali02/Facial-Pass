from keras.models import load_model
import numpy as np
import cv2
from tkinter import filedialog
import tensorflow as tf
from os import environ

# to prevent any warning from tf
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    model = load_model('complete_saved_model4/')
except FileNotFoundError:
    print("It seems you do not have the model files.")
    print("Make sure you have the complete_saved_model4 on your directory and rerun the program.")

modes = ['q', '0', '1']

while True:
    def main():
        print("Initializing Facial Pass\n")
        print("0- from an Image")
        print("1- from webcam")
        print("q- Quit")

        mode = input("\nSelect preferred mode:  ")

        if mode not in modes:
            print("\nSelect an available mode!! ")
        else:
            return mode.lower()


    def password(mode, source):
        if mode == 0:
            print("Image Loading...")
            img = tf.keras.preprocessing.image.load_img(source, target_size=(360, 640))
            plt.imshow(img)
            X = tf.keras.preprocessing.image.img_to_array(img)
            X = np.expand_dims(X, axis=0)
            images = np.vstack([X])
            val = model.predict(images)
            if val == 1:
                print("Access Granted!")
            elif val == 0:
                print("Access Denied!")

        elif mode == 1:
            source.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
            source.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
            while True:
                successful_frame_read, frame = source.read()

                cv2.imshow(f"Facial Pass", frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                key = cv2.waitKey(1)
                X = np.expand_dims(frame, axis=0)
                frame_arr = np.vstack([X])
                val = model.predict(frame_arr)
                print(val)

                if key == 81 or key == 113:
                    break
                elif val == 1:
                    print("Access Granted!")
                    break
                elif val == 0:
                    print("Access Denied!")

            source.release()
            cv2.destroyAllWindows()
        else:
            print("not a valid mode")


    mode = main()

    if mode == "0":
        print("Select Image file:  ")
        img = filedialog.askopenfilename()
        password(0, img)

    elif mode == "1":
        print("Webcam Starting...")
        source = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # to prevent [ WARN:1@37.534]
        password(1, source)

    elif mode == "q":
        break
