#Set-ExecutionPolicy Unrestricted -Scope Process
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from functools import partial
from tkinter.filedialog import askopenfile, askopenfilename
from keras.models import load_model, Model
import cv2
import numpy as np
from keras import backend as K
import time
import joblib
from PIL import Image, ImageTk

from utils.load_data import load_single_image

# creating main application window
root = tk.Tk()
root.geometry("720x720") # size of the top_frame
root.title("Brain Tumor Classifier")



#  Frame ###########
top_frame = Frame(root, bd = 10)
top_frame.pack()

middle_frame = Frame(root, bd =10)
middle_frame.pack()

bottom_frame = Frame(root, bd = 10)
bottom_frame.pack()

notification_frame = Frame(root, bd = 10)
notification_frame.pack()

"""User Defined Function"""

# open a h5/pkl file from hard-disk
def open_file(initialdir='/'):

    file_path  = askopenfilename(initialdir=initialdir, filetypes = [ ('Model Weights', '*.h5' ),('Model Weights', '*.pkl' ) ]  )
    if file_path[-3:] == '.h5':
        model_var.set(file_path)
        load_cnn_weights()
    else:
        model_var.set(file_path)
        load_ml_weights()

def load_cnn_weights():
    global model
    weight_path = h5_entry.get()
    model = load_model(weight_path)

    
def load_ml_weights():
    global model
    weight_path = h5_entry.get()
    model = joblib.load(weight_path)

# open a image file from hard-disk
def open_image(initialdir='/'):
    file_path  = askopenfilename(initialdir=initialdir, filetypes = [ ('Image File', '*.*' ) ]  )
    img_var.set(file_path)

    image = Image.open(file_path)
    image = image.resize((240,240))
    photo = ImageTk.PhotoImage(image)

    img_label = Label(middle_frame, image=photo, padx=10, pady=10)
    img_label.image = photo
    img_label.grid(row=3, column=1)

    load_image()
    return file_path

def load_image():
    path = img_entry.get()
    global imgs
    imgs = load_single_image(path)
    # print(imgs.shape)
    imgs = imgs.reshape(1, 240, 240, 1).astype('float32')
    # print(imgs.shape)
    return

# #####################  Test Image
def test_image():
    start_time = time.time()
    result = model.predict(imgs)
    # print(model.predict_proba(imgs))
    if result > 0.5:
        prediction = "tumor"
    else:
        prediction = "no tumor"
    print(f'Prediction: {result}')
    print(f'Prediction time:{time.time()-start_time}')
    result_text = "output class: "+ prediction
    test_result_var.set(result_text)


"""  Top Frame  """
# ##### H5 #################
btn_h5_fopen = Button(top_frame, text='Browse Model',  command =  lambda: open_file(h5_entry.get()), bg="black", fg="white" )
btn_h5_fopen.grid(row=2, column=1)

model_var = StringVar()
model_var.set("/")
h5_entry = Entry(top_frame, textvariable=model_var, width=80)
h5_entry.grid(row=2, column=2)


#######   IMAGE input
btn_img_fopen = Button(top_frame, text='Browse Image',  command =  lambda: open_image(img_entry.get()), bg="black", fg="white" )
btn_img_fopen.grid(row=7, column=1)

img_var = StringVar()
img_var.set("/")
img_entry = Entry(top_frame, textvariable=img_var, width=80)
img_entry.grid(row=7, column=2)


""" middle Frame  """
ml = Label(middle_frame, font=("Courier", 10),bg="gray", fg="white", text="Loaded Image Shown Below").grid(row=1, column=1)

####### Have Image show properly here in grid

""" bottom Frame  """

# Test Image butttom
btn_test = Button(bottom_frame, text='Test Image',  command = test_image , bg="green", fg="white" )
btn_test.pack()


test_result_var = StringVar()
test_result_var.set("Your result will be shown here")
test_result_label = Label(bottom_frame,font=("Courier", 20), height=3, textvariable=test_result_var, bg="white", fg="purple").pack()

# Entering the event mainloop
top_frame.mainloop()
print("finished")