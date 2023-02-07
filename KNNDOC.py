import sys
import os
import urllib.request
import zipfile
import cv2 as cv
import pandas as pd
import torch
from img2vec_pytorch import Img2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

#LEARNING PHASE

def ModelTeaching():
    rawImages = []
    features = []
    labels = []

    for f in os.listdir(r'tmp_imgs/train'):
        image = cv.imread(r'tmp_imgs/train/' + f)
        label = f.split(os.path.sep)[-1].split(".")[
            0]
        pixels = cv.resize(image, (32, 32)).flatten()


        rawImages.append(image)
        features.append(pixels)
        labels.append(label)

    trainLables = labels[:20001]
    testLables  = labels[20001:]
    trainFeatures = features[:20001]
    testFeatures  = features[20001:]
    trainImages = rawImages[:20001]
    testImages  = rawImages[20001:]



    img2vec = Img2Vec()

    vecs = []
    for img in rawImages:
      vecs.append( img2vec.get_vec( Image.fromarray(img) ))

    trainFeatures = vecs[:20001]
    testFeatures = vecs[20001:]
    model = KNeighborsClassifier(n_neighbors=50)
    model.fit(trainFeatures, trainLables)
    prediction = model.predict(testFeatures)
    print(accuracy_score(y_true=testLables, y_pred=prediction))
    torch.save(model, 'model.pt')



def ModelReady():
    rawImages = []
    labels = []
    for f in os.listdir(r'predictpic/'):
        image = cv.imread(r'predictpic/' + f)
        label = f.split(os.path.sep)[-1].split(".")[
            0]

    rawImages.append(image)
    labels.append(label)
    img2vec = Img2Vec()

    vecs = []
    for img in rawImages:
      vecs.append( img2vec.get_vec( Image.fromarray(img) ))

    testFeatures  = vecs[0]
    testFeatures=testFeatures.reshape(1,-1)
    model=torch.load('model.pt')
    with torch.no_grad():
        output = model.predict(testFeatures)


    if output[0] == 'cat':
        return 'cat';
    if output[0] == 'dog':
        return 'dog';



#MAIN
import customtkinter
from tkinter import messagebox

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root=customtkinter.CTk()
root.geometry("500x500")

def AI():
    animal=ModelReady()
    return animal


frame= customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand= True)

label=customtkinter.CTkLabel(master=frame, text="Dog or Cat?!", font=("Roboto", 24))
label.pack(pady=12, padx=10)

label2=customtkinter.CTkLabel(master=frame, text="This is an AI, which tells you if it is a dog, or a cat!", font=("Roboto", 20))
label.pack(pady=32, padx=30)


def select_file():
    file_path = customtkinter.filedialog.askopenfilename(initialdir = "/", title = "Select file", filetypes = (("PNG files", "*.png"), ("JPEG files", "*.jpg")))
    with open(file_path, 'rb') as f:
        data = f.read()
    save_file(data)


    animal= ModelReady()
    if animal == 'dog':
        messagebox.showinfo("Success", "Successful! It's a dog! ")
    if animal== 'cat':
        messagebox.showinfo("Success", "Successful! It's a cat! ")

def save_file(data):
    file_path = r"predictpic/image.jpg"
    with open(file_path, 'wb') as f:
        f.write(data)


select_file_button = customtkinter.CTkButton(frame, text="Choose a picture", command=select_file)
select_file_button.pack(pady=50, padx=50)


root.mainloop()