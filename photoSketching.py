#!/usr/bin/env python
# coding:utf-8
"""
Name : photoSketching.py
Author : Sahil Kumar
Email : sahilofficial74@gmail.com
Time    : 8/21/2021 4:23 PM

"""

import cv2
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog,ttk, messagebox
import scipy
import numpy as np
import random
import math
from sklearn.cluster import KMeans


file_path=None
class Sketch:
    def __init__(self, master):
        self.master = master
        self.master.geometry('1200x675')
        self.master.title("Photo Sketching")

        self.var=StringVar() # text variable for Combobox
        # Create Two Frames
        self.left_frame = Frame(self.master, bg="aquamarine", relief=GROOVE, bd=3)
        self.left_frame.place(x=0, y=0, width=900, height=400)

        
        self.initial_image = Image.open(r"F:\Photo_Sketching\initial_img.png")
        self.initial_image = self.initial_image.resize((720, 400), Image.ANTIALIAS) ## The (720, 400) is (width, height)
        self.img = ImageTk.PhotoImage(self.initial_image)
        self.imgLabel = Label(self.left_frame, borderwidth=0)
        self.imgLabel.image = self.img
        self.imgLabel.configure(image=self.img)
        self.imgLabel.pack()


        self.left_frame2 = Frame(self.master, bg="aquamarine", relief=GROOVE, bd=3)
        self.left_frame2.place(x=0, y=400, width=450, height=274)

        self.left_frame3 = Frame(self.master, bg="aquamarine", relief=GROOVE, bd=3)
        self.left_frame3.place(x=450, y=400, width=450, height=274)

        self.right_frame = Frame(self.master, bg="aquamarine", relief=GROOVE, bd=3)
        self.right_frame.place(x=900, y=0, width=300, height=250)

        self.right_frame2 = Frame(self.master, bg="aquamarine", relief=GROOVE, bd=3)
        self.right_frame2.place(x=900, y=250, width=300, height=425)

        opt = Label(self.right_frame2, text= "Select Option", font=('Helvetica', 16, 'bold'), bg="yellow", fg="black")
        opt.grid(row=0, column=0, padx=5, pady=0, sticky=W)
        option = ttk.Combobox(self.right_frame2, width=35, textvariable=self.var, state="readonly")
        option["values"] = ["Select Option", "Sketch", "Painting"]
        option.current(0)
        option.grid(row=2, column=0, padx=5, pady=20, sticky=W)

        upload_image = Button(self.right_frame, text="Upload", font=('Helvetica', 16, 'bold'),
                              width=15, command=self.upload)
        upload_image.grid(row=3, padx=40, pady=20, sticky=N)

        convert_image = Button(self.right_frame, text="Convert", font=('Helvetica', 16, 'bold'),
                               width=15, command=self.convert)
        convert_image.grid(row=5, padx=40, pady=20, sticky=N)

        exit_image = Button(self.right_frame, text="Exit", font=('Helvetica', 16, 'bold'),
                            width=15, command=quit)
        exit_image.grid(row=7, padx=40, pady=20, sticky=N)


    def upload(self):
        self.imgLabel.pack_forget()
        global file_path
        file_path = filedialog.askopenfilename(title="Select a Image", filetypes=[('Image Files', ('*jpeg', '*png','*jpg'))])
        image = Image.open(file_path)
        image = image.resize((720, 400), Image.ANTIALIAS) ## The (720, 400) is (width, height)
        img = ImageTk.PhotoImage(image)
        self.imgLabel = Label(self.left_frame, borderwidth=0)
        self.imgLabel.image = img
        self.imgLabel.configure(image=img)
        self.imgLabel.pack()
        # print(file_path)
        
    def darkPencilSketch(self):
        img = cv2.imread(file_path)
        dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=64, sigma_r=0.10, shade_factor=0.06)
        image = Image.fromarray(dst_gray)

        # Display sketch
        image = image.resize((448, 274), Image.ANTIALIAS) ## The (448, 274) is (width, height)
        img = ImageTk.PhotoImage(image)
        self.imgLabel = Label(self.left_frame3)
        self.imgLabel.image = img
        self.imgLabel.configure(image=img)
        self.imgLabel.pack()


    def pencilSketch(self):
        img=cv2.imread(file_path)
        grey_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Invert Image
        invert_img=cv2.bitwise_not(grey_img) #invert_img=255-grey_img
        # Blur image
        blur_img=cv2.GaussianBlur(invert_img, (11,11),0)

        # Invert Blurred Image
        invblur_img=cv2.bitwise_not(blur_img)
        #invblur_img=255-blur_img

        # Sketch Image
        sketch_img=cv2.divide(grey_img,invblur_img, scale=256.0)
        image = Image.fromarray(sketch_img)

        # Display sketch
        image = image.resize((448, 274), Image.ANTIALIAS) ## The (448, 274) is (width, height)
        img = ImageTk.PhotoImage(image)
        self.imgLabel = Label(self.left_frame2)
        self.imgLabel.image = img
        self.imgLabel.configure(image=img)
        self.imgLabel.pack()



    def watercolour(self):
        img = cv2.imread(file_path)
        water_img = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
        image = Image.fromarray(water_img)
        image = image.resize((448, 274), Image.ANTIALIAS) ## The (448, 274) is (width, height)
        img = ImageTk.PhotoImage(image)
        self.imgLabel = Label(self.left_frame3)
        self.imgLabel.image = img
        self.imgLabel.configure(image=img)
        self.imgLabel.pack()



    def compute_color_probabilities(self,pixels, palette):
        distances = scipy.spatial.distance.cdist(pixels, palette)
        maxima = np.amax(distances, axis=1)
        distances = maxima[:, None] - distances
        summ = np.sum(distances, 1)
        distances /= summ[:, None]
        return distances

    def get_color_from_prob(self,probabilities, palette):
        probs = np.argsort(probabilities)
        i = probs[-1]
        return palette[i]

    def randomized_grid(self,h, w, scale):
        assert (scale > 0)
        r = scale//2
        grid = []
        for i in range(0, h, scale):
            for j in range(0, w, scale):
                y = random.randint(-r, r) + i
                x = random.randint(-r, r) + j
        grid.append((y % h, x % w))
        random.shuffle(grid)
        return grid

    def get_color_palette(self,img, n=20):
        clt = KMeans(n_clusters=n)
        clt.fit(img.reshape(-1, 3))
        return clt.cluster_centers_
    
    def complement(self,colors):
        return 255 - colors

    def pointillismArt(self):
        
        img = cv2.imread(file_path)
        radius_width = int(math.ceil(max(img.shape) / 1000))
        palette = self.get_color_palette(img, 20)
        complements = self.complement(palette)
        palette = np.vstack((palette, complements))
        canvas = img.copy()
        grid = self.randomized_grid(img.shape[0], img.shape[1], scale=3)
        
        pixel_colors = np.array([img[x[0], x[1]] for x in grid])
        
        color_probabilities = self.compute_color_probabilities(pixel_colors, palette)
        for i, (y, x) in enumerate(grid):
            color = self.get_color_from_prob(color_probabilities[i], palette)
            cv2.ellipse(canvas, (x, y), (radius_width, radius_width), 0, 0, 360, color, -1, cv2.LINE_AA)
        image = Image.fromarray(canvas)
        image = image.resize((448, 274), Image.ANTIALIAS) ## The (448, 274) is (width, height)
        img = ImageTk.PhotoImage(image)
        self.imgLabel = Label(self.left_frame2)
        self.imgLabel.image = img
        self.imgLabel.configure(image=img)
        self.imgLabel.pack()



    def convert(self):
        
        if self.var.get() == "Sketch":
            self.pencilSketch()
            self.darkPencilSketch()

        elif self.var.get() == "Painting":
            self.watercolour()
            #self.pointillismArt()
            

        else:
            messagebox.showerror("Invalid", "Please select a option")



if __name__ == "__main__":
    window = Tk()
    a = Sketch(window)
    window.mainloop()


