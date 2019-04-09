from tkinter import *
import cv2
from PIL import Image, ImageTk
from tkinter import ttk
import numpy as np
import time


cap = cv2.VideoCapture(0)
root = Tk()
root.title("WEBCAM FILTERS")
topFrame = Frame(root).pack()

def screenshoot():
    time.sleep(0.1)  # If you don't wait, the image will be dark
    return_value, frame = cap.read()
    cv2.imwrite("opencv.png", frame)

button6 = Button(topFrame, text="SNAPSHOT", command = lambda: screenshoot() )
photox = PhotoImage(file="screen.png")
button6.config(image=photox)
button6.pack(fill=X)

middleFrame = Frame(root).pack(side=BOTTOM)
lmain = Label(middleFrame, text="description")
lmain.pack()

job = None


def cancel():
    global job
    if job is not None:
        lmain.after_cancel(job)
        job = None

def show():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    global job
    job = lmain.after(10, show)

def noFilter():
    cancel()
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    blur = cv2.detailEnhance(cv2image, sigma_s=10, sigma_r=0.15)
    img = Image.fromarray(blur)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    global job
    job = lmain.after(10, noFilter)


def blueFilter():
    cancel()
    ret, frame = cap.read()
    dstImage = frame
    cv2.applyColorMap(frame, cv2.COLORMAP_PINK, dstImage)

    one = dstImage

    img = Image.fromarray(one)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    global job
    job = lmain.after(10, blueFilter)



# fix this so it is red
def redFilter():
    cancel()
    _, frame = cap.read()
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = 0
    hsv[:, :, 2] = 0
    img = hsv

    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    global job
    job = lmain.after(10, redFilter)


def cartoon():
    cancel()
    ret, frame = cap.read()
    numDownSamples = 2  # number of downscaling steps
    numBilateralFilters = 10  # number of bilateral filtering steps

    img_color = frame
    for i in range(numDownSamples):
        img_color = cv2.pyrDown(img_color)

    for x in range(numBilateralFilters):
        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

    for w in range(numDownSamples):
        img_color = cv2.pyrUp(img_color)

    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 9, 2)

    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    one = cv2.bitwise_and(img_color, img_edge)

    img = Image.fromarray(one)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    global job
    job = lmain.after(10, cartoon)


def manyColors():
    cancel()
    ret, frame = cap.read()
    r = frame[:, :, 0].copy()
    g = frame[:, :, 1].copy()
    b = frame[:, :, 2].copy()
    frame[:, :, 0] = np.minimum(255, np.abs(r - g - b) * 3 / 2)
    frame[:, :, 1] = np.minimum(255, np.abs(g - b - r) * 3 / 2)
    frame[:, :, 2] = np.minimum(255, np.abs(b - r - g) * 3 / 2)
    one = frame

    img = Image.fromarray(one)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    global job
    job = lmain.after(10, manyColors)


def blackWhite():
    cancel()
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    sigma = 0.33
    v = np.median(gray)  # calcula do valor medio da matriz em tons de cinza
    # aplica a deteccao de bordas por Canny utilizando a media
    minimo = int(max(0, (1.0 - sigma) * v))  # calcula o valor minimo
    maximo = int(min(255, (1.0 + sigma) * v))  # calcula o valor maximo
    borda = cv2.Canny(gray, minimo, maximo)  # realiza a operacao de deteccao de bordas
    img = borda

    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    global job
    job = lmain.after(10, blackWhite)


def motionBlur():
    cancel()
    ret, frame = cap.read()
    size = 50
    kernel_motion = np.zeros((size, size))
    kernel_motion[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion = kernel_motion / size

    blur = cv2.filter2D(frame, -1, kernel_motion)

    img = Image.fromarray(blur)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    global job
    job = lmain.after(10, motionBlur)


def stuff():
    cancel()
    ret, img = cap.read()
    red = 250
    green = 88
    blue = 244
    red = min(max(0, red), 255)
    green = min(max(0, green), 255)
    blue = min(max(0, blue), 255)

    gray_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.uint32)
    img[:, :, 0] = red * gray_img / 255
    img[:, :, 1] = green * gray_img / 255
    img[:, :, 2] = blue * gray_img / 255
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    global job
    job = lmain.after(10, stuff)


def show_frame(x):
    if x is "nofilter":
        noFilter()

    elif x is "cartoon":
        cartoon()

    elif x is "blue":
        blueFilter()

    elif x is "manyColors":
        manyColors()

    elif x is "blur":
        motionBlur()

    elif x is "bw":
        blackWhite()

    elif x is "red":
        redFilter()



bottomFrame = Frame(root).pack(side=BOTTOM)
bottomFrame2 = Frame(root).pack(side=TOP)

photo0 = PhotoImage(file="filters.png")
label = Label(bottomFrame, image=photo0)
label.photo0 = photo0
label.pack(side=LEFT)


button1 = ttk.Button(bottomFrame, command=lambda: noFilter())  # blur filter
photo = PhotoImage(file="blur.png")
button1.config(image=photo)

button2 = ttk.Button(bottomFrame, command=lambda: cartoon())  # cartoon filter
photo2 = PhotoImage(file="cartoon.png")
button2.config(image=photo2)

button3 = ttk.Button(bottomFrame, command=lambda: motionBlur())  # cartoon filter
photo3 = PhotoImage(file="motion.png")
button3.config(image=photo3)

button4 = ttk.Button(bottomFrame, command=lambda: blueFilter()) # cartoon filter
photo4 = PhotoImage(file="blue.png")
button4.config(image=photo4)

button5 = ttk.Button(bottomFrame2, command= lambda: blackWhite())  # cartoon filter
photo5 = PhotoImage(file="b&w.png")
button5.config(image=photo5)

button7 = ttk.Button(bottomFrame2, command=lambda: redFilter())  # cartoon filter
photo7 = PhotoImage(file="red.png")
button7.config(image=photo7)

button8 = ttk.Button(bottomFrame2, command=lambda: manyColors())  # cartoon filter
photo8 = PhotoImage(file="colorful.png")
button8.config(image=photo8)

button9 = ttk.Button(bottomFrame2,command=lambda: stuff())
photo9= PhotoImage(file="purple.png")
button9.config(image=photo9)


button1.pack(side=LEFT)
button2.pack(side=LEFT)
button3.pack(side = LEFT)
button4.pack(side = LEFT)
button5.pack(side = LEFT)
button7.pack(side = LEFT)
button8.pack(side =LEFT)
button9.pack()



root.mainloop()
