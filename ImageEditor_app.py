import math
import tkinter as tk
from tkinter import filedialog, Menu
from PIL import Image, ImageTk, ImageOps
import numpy as np
from skimage.util import view_as_windows

""" Load image function """
def load_image():
    # filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.bmp;*.jpg"), ("BMP Files", "*.bmp"), ("JPG Files", "*.jpg")])
    # if filepath:
    if True:
        global img, img_original, img_tk, img_tk_filter, img_np, img_filter, img_filter_np
        #img = Image.open(filepath)
        img = Image.open(r"images\image.jpg")
        img_original = img.copy()  # Copy of the original image for backup
        img_filter = img.copy()  # Copy of the image for applying filters

        img_np = np.array(img)
        img_filter_np = np.array(img_filter)

        # Tworzenie obiektu ImageTk dla obu pól
        img_tk = ImageTk.PhotoImage(img)
        img_tk_filter = ImageTk.PhotoImage(img_filter)

        # Wyświetlanie obrazów na obu kanwach
        canvas1.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas2.create_image(0, 0, anchor=tk.NW, image=img_tk_filter)


""" Function to aplly filters """
def apply_filters():
    global img_filter, img_filter_np, img_tk_filter

    #we start from original image
    img_filter = img_original.copy()
    img_filter_np = np.array(img_filter)

    # apply each filter from the stack
    for filter_func in img_filter_stack:
        img_filter_np = filter_func(img_filter_np)

    # update canva
    img_filter = Image.fromarray(img_filter_np)
    img_tk_filter = ImageTk.PhotoImage(img_filter)
    canvas2.create_image(0, 0, anchor=tk.NW, image=img_tk_filter)
    canvas2.image = img_tk_filter


""" 
FILTER FUNCTIONS
input: np.array of image
output: np.array of image
"""
def to_grayscale(np_image):
    gray_np = np.dot(img_filter_np[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return gray_np

def to_negative(np_image):
    return 255 - np_image

def to_binary(np_image):
    threshold = 128
    if grayscale_var.get():
        binary_np = np.where(np_image>threshold, 255, 0).astype(np.uint8)
    else:
        gray_np = np.dot(img_filter_np[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        binary_np = np.where(gray_np>threshold, 255, 0).astype(np.uint8)
    return binary_np

def adjust_brightness(np_image):
    try:
        brightness = int(entry_var_brightness.get())
        if brightness < -100 or brightness > 100:
            print("Proszę podać wartość jasności w zakresie -100 do 100.")
            return np_image
    except ValueError:
        print("Proszę podać poprawną liczbę dla jasności.")
        return np_image

    adjusted_image = np.clip(np_image + brightness, 0, 255)

    return adjusted_image.astype(np.uint8)

def adjust_contrast(np_image):
    try:
        contrast = int(entry_var_contrast.get())
        if contrast < -100 or contrast > 100:
            print("Proszę podać wartość kontrastu w zakresie -100 do 100.")
            return np_image
    except ValueError:
        print("Proszę podać poprawną liczbę dla kontrastu.")
        return np_image

    factor = 1 + (contrast / 100.0)

    adjusted_image = np.clip(128 + (np_image - 128) * factor, 0, 255)

    return adjusted_image.astype(np.uint8)


def to_averaging(np_image):
    min_size = min(np_image.shape[0], np_image.shape[1])

    kernel_size = max(3, min(int(entry_var_avg.get()) // 10 * 2 + 1, min_size - 1))
    pad = kernel_size // 2

    is_grayscale = len(np_image.shape) == 2
    if is_grayscale:
        np_image = np.expand_dims(np_image, axis=-1)

    padded_image = np.pad(np_image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    if kernel_size > min_size - 1:
        print(f"Zbyt duży rozmiar jądra: {kernel_size}. Dopasowano do {min_size - 1}.")
        kernel_size = min_size - 1

    if is_grayscale:
        windows = view_as_windows(padded_image, (kernel_size, kernel_size, 1))
    else:
        windows = view_as_windows(padded_image, (kernel_size, kernel_size, 3))

    averaged_image = np.zeros_like(np_image)

    for k in range(averaged_image.shape[2]):
        for i in range(averaged_image.shape[0]):
            for j in range(averaged_image.shape[1]):
                averaged_image[i, j, k] = np.mean(windows[i, j, :, :, k])

    if is_grayscale:
        averaged_image = averaged_image.squeeze()

    return averaged_image.astype(np.uint8)

def to_gaussian(np_image):
    try:
        sigma = float(entry_var_gauss.get())
        if sigma <= 0 or sigma > 100:
            print("Proszę podać wartość sigma w zakresie 0-100.")
            return np_image
    except ValueError:
        print("Proszę podać poprawną liczbę dla sigmy.")
        return np_image

    kernel_size = max(3, int(sigma // 10) * 2 + 1)
    pad = kernel_size // 2

    is_grayscale = len(np_image.shape) == 2
    if is_grayscale:
        np_image = np.expand_dims(np_image, axis=-1)

    # Stwórz jądro Gaussa
    def create_gaussian_kernel(size, sigma):
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        sum_val = 0

        for i in range(size):
            for j in range(size):
                diff = (i - center) ** 2 + (j - center) ** 2
                kernel[i, j] = math.exp(-diff / (2 * sigma ** 2))
                sum_val += kernel[i, j]

        kernel /= sum_val
        return kernel

    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)

    padded_image = np.pad(np_image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    output_image = np.zeros_like(np_image)

    for k in range(output_image.shape[2]):
        for i in range(output_image.shape[0]):
            for j in range(output_image.shape[1]):
                region = padded_image[i:i + kernel_size, j:j + kernel_size, k]
                output_image[i, j, k] = np.sum(region * gaussian_kernel)

    if is_grayscale:
        output_image = output_image.squeeze()

    return output_image.astype(np.uint8)

def to_sharpening(np_image):
    try:
        sharpness_level = float(entry_var_sharp.get())
        if sharpness_level < 0 or sharpness_level > 100:
            print("Proszę podać wartość w zakresie 0-100.")
            return np_image
    except ValueError:
        print("Proszę podać poprawną liczbę dla ostrości.")
        return np_image

    s = sharpness_level / 50

    kernel = np.array([
        [0, -s, 0],
        [-s, (4 * s + 1), -s],
        [0, -s, 0]
    ])

    is_grayscale = len(np_image.shape) == 2
    if is_grayscale:
        np_image = np.expand_dims(np_image, axis=-1)

    pad = 1
    padded_image = np.pad(np_image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    output_image = np.zeros_like(np_image)

    for k in range(output_image.shape[2]):
        for i in range(output_image.shape[0]):
            for j in range(output_image.shape[1]):
                region = padded_image[i:i + 3, j:j + 3, k]
                output_image[i, j, k] = np.clip(np.sum(region * kernel), 0, 255)

    if is_grayscale:
        output_image = output_image.squeeze()

    return output_image.astype(np.uint8)


""" functions for apllaying filters """
def grayscale():
    if grayscale_var.get():
        img_filter_stack.append(to_grayscale)
    else:
        img_filter_stack.remove(to_grayscale)
    apply_filters()

def negative():
    if negative_var.get():
        img_filter_stack.append(to_negative)
    else:
        img_filter_stack.remove(to_negative)
    apply_filters()

def binary():
    if binary_var.get():
        img_filter_stack.append(to_binary)
    else:
        img_filter_stack.remove(to_binary)
    apply_filters()

def brightness():
    if adjust_brightness in img_filter_stack:
        img_filter_stack.remove(adjust_brightness)
    img_filter_stack.append(adjust_brightness)
    apply_filters()

def contrast():
    if adjust_contrast in img_filter_stack:
        img_filter_stack.remove(adjust_contrast)
    img_filter_stack.append(adjust_contrast)
    apply_filters()

def averaging():
    if to_averaging in img_filter_stack:
        img_filter_stack.remove(to_averaging)
    img_filter_stack.append(to_averaging)
    apply_filters()

def gaussian():
    if to_gaussian in img_filter_stack:
        img_filter_stack.remove(to_gaussian)
    img_filter_stack.append(to_gaussian)
    apply_filters()

def sharpening():
    if to_sharpening in img_filter_stack:
        img_filter_stack.remove(to_sharpening)
    img_filter_stack.append(to_sharpening)
    apply_filters()


""" Main """
root = tk.Tk()
root.title("BMP image editor")
root.geometry("1000x700")

# Menu aplikacji
menu = Menu(root)
root.config(menu=menu)

# Menu Plik
file_menu = Menu(menu, tearoff=0)
menu.add_cascade(label="Plik", menu=file_menu)
file_menu.add_command(label="Wczytaj obraz", command=load_image)
file_menu.add_separator()
file_menu.add_command(label="Zamknij", command=root.quit)

# Ramka do wyświetlania kanw obok siebie
frame = tk.Frame(root)
frame.pack(pady=20)

# Kanwy do wyświetlania obrazów
canvas1 = tk.Canvas(frame, width=280, height=360, bg="gray")
canvas1.grid(row=0, column=0, padx=10)

canvas2 = tk.Canvas(frame, width=280, height=360, bg="gray")
canvas2.grid(row=0, column=1, padx=10)

load_image()  ## DO USUNIECIA JKA BEDE POBIERAC ZDJECIA Z FILE PATH
# Ramka na checkboxy i inne filtry w dwóch kolumnach
filter_frame = tk.Frame(root)
filter_frame.pack(pady=10)

checkbox_frame = tk.Frame(filter_frame)
checkbox_frame.grid(row=0, column=0, padx=20, sticky="n")

adjustments_frame = tk.Frame(filter_frame)
adjustments_frame.grid(row=0, column=1, padx=20, sticky="n")

filters_frame = tk.Frame(filter_frame)
filters_frame.grid(row=0, column=2, padx=20, sticky="n")

img_filter_stack = []

""" Checkboxes """
# Zmienna do przechowywania informacji, czy checkbox jest zaznaczony
grayscale_var = tk.IntVar()
negative_var = tk.IntVar()
binary_var = tk.IntVar()

# Checkbox - konwersja do szarości, negatyw, binarizacja
tk.Checkbutton(checkbox_frame, text="Konwersja do odcieni szarości", variable=grayscale_var, command=grayscale).pack(anchor="w")
tk.Checkbutton(checkbox_frame, text="Negatyw", variable=negative_var, command=negative).pack(anchor="w")
tk.Checkbutton(checkbox_frame, text="Binaryzacja", variable=binary_var, command=binary).pack(anchor="w")

""" Korekty"""
tk.Label(adjustments_frame, text="Korekta jasności (-100,100) :").pack()
brightness_frame = tk.Frame(adjustments_frame)
brightness_frame.pack(pady=2)
entry_var_brightness = tk.StringVar()
entry_box_brightness = tk.Entry(brightness_frame, textvariable=entry_var_brightness, width=10)
entry_box_brightness.grid(row=0, column=0, padx=5)
tk.Button(brightness_frame, text="Zastosuj", command=brightness).grid(row=0, column=1)

tk.Label(adjustments_frame, text="Korekta kontrastu (-100,100) :").pack()
contrast_frame = tk.Frame(adjustments_frame)
contrast_frame.pack(pady=2)
entry_var_contrast = tk.StringVar()
entry_box_contrast = tk.Entry(contrast_frame, textvariable=entry_var_contrast, width=10)
entry_box_contrast.grid(row=0, column=0, padx=5)
tk.Button(contrast_frame, text="Zastosuj", command=contrast).grid(row=0, column=1)

""" Inne filtry """
# Filtr uśredniający
tk.Label(filters_frame, text="Filtr uśredniający (0-100):").pack()
avg_frame = tk.Frame(filters_frame)
avg_frame.pack(pady=2)
entry_var_avg = tk.StringVar()
entry_box_avg = tk.Entry(avg_frame, textvariable=entry_var_avg, width=10)
entry_box_avg.grid(row=0, column=0, padx=5)
tk.Button(avg_frame, text="Zastosuj", command=averaging).grid(row=0, column=1)

# Filtr Gaussa
tk.Label(filters_frame, text="Filtr Gaussa (0-100):").pack()
gauss_frame = tk.Frame(filters_frame)
gauss_frame.pack(pady=2)
entry_var_gauss = tk.StringVar()
entry_box_gauss = tk.Entry(gauss_frame, textvariable=entry_var_gauss, width=10)
entry_box_gauss.grid(row=0, column=0, padx=5)
tk.Button(gauss_frame, text="Zastosuj", command=gaussian).grid(row=0, column=1)

# Filtr wyostrzający
tk.Label(filters_frame, text="Filtr wyostrzający (0-100):").pack()
sharp_frame = tk.Frame(filters_frame)
sharp_frame.pack(pady=2)
entry_var_sharp = tk.StringVar()
entry_box_sharp = tk.Entry(sharp_frame, textvariable=entry_var_sharp, width=10)
entry_box_sharp.grid(row=0, column=0, padx=5)
tk.Button(sharp_frame, text="Zastosuj", command=sharpening).grid(row=0, column=1)

root.mainloop()