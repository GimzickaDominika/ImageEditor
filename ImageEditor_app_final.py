import math
import tkinter as tk
from tkinter import filedialog, Menu
from PIL import Image, ImageTk
import numpy as np
from skimage.util import view_as_windows
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


""" Load image function """
def load_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.bmp;*.jpg"), ("BMP Files", "*.bmp"), ("JPG Files", "*.jpg")])
    if filepath:
        global img, img_original, img_tk, img_tk_filter, img_np, img_filter, img_filter_np
        img = Image.open(filepath)
        img_original = img.copy()  # Copy of the original image for backup
        img_filter = img.copy()  # Copy of the image for applying filters

        img_np = np.array(img)
        img_filter_np = np.array(img_filter)

        # Create ImageTk object for both fields
        img_tk = ImageTk.PhotoImage(img)
        img_tk_filter = ImageTk.PhotoImage(img_filter)

        # Display images on both canvases
        canvas1.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas2.create_image(0, 0, anchor=tk.NW, image=img_tk_filter)

        # Draw histograms for both images
        draw_rgb_histogram(hist_frame1, img_np)
        draw_rgb_histogram(hist_frame2, img_filter_np)

    else:
        print("No file chosen.")


""" Save image function"""
def save_image():
    filepath = filedialog.asksaveasfilename(defaultextension=".bmp",
                                            filetypes=[("BMP Files", "*.bmp"), ("JPG Files", "*.jpg")])
    if filepath:
        try:
            img_to_save = Image.fromarray(img_filter_np)
            if img_filter_np.ndim == 3 and img_filter_np.shape[2] == 3:
                img_to_save = img_to_save.convert("RGB")
            img_to_save.save(filepath)
            print(f"Image saved to {filepath}")
        except Exception as e:
            print(f"Error saving image: {e}")


""" Function to apply filters """
def apply_filters():
    clear_error()
    global img_filter_np, img_filter

    # Start from an original image
    img_filter = img_original.copy()
    img_filter_np = np.array(img_filter)

    # Apply each filter from the stack
    for filter_func in img_filter_stack:
        img_filter_np = filter_func(img_filter_np)

    # Update canvas
    img_filter = Image.fromarray(img_filter_np)
    img_tk_filter = ImageTk.PhotoImage(img_filter)
    canvas2.create_image(0, 0, anchor=tk.NW, image=img_tk_filter)
    canvas2.image = img_tk_filter

    draw_rgb_histogram(hist_frame2, img_filter_np)


"""" Function for drawing histogram"""
def draw_rgb_histogram(canvas_container, image_array):
    for widget in canvas_container.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(6.1, 3.5), dpi=100)
    ax = fig.add_subplot(111)

    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)

    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        channel = image_array[:, :, i].flatten()
        ax.hist(channel, bins=128, range=(0, 255), color=color, alpha=0.4, label=f'{color.capitalize()}')

    ax.set_xlim(0, 255)
    ax.set_ylabel("Number of pixels", fontsize=6)
    ax.set_xlabel("Brightness", fontsize=6)
    ax.tick_params(axis='both', labelsize=6)
    ax.legend(fontsize=6, loc='upper right')
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=canvas_container)
    canvas.draw()
    canvas.get_tk_widget().pack()


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
        brightness_val = int(entry_var_brightness.get())
        if brightness_val < -100 or brightness_val > 100:
            show_error("Please enter a brightness level in the range [-100, 100].")
            return np_image
    except ValueError:
        show_error("Please enter a valid brightness level.")
        return np_image

    np_image = np_image.astype(np.float32)

    adjusted_image = np.clip(np_image + brightness_val, 0, 255)

    return adjusted_image.astype(np.uint8)


def adjust_contrast(np_image):
    try:
        contrast_val = int(entry_var_contrast.get())
        if contrast_val < -100 or contrast_val > 100:
            show_error("Please enter a contrast level in the range [-100, 100].")
            return np_image
    except ValueError:
        show_error("Please enter a valid contrast level.")
        return np_image

    np_image = np_image.astype(np.float32)

    factor = (259 * (contrast_val + 255)) / (255 * (259 - contrast_val))
    adjusted_image = factor * (np_image - 128) + 128

    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    return adjusted_image


def to_averaging(np_image):
    try:
        value = float(entry_var_avg.get())
        if value < 0 or value > 100:
            show_error("Please enter an averaging level in the range [0, 100].")
            return np_image
    except ValueError:
        show_error("Please enter a valid averaging level.")
        return np_image

    min_size = min(np_image.shape[0], np_image.shape[1])

    kernel_size = max(3, min(int(value) // 10 * 2 + 1, min_size - 1))
    if kernel_size % 2 == 0:
        kernel_size -= 1

    pad = kernel_size // 2

    is_grayscale = len(np_image.shape) == 2
    if is_grayscale:
        np_image = np.expand_dims(np_image, axis=-1)

    padded_image = np.pad(np_image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    if kernel_size > min_size - 1:
        show_error(f"Kernel size too large: {kernel_size}. Adjusted to {min_size - 1}.")
        kernel_size = min_size - 1
        if kernel_size % 2 == 0:
            kernel_size -= 1

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
            show_error("Please enter a sigma value in the range (0, 100].")
            return np_image
    except ValueError:
        show_error("Please enter a valid sigma value.")
        return np_image

    kernel_size = max(3, int(sigma // 10) * 2 + 1)
    pad = kernel_size // 2

    is_grayscale = len(np_image.shape) == 2
    if is_grayscale:
        np_image = np.expand_dims(np_image, axis=-1)

    # Create Gaussian kernel
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
            show_error("Please enter a sharpness level in the range [0, 100].")
            return np_image
    except ValueError:
        show_error("Please enter a valid sharpness level.")
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


""" Functions for applying filters """
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


""" Function for reseting all filters """
def reset_filters():
    img_filter_stack.clear()

    grayscale_var.set(0)
    negative_var.set(0)
    binary_var.set(0)

    entry_var_brightness.set("")
    entry_var_contrast.set("")
    entry_var_avg.set("")
    entry_var_gauss.set("")
    entry_var_sharp.set("")

    apply_filters()


""" Error messages handling """
def show_error(msg):
    error_label.config(text=msg)


def clear_error():
    error_label.config(text="")


""" Main """
img_filter_stack = []

root = tk.Tk()
# Scrollable main area
main_canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
scrollable_frame = tk.Frame(main_canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: main_canvas.configure(
        scrollregion=main_canvas.bbox("all")
    )
)

main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
main_canvas.configure(yscrollcommand=scrollbar.set)

main_canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

root.title("BMP image editor")
root.geometry("1270x600+0+20")

# Application menu
menu = Menu(root)
root.config(menu=menu)

# Top button frame
top_button_frame = tk.Frame(scrollable_frame)
top_button_frame.pack(anchor="nw", padx=15, pady=10)

load_button = tk.Button(top_button_frame, text="Load image", command=load_image, width=12)
load_button.pack(side="left", padx=(0, 5))

save_button = tk.Button(top_button_frame, text="Save image", command=save_image, width=12)
save_button.pack(side="left")

reset_button = tk.Button(top_button_frame, text="Reset filters", command=reset_filters, width=12)
reset_button.pack(side="left", padx=(5, 0))

# Variable for storing whether the checkbox is ticked
grayscale_var = tk.IntVar()
negative_var = tk.IntVar()
binary_var = tk.IntVar()

filter_frame = tk.Frame(scrollable_frame)
filter_frame.pack(pady=(10, 5), anchor="nw")

# === Column 1: Checkboxes ===
checkbox_column = tk.Frame(filter_frame)
checkbox_column.grid(row=0, column=0, padx=(10, 30), sticky="n")

tk.Checkbutton(checkbox_column, text="Grayscale conversion", variable=grayscale_var, command=grayscale).pack(anchor="w", pady=2)
tk.Checkbutton(checkbox_column, text="Negative", variable=negative_var, command=negative).pack(anchor="w", pady=2)
tk.Checkbutton(checkbox_column, text="Binarization", variable=binary_var, command=binary).pack(anchor="w", pady=2)

# === Column 2: Correction labels ===
corrections_labels = tk.Frame(filter_frame)
corrections_labels.grid(row=0, column=1, padx=(10, 5), sticky="n")

tk.Label(corrections_labels, text="Brightness correction [-100, 100]:").pack(anchor="w", pady=4)
tk.Label(corrections_labels, text="Contrast correction [-100, 100]:").pack(anchor="w", pady=4)

# === Column 3: Entry + Apply for correction ===
corrections_inputs = tk.Frame(filter_frame)
corrections_inputs.grid(row=0, column=2, padx=(0, 30), sticky="n")

# Brightness
entry_var_brightness = tk.StringVar()
brightness_row = tk.Frame(corrections_inputs)
brightness_row.pack(anchor="w", pady=2)
tk.Entry(brightness_row, textvariable=entry_var_brightness, width=5).pack(side="left", padx=(0, 5))
tk.Button(brightness_row, text="Apply", command=brightness).pack(side="left")

# Contrast
entry_var_contrast = tk.StringVar()
contrast_row = tk.Frame(corrections_inputs)
contrast_row.pack(anchor="w", pady=2)
tk.Entry(contrast_row, textvariable=entry_var_contrast, width=5).pack(side="left", padx=(0, 5))
tk.Button(contrast_row, text="Apply", command=contrast).pack(side="left")

# === Column 4: Filters labels ===
filters_labels = tk.Frame(filter_frame)
filters_labels.grid(row=0, column=3, padx=(10, 5), sticky="n")

tk.Label(filters_labels, text="Averaging filter [0, 100]:").pack(anchor="w", pady=4)
tk.Label(filters_labels, text="Gaussian filter (0, 100]:").pack(anchor="w", pady=4)
tk.Label(filters_labels, text="Sharpening filter [0, 100]:").pack(anchor="w", pady=4)

# === Column 5: Entry + Apply for filters ===
filters_inputs = tk.Frame(filter_frame)
filters_inputs.grid(row=0, column=4, padx=(0, 10), sticky="n")

# Averaging
entry_var_avg = tk.StringVar()
avg_row = tk.Frame(filters_inputs)
avg_row.pack(anchor="w", pady=2)
tk.Entry(avg_row, textvariable=entry_var_avg, width=5).pack(side="left", padx=(0, 5))
tk.Button(avg_row, text="Apply", command=averaging).pack(side="left")

# Gaussian
entry_var_gauss = tk.StringVar()
gauss_row = tk.Frame(filters_inputs)
gauss_row.pack(anchor="w", pady=2)
tk.Entry(gauss_row, textvariable=entry_var_gauss, width=5).pack(side="left", padx=(0, 5))
tk.Button(gauss_row, text="Apply", command=gaussian).pack(side="left")

# Sharpening
entry_var_sharp = tk.StringVar()
sharp_row = tk.Frame(filters_inputs)
sharp_row.pack(anchor="w", pady=2)
tk.Entry(sharp_row, textvariable=entry_var_sharp, width=5).pack(side="left", padx=(0, 5))
tk.Button(sharp_row, text="Apply", command=sharpening).pack(side="left")

# Error labels
error_label = tk.Label(scrollable_frame, text="", fg="red", font=("Arial", 10))
error_label.pack(pady=(5, 0))

# Frame for displaying canvases side by side
frame = tk.Frame(scrollable_frame)
frame.pack(pady=(0, 10), anchor="center")

# Canvases for displaying images
canvas1 = tk.Canvas(frame, width=610, height=352, bg="gray")
canvas1.grid(row=0, column=0, padx=10, pady=(5, 8))

canvas2 = tk.Canvas(frame, width=610, height=352, bg="gray")
canvas2.grid(row=0, column=1, padx=10, pady=(5, 8))

# Canvases labels
label1 = tk.Label(frame, text="Original image")
label1.grid(row=1, column=0)

label2 = tk.Label(frame, text="Processed image")
label2.grid(row=1, column=1)

hist_frame1 = tk.Frame(frame, width=610, height=352, bg="gray")
hist_frame1.grid(row=2, column=0, padx=10, pady=(25, 10))

hist_frame2 = tk.Frame(frame, width=610, height=352, bg="gray")
hist_frame2.grid(row=2, column=1, padx=10, pady=(25, 10))

label_hist1 = tk.Label(frame, text="RGB histogram - original image")
label_hist1.grid(row=3, column=0)

label_hist2 = tk.Label(frame, text="RGB histogram - processed image")
label_hist2.grid(row=3, column=1)

def _on_mousewheel(event):
    main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

main_canvas.bind_all("<MouseWheel>", _on_mousewheel)


# --- Frame for projections ---
proj_section = tk.Frame(scrollable_frame)
proj_section.pack(pady=8, padx=8)


def load_binary_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.bmp;*.jpg")])
    if not file_path:
        return

    img_bin = Image.open(file_path).convert("L")
    img_bin = img_bin.point(lambda x: 0 if x < 128 else 255, '1')
    binary_np = np.array(img_bin).astype(np.uint8) * 255

    plot_projections(binary_np, proj_display)


# Drawing horizontal and vertical projections + image in the center
def plot_projections(binary_img, canvas_container):
    for widget in canvas_container.winfo_children():
        widget.destroy()

    rows = np.any(binary_img == 0, axis=1)
    cols = np.any(binary_img == 0, axis=0)

    if not rows.any() or not cols.any():
        print("No black pixels in the image.")
        return

    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    cropped_img = binary_img[top:bottom + 1, left:right + 1]

    proj_vert = np.sum(cropped_img == 0, axis=0)
    proj_horiz = np.sum(cropped_img == 0, axis=1)

    fig = Figure(figsize=(6, 2.5), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 2), height_ratios=(2, 4),
                          wspace=0.3, hspace=0.3)

    height = cropped_img.shape[0]

    # Horizontal projection (top)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.bar(np.arange(len(proj_vert)), proj_vert, color='black', width=1.0)
    ax_top.set_xlim([0, len(proj_vert)])
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.tick_params(axis='both', labelsize=8)

    # Image display
    ax_img = fig.add_subplot(gs[1, 0])
    ax_img.imshow(cropped_img, cmap='gray', vmin=0, vmax=255, aspect='auto')
    ax_img.set_xlim([0, cropped_img.shape[1]])
    ax_img.set_ylim([cropped_img.shape[0], 0])
    ax_img.axis('off')

    # Vertical projection (right)
    ax_side = fig.add_subplot(gs[1, 1], sharey=ax_img)
    ax_side.barh(np.arange(height), proj_horiz, color='black', height=1.0)
    ax_side.set_xlim([0, np.max(proj_horiz)])
    ax_side.set_ylim([height, 0])
    ax_side.spines['bottom'].set_visible(False)
    ax_side.spines['right'].set_visible(False)
    ax_side.xaxis.tick_top()
    ax_side.tick_params(axis='both', labelsize=8)

    canvas = FigureCanvasTkAgg(fig, master=canvas_container)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)


# Load binary image button
btn_load_binary = tk.Button(proj_section, text="Load binary image", command=load_binary_image)
btn_load_binary.pack(pady=10)

# Frame for projection plot
proj_display = tk.Frame(proj_section, width=650, height=250, bg="gray")
proj_display.pack_propagate(False)
proj_display.pack()

root.mainloop()