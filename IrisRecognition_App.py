import tkinter as tk
from tkinter import filedialog, Menu
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
from scipy.ndimage import binary_closing, binary_opening
from scipy.ndimage import generate_binary_structure, iterate_structure
from scipy.ndimage import label


class IrisRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Recognition App")
        self.root.geometry("1000x800")

        self.display_width = 300
        self.display_height = 200

        self.img = None
        self.img_original = None
        self.img_tk = None
        self.canvas_original = None

        # PUPIL
        self.pupil_img_tk = None
        self.canvas_pupil = None
        self.canvas_pupil_cleaned = None
        self.pupil_cleaned_img_tk = None
        self.canvas_pupil_largest_blob = None
        self.pupil_largest_blob_img_tk = None
        self.canvas_pupil_detected = None
        self.pupil_detected_img_tk = None

        self.xp_value_default = tk.DoubleVar(value=5.0)
        self.use_custom_xp = tk.BooleanVar(value=False)
        self.open_strength_pupil = tk.IntVar(value=1)
        self.close_strength_pupil = tk.IntVar(value=2)


        self.setup_ui()

    def setup_ui(self):
        menu = Menu(self.root)
        self.root.config(menu=menu)

        file_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        self.canvas_original = tk.Canvas(self.root, width=self.display_width, height=self.display_height, bg="gray")
        self.canvas_original.pack(pady=10)


        # PUPIL
        title_label = tk.Label(self.root, text="Wykrywanie źrenicy", font=("Arial", 14))
        title_label.pack(pady=5)

        checkbox_frame = tk.Frame(self.root)
        checkbox_frame.pack(pady=5)
        self.checkbox = tk.Checkbutton(checkbox_frame, text="Wybierz wartości paramtrów samodzielnie",
                                       variable=self.use_custom_xp, command=self.toggle_custom_option)
        self.checkbox.pack(padx=5)

        option_frame = tk.Frame(self.root)
        option_frame.pack(pady=5)
        self.slider_frame = tk.Frame(option_frame)
        tk.Label(self.slider_frame, text="Wartość do wyboru progu:").pack(side=tk.LEFT, padx=5)
        self.slider = tk.Scale(self.slider_frame, from_=1.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL,
                               variable=self.xp_value_default)
        self.slider.pack(side=tk.LEFT)


        tk.Label(self.slider_frame, text="Siła filtrów:").pack(side=tk.LEFT, padx=5)
        tk.Label(self.slider_frame, text="Otwarcie").pack(side=tk.LEFT, padx=5)
        self.open_entry = tk.Entry(self.slider_frame, width=3, textvariable=self.open_strength_pupil)
        self.open_entry.pack(side=tk.LEFT)

        tk.Label(self.slider_frame, text="Zamknięcie").pack(side=tk.LEFT, padx=5)
        self.close_entry = tk.Entry(self.slider_frame, width=3, textvariable=self.close_strength_pupil)
        self.close_entry.pack(side=tk.LEFT)

        tk.Label(self.slider_frame, text="Metoda wykrywania środka źrenicy:").pack(side=tk.LEFT, padx=5)
        self.center_method = tk.StringVar(value="projection")

        tk.Radiobutton(self.slider_frame, text="Centroid", variable=self.center_method, value="centroid").pack(side=tk.LEFT)
        tk.Radiobutton(self.slider_frame, text="Projekcja", variable=self.center_method, value="projection").pack(side=tk.LEFT)

        self.apply_button = tk.Button(self.slider_frame, text="Zastosuj", command=self.detect_pupil)
        self.apply_button.pack(side=tk.LEFT, padx=10)

        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=10)

        def add_labeled_canvas(parent, label_text):
            frame = tk.Frame(parent)
            label = tk.Label(frame, text=label_text, font=("Arial", 10, "bold"))
            label.pack()
            canvas = tk.Canvas(frame, width=self.display_width, height=self.display_height, bg="gray")
            canvas.pack()
            frame.pack(side=tk.LEFT, padx=5)
            return canvas

        self.canvas_pupil = add_labeled_canvas(image_frame, "Binarna źrenica")
        self.canvas_pupil_cleaned = add_labeled_canvas(image_frame, "Po czyszczeniu")
        self.canvas_pupil_largest_blob = add_labeled_canvas(image_frame, "Największy blob")
        self.canvas_pupil_detected = add_labeled_canvas(image_frame, "Wykryta źrenica")


    def toggle_custom_option(self):
        if self.use_custom_xp.get():
            self.slider_frame.pack(pady=5)
        else:
            self.slider_frame.pack_forget()
            self.xp_value_default.set(5.0)
            self.open_strength_pupil.set(1)
            self.close_strength_pupil.set(2)
            self.detect_pupil()

    def load_image(self):
        filepath = filedialog.askopenfilename(
            initialdir="data",
            title="Select Image",
            filetypes=[("JPG files", "*.jpg")]
        )
        if filepath:
            self.img_original = Image.open(filepath)
            self.img = self.img_original.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(self.img)

            self.canvas_original.delete("all")
            self.canvas_original.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
            self.canvas_original.image = self.img_tk

            self.detect_pupil()


    """ ŹRENICA """
    def detect_pupil(self):
        if self.img_original is None:
            return

        X_P = self.xp_value_default.get()
        img_resized = self.img_original.resize((self.display_width, self.display_height))

        # Step 1 – Binary pupil
        binary_pupil = self.pupil_detect_binary(img_resized, X_P)
        binary_img = Image.fromarray(binary_pupil)
        self.pupil_img_tk = ImageTk.PhotoImage(binary_img)
        self.canvas_pupil.delete("all")
        self.canvas_pupil.create_image(0, 0, anchor=tk.NW, image=self.pupil_img_tk)
        self.canvas_pupil.image = self.pupil_img_tk

        # Step 2 – Cleaned pupil image
        cleaned_pupil = self.pupil_morphologically_clean(binary_pupil)
        cleaned_img = Image.fromarray(cleaned_pupil)
        self.pupil_cleaned_img_tk = ImageTk.PhotoImage(cleaned_img)
        self.canvas_pupil_cleaned.delete("all")
        self.canvas_pupil_cleaned.create_image(0, 0, anchor=tk.NW, image=self.pupil_cleaned_img_tk)
        self.canvas_pupil_cleaned.image = self.pupil_cleaned_img_tk

        # Step 3 – Largest blob from cleaned image
        largest_pupil = self.pupil_get_largest_blob(cleaned_pupil)
        blob_img = Image.fromarray(largest_pupil)
        self.blob_img_tk = ImageTk.PhotoImage(blob_img)
        self.canvas_pupil_largest_blob.delete("all")
        self.canvas_pupil_largest_blob.create_image(0, 0, anchor=tk.NW, image=self.blob_img_tk)
        self.canvas_pupil_largest_blob.image = self.blob_img_tk

        # Step 4 – Circle on original
        self.pupil_draw_detected(img_resized, largest_pupil)

    def pupil_detect_binary(self, image, X_P):
        img_np = np.array(image)
        gray_np = np.dot(img_np[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        P = np.mean(gray_np)
        P_P = P / X_P
        binary_pupil = np.where(gray_np < P_P, 255, 0).astype(np.uint8)
        return binary_pupil

    def erosion(self, binary_img, structure):
        pad_h, pad_w = structure.shape[0] // 2, structure.shape[1] // 2
        padded_img = np.pad(binary_img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        result = np.zeros_like(binary_img)

        for i in range(binary_img.shape[0]):
            for j in range(binary_img.shape[1]):
                region = padded_img[i:i + structure.shape[0], j:j + structure.shape[1]]
                if np.array_equal(region[structure == 1], np.ones(np.sum(structure))):
                    result[i, j] = 1
        return result
    def dilation(self, binary_img, structure):
        pad_h, pad_w = structure.shape[0] // 2, structure.shape[1] // 2
        padded_img = np.pad(binary_img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        result = np.zeros_like(binary_img)

        for i in range(binary_img.shape[0]):
            for j in range(binary_img.shape[1]):
                region = padded_img[i:i + structure.shape[0], j:j + structure.shape[1]]
                if np.any(region[structure == 1]):
                    result[i, j] = 1
        return result

    def pupil_morphologically_clean(self, binary_pupil):
        binary_bool = (binary_pupil > 0).astype(int)
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=int)

        open_size = self.open_strength_pupil.get()
        close_size = self.close_strength_pupil.get()

        opened = binary_bool.copy()
        for _ in range(open_size):
            opened = self.erosion(opened, structure)
        for _ in range(open_size):
            opened = self.dilation(opened, structure)

        closed = opened.copy()
        for _ in range(close_size):
            closed = self.dilation(closed, structure)
        for _ in range(close_size):
            closed = self.erosion(closed, structure)

        return (closed * 255).astype(np.uint8)

    def pupil_morphologically_clean2(self, binary_pupil):
        binary_bool = binary_pupil.astype(bool)
        structure = generate_binary_structure(2, 1)
        open_size = self.open_strength_pupil.get()
        close_size = self.close_strength_pupil.get()

        structure_open = iterate_structure(structure, open_size)
        structure_close = iterate_structure(structure, close_size)

        cleaned = binary_opening(binary_bool, structure=structure_open)
        cleaned = binary_closing(cleaned, structure=structure_close)

        return (cleaned * 255).astype(np.uint8)

    def pupil_get_largest_blob(self, binary_img):
        labeled_array, num_features = label(binary_img)
        if num_features == 0:
            return binary_img

        sizes = [(labeled_array == i).sum() for i in range(1, num_features + 1)]
        max_label = np.argmax(sizes) + 1
        return (labeled_array == max_label).astype(np.uint8) * 255

    def get_horizontal_projection(self, binary):
        return np.sum(binary, axis=1)

    def get_vertical_projection(self, binary):
        return np.sum(binary, axis=0)

    def get_diagonal_projection_45(self, binary):
        return np.array([np.sum(np.diagonal(binary, offset=o)) for o in range(-binary.shape[0] + 1, binary.shape[1])])

    def get_diagonal_projection_135(self, binary):
        flipped = np.flipud(binary)
        return np.array(
            [np.sum(np.diagonal(flipped, offset=o)) for o in range(-flipped.shape[0] + 1, flipped.shape[1])])

    def get_pupil_center_by_projections(self, binary_pupil):
        vertical_proj = self.get_vertical_projection(binary_pupil)
        horizontal_proj = self.get_horizontal_projection(binary_pupil)
        diag45 = self.get_diagonal_projection_45(binary_pupil)
        diag135 = self.get_diagonal_projection_135(binary_pupil)

        projections = {
            "vertical": (np.max(vertical_proj), int(np.argmax(vertical_proj))),
            "horizontal": (np.max(horizontal_proj), int(np.argmax(horizontal_proj))),
            "diag45": (np.max(diag45), int(np.argmax(diag45))),
            "diag135": (np.max(diag135), int(np.argmax(diag135)))
        }
        sorted_projections = sorted(projections.items(), key=lambda x: x[1][0], reverse=True)

        (first_name, (first_value, first_idx)) = sorted_projections[0]
        (second_name, (second_value, second_idx)) = sorted_projections[1]

        def index_to_xy(proj_name, idx):
            if proj_name == "vertical":
                return idx, None  # X znany
            elif proj_name == "horizontal":
                return None, idx  # Y znany
            elif proj_name == "diag45":
                x = (binary_pupil.shape[1] + idx) // 2
                y = (binary_pupil.shape[0] - 1 - idx) // 2
                return x, y
            elif proj_name == "diag135":
                x = (binary_pupil.shape[1] + idx) // 2
                y = (binary_pupil.shape[0] + idx) // 2
                return x, y

        first_x, first_y = index_to_xy(first_name, first_idx)
        second_x, second_y = index_to_xy(second_name, second_idx)

        center_x = first_x if first_x is not None else second_x
        center_y = first_y if first_y is not None else second_y

        return int(center_x), int(center_y)

    def pupil_draw_detected(self, img_resized, binary_pupil):
        # vertical_proj = np.sum(binary_pupil, axis=0)
        # horizontal_proj = np.sum(binary_pupil, axis=1)

        if self.center_method.get() == "projection":
            # center_x = int(np.argmax(vertical_proj))
            # center_y = int(np.argmax(horizontal_proj))
            center_x, center_y = self.get_pupil_center_by_projections(binary_pupil)
        else:
            coords = np.column_stack(np.where(binary_pupil == 255))
            if coords.size > 0:
                center_y, center_x = coords.mean(axis=0)
            else:
                center_x = center_y = 0

        coords = np.column_stack(np.where(binary_pupil == 255))
        if coords.size > 0:
            radius = np.max(np.linalg.norm(coords - [center_y, center_x], axis=1))
        else:
            radius = 0

        marked_img = img_resized.copy()
        draw = ImageDraw.Draw(marked_img)
        draw.ellipse(
            [(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)],
            outline="red", width=2
        )

        self.pupil_detected_img_tk = ImageTk.PhotoImage(marked_img)
        self.canvas_pupil_detected.delete("all")
        self.canvas_pupil_detected.create_image(0, 0, anchor=tk.NW, image=self.pupil_detected_img_tk)
        self.canvas_pupil_detected.image = self.pupil_detected_img_tk


if __name__ == '__main__':
    root = tk.Tk()
    app = IrisRecognitionApp(root)
    root.mainloop()


"""
Przypadki gdzie wykrywanie źrenicy napewno działa (przy opcji "centroid" bo te projekcje mi beznadziejnie działają):
- 76, 1, X_p=3.6, 1,2
- 41, 2, X_p=3.4, 3,3
- 34, 1, X_p=3.9

"""