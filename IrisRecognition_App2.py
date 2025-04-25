"""
Aplikacja służąca do wykrywania tęczówki i źrenicy na podstawie obrazu ludzkiego oka
oraz do wyodrębniania tęczówki z obrazu i rozwijania jej w prostokąt
"""


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from scipy.ndimage import generate_binary_structure, iterate_structure
from scipy.ndimage import label
from scipy.ndimage import map_coordinates


# Cała aplikacja została zaimplementowana w jednej klasie
class IrisRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Recognition App")

        # Główne elementy GUI
        self.main_canvas = tk.Canvas(self.root)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = tk.Frame(self.main_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(
                scrollregion=self.main_canvas.bbox("all")
            )
        )

        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.root.geometry("1270x600+0+20")

        # Jednakowe wymiary wyświetlanych obrazów w oknie aplikacji
        self.display_width = 300
        self.display_height = 200

        self.img = None
        self.img_original = None
        self.img_tk = None
        self.canvas_original = None

        # === Źrenica ===
        # Zmienne dla wyświetlanych obrazów i kanw
        self.pupil_img_tk = None
        self.canvas_pupil = None
        self.canvas_pupil_cleaned = None
        self.pupil_cleaned_img_tk = None
        self.canvas_pupil_largest_blob = None
        self.pupil_largest_blob_img_tk = None
        self.canvas_pupil_detected = None
        self.pupil_detected_img_tk = None

        # Parametry metody wykrywania źrenicy (wartości domyślne)
        self.xp_value_default = tk.DoubleVar(value=5.0)  # współczynnik czułości binaryzacji obrazu dla źrenicy
        self.open_strength_pupil = tk.IntVar(value=1)  # siła (liczba iteracji) operacji morfologicznej otwarcia
        self.close_strength_pupil = tk.IntVar(value=2)  # siła (liczba iteracji) operacji morfologicznej zamknięcia

        # Zmienne dla współrzędnych wykrytego środka źrenicy i długości jej promienia
        self.pupil_center = None
        self.pupil_radius = None

        # === Tęczówka ===
        # Zmienne dla wyświetlanych obrazów i kanw
        self.iris_img_tk = None
        self.canvas_iris = None
        self.canvas_iris_cleaned = None
        self.iris_cleaned_img_tk = None
        self.canvas_iris_largest_blob = None
        self.iris_largest_blob_img_tk = None
        self.canvas_iris_detected = None
        self.iris_detected_img_tk = None

        # Parametry metody wykrywania tęczówki (wartości domyślne)
        self.xi_value_default = tk.DoubleVar(value=1.0)  # współczynnik czułości binaryzacji obrazu dla tęczówki
        self.open_strength_iris = tk.IntVar(value=2)   # siła (liczba iteracji) operacji morfologicznej otwarcia
        self.close_strength_iris = tk.IntVar(value=3)  # siła (liczba iteracji) operacji morfologicznej zamknięcia

        # Zmienna dla wyznaczonej długości promienia tęczówki
        self.iris_radius = None

        # Zmienne dla obrazów i kanw w sekcji rozwijania tęczówki do prostokąta
        self.canvas_iris_drawn = None
        self.canvas_unwrapped = None
        self.unwrapped_img_tk = None
        self.iris_drawn_tk = None

        # Zmienna określająca tryb wyboru parametrów dla detekcji źrenicy i tęczówki -
        # - ręczny ("custom") lub automatyczny ("default")
        self.param_mode = tk.StringVar(value="none")

        # Zmienna określająca metodę wyznaczania środka źrenicy - projekcja lub centroid
        self.center_method = tk.StringVar(value="centroid")

        self.setup_ui()

        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def setup_ui(self):
        """
        Tworzy i konfiguruje interfejs użytkownika aplikacji:
        - dodaje przycisk do wczytywania obrazu,
        - konfiguruje sekcje i elementy sterujące dla detekcji źrenicy i tęczówki (suwaki, przyciski, pola tekstowe),
        - tworzy kanwy do wyświetlania wyników przetwarzania: obrazy binarne, po czyszczeniu, największy blob,
          wykryte kontury,
        - dodaje sekcję do rozwijania tęczówki w prostokąt i kanwy do prezentacji wyników.
        """
        top_button_frame = tk.Frame(self.scrollable_frame)
        top_button_frame.pack(anchor="nw", pady=10, padx=10)

        load_button = tk.Button(top_button_frame, text="Load image", command=self.load_image)
        load_button.pack(side="left")

        self.canvas_original = tk.Canvas(self.scrollable_frame, width=self.display_width, height=self.display_height,
                                         bg="gray")
        self.canvas_original.pack(pady=10)

        # === Sekcja detekcji źrenicy ===
        title_label = tk.Label(self.scrollable_frame, text="Pupil detection", font=("Arial", 14))
        title_label.pack(pady=5)

        checkbox_frame = tk.Frame(self.scrollable_frame)
        checkbox_frame.pack(pady=5)

        tk.Radiobutton(checkbox_frame, text="Select parameters manually",
                       variable=self.param_mode, value="custom",
                       command=self.toggle_custom_option).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(checkbox_frame, text="Default parameters",
                       variable=self.param_mode, value="default",
                       command=self.toggle_custom_option).pack(side=tk.LEFT, padx=5)

        option_frame = tk.Frame(self.scrollable_frame)
        option_frame.pack(pady=5)

        self.slider_frame = tk.Frame(option_frame)
        tk.Label(self.slider_frame, text="Binarization sensitivity factor:").pack(side=tk.LEFT, padx=5)
        self.slider = tk.Scale(self.slider_frame, from_=1.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL,
                               variable=self.xp_value_default)
        self.slider.pack(side=tk.LEFT, padx=(5, 25))

        tk.Label(self.slider_frame, text="Filter strength:").pack(side=tk.LEFT, padx=5)
        tk.Label(self.slider_frame, text="Opening").pack(side=tk.LEFT, padx=5)
        self.open_entry = tk.Entry(self.slider_frame, width=3, textvariable=self.open_strength_pupil)
        self.open_entry.pack(side=tk.LEFT)

        tk.Label(self.slider_frame, text="Closing").pack(side=tk.LEFT, padx=5)
        self.close_entry = tk.Entry(self.slider_frame, width=3, textvariable=self.close_strength_pupil)
        self.close_entry.pack(side=tk.LEFT, padx=(5, 25))

        tk.Label(self.slider_frame, text="Pupil center detection method:").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(self.slider_frame, text="Centroid", variable=self.center_method,
                       value="centroid").pack(side=tk.LEFT)
        tk.Radiobutton(self.slider_frame, text="Projection", variable=self.center_method,
                       value="projection").pack(side=tk.LEFT, padx=(5, 25))

        self.apply_button = tk.Button(self.slider_frame, text="Apply", command=self.detect_pupil)
        self.apply_button.pack(side=tk.LEFT, padx=10)

        image_frame = tk.Frame(self.scrollable_frame)
        image_frame.pack(pady=(10, 40))

        def add_labeled_canvas(parent, label_text):
            frame = tk.Frame(parent)
            label = tk.Label(frame, text=label_text, font=("Arial", 10, "bold"))
            label.pack()
            canvas = tk.Canvas(frame, width=self.display_width, height=self.display_height, bg="gray")
            canvas.pack()
            frame.pack(side=tk.LEFT, padx=5)
            return canvas

        self.canvas_pupil = add_labeled_canvas(image_frame, "Binary pupil")
        self.canvas_pupil_cleaned = add_labeled_canvas(image_frame, "After cleaning")
        self.canvas_pupil_largest_blob = add_labeled_canvas(image_frame, "Largest blob")
        self.canvas_pupil_detected = add_labeled_canvas(image_frame, "Detected pupil")

        # === Sekcja detekcji tęczówki ===
        title_label_iris = tk.Label(self.scrollable_frame, text="Iris detection", font=("Arial", 14))
        title_label_iris.pack(pady=5)

        option_frame_iris = tk.Frame(self.scrollable_frame)
        option_frame_iris.pack(pady=5)

        self.slider_frame_iris = tk.Frame(option_frame_iris)
        tk.Label(self.slider_frame_iris, text="Binarization sensitivity factor:").pack(side=tk.LEFT, padx=5)
        self.slider_iris = tk.Scale(self.slider_frame_iris, from_=0.1, to=5.0, resolution=0.05, orient=tk.HORIZONTAL,
                               variable=self.xi_value_default)
        self.slider_iris.pack(side=tk.LEFT, padx=(5, 25))

        tk.Label(self.slider_frame_iris, text="Filter strength:").pack(side=tk.LEFT, padx=5)
        tk.Label(self.slider_frame_iris, text="Opening").pack(side=tk.LEFT, padx=5)
        self.open_entry_iris = tk.Entry(self.slider_frame_iris, width=3, textvariable=self.open_strength_iris)
        self.open_entry_iris.pack(side=tk.LEFT)

        tk.Label(self.slider_frame_iris, text="Closing").pack(side=tk.LEFT, padx=5)
        self.close_entry_iris = tk.Entry(self.slider_frame_iris, width=3, textvariable=self.close_strength_iris)
        self.close_entry_iris.pack(side=tk.LEFT, padx=(5, 25))

        self.apply_button_iris = tk.Button(self.slider_frame_iris, text="Apply", command=self.detect_iris)
        self.apply_button_iris.pack(side=tk.LEFT, padx=10)

        image_frame_iris = tk.Frame(self.scrollable_frame)
        image_frame_iris.pack(pady=(10, 40))

        self.canvas_iris = add_labeled_canvas(image_frame_iris, "Binary iris")
        self.canvas_iris_cleaned = add_labeled_canvas(image_frame_iris, "After cleaning")
        self.canvas_iris_largest_blob = add_labeled_canvas(image_frame_iris, "Largest blob")
        self.canvas_iris_detected = add_labeled_canvas(image_frame_iris, "Detected iris")

        # === Sekcja rozwijania tęczówki ===
        title_label_unwrap = tk.Label(self.scrollable_frame, text="Iris unwrapping", font=("Arial", 14))
        title_label_unwrap.pack(pady=5)

        unwrap_frame = tk.Frame(self.scrollable_frame)
        unwrap_frame.pack(pady=5)

        self.unwrap_button = tk.Button(unwrap_frame, text="Unwrap iris", command=self.unwrap_iris)
        self.unwrap_button.pack(side=tk.LEFT, padx=10)

        image_frame_unwrap = tk.Frame(self.scrollable_frame)
        image_frame_unwrap.pack(pady=(10, 40))

        def add_unwrap_canvas(parent, width, height):
            frame = tk.Frame(parent)
            canvas = tk.Canvas(frame, width=width, height=height, bg="gray")
            canvas.pack()
            frame.pack(side=tk.LEFT, padx=5)
            return canvas

        unwrapped_width = (self.display_width * 3) // 2
        unwrapped_height = self.display_height

        self.canvas_iris_drawn = add_unwrap_canvas(image_frame_unwrap, self.display_width, self.display_height)
        self.canvas_unwrapped = add_unwrap_canvas(image_frame_unwrap, unwrapped_width, unwrapped_height)

    def toggle_custom_option(self):
        """
        Przełącza tryb ustawień parametrów detekcji źrenicy i tęczówki:
        - Dla trybu „custom” pokazuje suwaki i pola do ręcznej regulacji parametrów.
        - Dla trybu „default” ukrywa suwaki, resetuje parametry do wartości domyślnych
          i automatycznie uruchamia detekcję źrenicy i tęczówki.
        """
        mode = self.param_mode.get()
        if mode == "custom":
            self.slider_frame.pack(pady=5)
            self.slider_frame_iris.pack(pady=5)
        elif mode == "default":
            self.slider_frame.pack_forget()
            self.slider_frame_iris.pack_forget()
            self.xp_value_default.set(5.0)
            self.xi_value_default.set(1.0)
            self.open_strength_pupil.set(1)
            self.close_strength_pupil.set(2)
            self.open_strength_pupil.set(2)
            self.close_strength_pupil.set(3)
            self.center_method.set("centroid")
            self.detect_pupil()
            self.detect_iris()

    def load_image(self):
        """
        Wczytuje nowy obraz z pliku (otwiera okno dialogowe do wyboru pliku JPG) i resetuje stan aplikacji -
        czyści kanwy w interfejsie aplikacji i usuwa poprzednie utworzone obrazy, resetuje wartości parametrów
        detekcji źrenicy i tęczówki.
        """
        filepath = filedialog.askopenfilename(
            initialdir="data",
            title="Select Image",
            filetypes=[("JPG files", "*.jpg")]
        )
        if filepath:
            canvases = [
                self.canvas_pupil, self.canvas_pupil_cleaned,
                self.canvas_pupil_largest_blob, self.canvas_pupil_detected,
                self.canvas_iris, self.canvas_iris_cleaned,
                self.canvas_iris_largest_blob, self.canvas_iris_detected,
                self.canvas_iris_drawn, self.canvas_unwrapped
            ]
            for canvas in canvases:
                if canvas:
                    canvas.delete("all")

            self.pupil_img_tk = None
            self.pupil_cleaned_img_tk = None
            self.pupil_largest_blob_img_tk = None
            self.pupil_detected_img_tk = None

            self.iris_img_tk = None
            self.iris_cleaned_img_tk = None
            self.iris_largest_blob_img_tk = None
            self.iris_detected_img_tk = None

            self.unwrapped_img_tk = None
            self.iris_drawn_tk = None

            self.pupil_center = None
            self.pupil_radius = None
            self.iris_radius = None

            self.slider_frame.pack_forget()
            self.slider_frame_iris.pack_forget()
            self.xp_value_default.set(5.0)
            self.xi_value_default.set(1.0)
            self.open_strength_pupil.set(1)
            self.close_strength_pupil.set(2)
            self.open_strength_iris.set(2)
            self.close_strength_iris.set(3)
            self.param_mode.set("none")
            self.center_method.set("centroid")

            self.img_original = Image.open(filepath)
            self.img = self.img_original.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(self.img)

            self.canvas_original.delete("all")
            self.canvas_original.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
            self.canvas_original.image = self.img_tk

    def _on_mousewheel(self, event):
        """
        Przewijanie zawartości okna aplikacji za pomocą kółka myszy.
        """
        self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


    # ------------------------------- ŹRENICA ---------------------------------
    def detect_pupil(self):
        """
        Wykonuje pełną sekwencję detekcji źrenicy przy użyciu funkcji pomocniczych.
        Wyniki każdego kroku wyświetlane są na odpowiednich kanwach interfejsu graficznego.
        """
        if self.img_original is None:
            return

        X_P = self.xp_value_default.get()
        img_resized = self.img_original.resize((self.display_width, self.display_height))

        # Krok 1:
        # Wygenerowanie binarnego obrazu źrenicy na podstawie oryginalnego obrazu oka
        # za pomocą funkcji 'pupil_detect_binary'.
        binary_pupil = self.pupil_detect_binary(img_resized, X_P)
        binary_img = Image.fromarray(binary_pupil)
        self.pupil_img_tk = ImageTk.PhotoImage(binary_img)
        self.canvas_pupil.delete("all")
        self.canvas_pupil.create_image(0, 0, anchor=tk.NW, image=self.pupil_img_tk)
        self.canvas_pupil.image = self.pupil_img_tk

        # Krok 2:
        # Czyszczenie binarnego obrazu źrenicy za pomocą operacji morfologicznych,
        # przy użyciu funkcji 'pupil_morphologically_clean'.
        cleaned_pupil = self.pupil_morphologically_clean(binary_pupil)
        cleaned_img = Image.fromarray(cleaned_pupil)
        self.pupil_cleaned_img_tk = ImageTk.PhotoImage(cleaned_img)
        self.canvas_pupil_cleaned.delete("all")
        self.canvas_pupil_cleaned.create_image(0, 0, anchor=tk.NW, image=self.pupil_cleaned_img_tk)
        self.canvas_pupil_cleaned.image = self.pupil_cleaned_img_tk

        # Krok 3:
        # Wybór największego obszaru (blob) jako kandydata na źrenicę,
        # przy użyciu funkcji 'pupil_get_largest_blob'.
        largest_pupil = self.pupil_get_largest_blob(cleaned_pupil)
        blob_img = Image.fromarray(largest_pupil)
        self.blob_img_tk = ImageTk.PhotoImage(blob_img)
        self.canvas_pupil_largest_blob.delete("all")
        self.canvas_pupil_largest_blob.create_image(0, 0, anchor=tk.NW, image=self.blob_img_tk)
        self.canvas_pupil_largest_blob.image = self.blob_img_tk

        # Krok 4:
        # Narysowanie konturu wykrytego obszaru żrenicy na oryginalnym obrazie oka
        # przy użyciu funkcji 'pupil_draw_detected'.
        self.pupil_draw_detected(img_resized, largest_pupil)

    def pupil_detect_binary(self, image, X_P):
        """
        Generuje binarny obraz źrenicy na podstawie podanego jako argument obrazu oka
        i współczynnika czułości binaryzacji (białe piksele - obszar potencjalnej źrenicy).
        Zwraca binarny obraz źrenicy w postaci tablicy NumPy (wartości 0 i 255).
        """
        # Konwersja obrazu RGB na skalę szarości
        img_np = np.array(image)
        gray_np = np.dot(img_np[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        # Obliczenie progu binaryzacji na podstawie średniej jasności obrazu i parametru X_P (współczynnika czułości)
        P = np.mean(gray_np)
        P_P = P / X_P

        # Piksele ciemniejsze od progu uznawane są za fragmenty źrenicy i kolorowane na biało
        binary_pupil = np.where(gray_np < P_P, 255, 0).astype(np.uint8)

        return binary_pupil

    def erosion(self, binary_img, structure):
        """
        Wykonuje operację erozji na podanym obrazie binarnym przy użyciu zadanego elementu strukturalnego.
        Zwraca obraz po erozji w postaci tablicy NumPy (wartości 0 i 1).
        """
        # Obliczenie wielkości obramowania na podstawie wymiarów elementu strukturalnego
        pad_h, pad_w = structure.shape[0] // 2, structure.shape[1] // 2

        # Dodanie obramowania do obrazu (wypełnienie zerami)
        padded_img = np.pad(binary_img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

        # Przygotowanie pustej macierzy na wynik erozji
        result = np.zeros_like(binary_img)

        # Dla każdego piksela obrazu sprawdzana jest lokalna okolica (region).
        # Jeśli w regionie pod maską (tam gdzie struktura ma jedynki) wszystkie piksele są równe 1,
        # to ustawiamy dany piksel wyniku na 1 (inaczej zostaje 0)
        for i in range(binary_img.shape[0]):
            for j in range(binary_img.shape[1]):
                region = padded_img[i:i + structure.shape[0], j:j + structure.shape[1]]
                if np.array_equal(region[structure == 1], np.ones(np.sum(structure))):
                    result[i, j] = 1
        return result

    def dilation(self, binary_img, structure):
        """
        Wykonuje operację dylatacji na podanym obrazie binarnym przy użyciu zadanego elementu strukturalnego.
        Zwraca obraz po dylatacji w postaci tablicy NumPy (wartości 0 i 1).
        """
        # Obliczenie wielkości obramowania na podstawie wymiarów elementu strukturalnego
        pad_h, pad_w = structure.shape[0] // 2, structure.shape[1] // 2

        # Dodanie obramowania do obrazu (wypełnienie zerami)
        padded_img = np.pad(binary_img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

        # Przygotowanie pustej macierzy na wynik dylatacji
        result = np.zeros_like(binary_img)

        # Dla każdego piksela obrazu sprawdzana jest lokalna okolica (region)for i in range(binary_img.shape[0]):
        # Jeśli w regionie pod mask (tam gdzie struktura ma jedynki) występuje chociaż jedno 1,
        # to ustawiamy dany piksel wyniku na 1 (inaczej zostaje 0)
        for i in range(binary_img.shape[0]):
            for j in range(binary_img.shape[1]):
                region = padded_img[i:i + structure.shape[0], j:j + structure.shape[1]]
                if np.any(region[structure == 1]):
                    result[i, j] = 1
        return result

    def pupil_morphologically_clean(self, binary_pupil):
        """
        Czyści binarny obraz źrenicy za pomocą operacji morfologicznych (erozja, dylatacja, otwarcie, zamknięcie) -
        - przy użyciu ręcznie zaimplementowanych funkcji 'erosion' i 'dilation'.
        Zwraca binarny obraz źrenicy w postaci tablicy NumPy (wartości 0 i 255).
        """
        # Konwersja obrazu na postać logiczną (0/1)
        binary_bool = (binary_pupil > 0).astype(int)

        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=int)

        # Liczba iteracji otwarcia i zamknięcia pobierana jest z wartości ustawionych przez użytkownika
        open_size = self.open_strength_pupil.get()
        close_size = self.close_strength_pupil.get()

        # Wykonanie operacji otwarcia (erozja → dylatacja) zadaną liczbę razy (open_size) w celu usunięcia
        # drobnych zakłóceń i szumów
        opened = binary_bool.copy()
        for _ in range(open_size):
            opened = self.erosion(opened, structure)
        for _ in range(open_size):
            opened = self.dilation(opened, structure)

        # Wykonanie operacji zamknięcia (dylatacja → erozja) zadaną liczbę razy (close_size) w celu wypełnienia
        # niewielkich dziur i ujednolicenia konturów
        closed = opened.copy()
        for _ in range(close_size):
            closed = self.dilation(closed, structure)
        for _ in range(close_size):
            closed = self.erosion(closed, structure)

        return (closed * 255).astype(np.uint8)

    def pupil_morphologically_clean2(self, binary_pupil):
        """
        Czyści binarny obraz źrenicy za pomocą gotowych operacji z biblioteki SciPy:
        binary_closing, binary_opening, generate_binary_structure, iterate_structure.
        Zwraca binarny obraz źrenicy w postaci tablicy NumPy (wartości 0 i 255).
        """
        binary_bool = binary_pupil.astype(bool)
        structure = generate_binary_structure(2, 1)
        open_size = self.open_strength_pupil.get()
        close_size = self.close_strength_pupil.get()

        # Struktury (maski) dla operacji otwarcia i zamknięcia na podstawie zdefiniowanej liczby iteracji
        structure_open = iterate_structure(structure, open_size)
        structure_close = iterate_structure(structure, close_size)

        # 1. Otwarcie, 2. Zamknięcie
        cleaned = binary_opening(binary_bool, structure=structure_open)
        cleaned = binary_closing(cleaned, structure=structure_close)

        return (cleaned * 255).astype(np.uint8)

    def pupil_get_largest_blob(self, binary_img):
        """
        Wybiera największy fragment (blob) z binarnego obrazu źrenicy.
        Zwraca maskę wybranego bloba jako obraz binarny (0 i 255).
        """
        # Etykietowanie wszystkich spójnych obszarów (blobów) w obrazie binarnym
        labeled_array, num_features = label(binary_img)

        # Jeśli nie wykryto żadnych obszarów, zwracany jest oryginalny obraz
        if num_features == 0:
            return binary_img

        # Obliczenie rozmiaru (liczby pikseli) dla każdego z wykrytych obszarów
        sizes = [(labeled_array == i).sum() for i in range(1, num_features + 1)]

        # Znalezienie etykiety obszaru o największym rozmiarze
        max_label = np.argmax(sizes) + 1

        # Utworzenie maski największego obszaru i konwersja na obraz binarny (0 i 255)
        largest_pupil = (labeled_array == max_label).astype(np.uint8) * 255
        return largest_pupil

    def get_horizontal_projection(self, binary):
        """
        Oblicza projekcję poziomą obrazu binarnego (suma wartości wzdłuż kolumn).
        Zwraca wektor sum dla każdego wiersza.
        """
        return np.sum(binary, axis=1)

    def get_vertical_projection(self, binary):
        """
        Oblicza projekcję pionową obrazu binarnego (suma wartości wzdłuż wierszy).
        Zwraca wektor sum dla każdej kolumny.
        """
        return np.sum(binary, axis=0)

    def get_diagonal_projection_45(self, binary):
        """
        Oblicza projekcję wzdłuż przekątnych o kącie 45° (↘).
        Dla każdej przekątnej sumuje wartości pikseli leżących na tej przekątnej.
        """
        return np.array([np.sum(np.diagonal(binary, offset=o)) for o in range(-binary.shape[0] + 1, binary.shape[1])])

    def get_diagonal_projection_135(self, binary):
        """
        Oblicza projekcję wzdłuż przekątnych o kącie 135° (↙).
        Najpierw obraz jest odbijany w pionie (flipud), a następnie obliczana jest suma dla każdej przekątnej.
        """
        flipped = np.flipud(binary)
        return np.array(
            [np.sum(np.diagonal(flipped, offset=o)) for o in range(-flipped.shape[0] + 1, flipped.shape[1])])

    def get_pupil_center_by_projections(self, binary_pupil):
        """
        Określa przybliżone położenie środka źrenicy na podstawie projekcji (pionowej, poziomej oraz dwóch przekątnych).
        Wybierane są dwie najmocniejsze projekcje (największe sumy) i na ich podstawie obliczany jest środek.
        Zwraca współrzędne środka źrenicy w postaci (x, y).
        """
        # Obliczenie projekcji w czterech kierunkach
        vertical_proj = self.get_vertical_projection(binary_pupil)
        horizontal_proj = self.get_horizontal_projection(binary_pupil)
        diag45 = self.get_diagonal_projection_45(binary_pupil)
        diag135 = self.get_diagonal_projection_135(binary_pupil)

        # Zebranie maksymalnych wartości projekcji oraz ich indeksów
        projections = {
            "vertical": (np.max(vertical_proj), int(np.argmax(vertical_proj))),
            "horizontal": (np.max(horizontal_proj), int(np.argmax(horizontal_proj))),
            "diag45": (np.max(diag45), int(np.argmax(diag45))),
            "diag135": (np.max(diag135), int(np.argmax(diag135)))
        }

        # Sortowanie projekcji od najsilniejszej do najsłabszej (według maksymalnej sumy liczby białych pikseli)
        # i następnie wybór dwóch najsilniejszych
        sorted_projections = sorted(projections.items(), key=lambda x: x[1][0], reverse=True)
        (first_name, (first_value, first_idx)) = sorted_projections[0]
        (second_name, (second_value, second_idx)) = sorted_projections[1]

        def index_to_xy(proj_name, idx):
            """
            Mapuje indeks z projekcji na współrzędne (x, y), w zależności od rodzaju projekcji.
            W przypadku pionowej lub poziomej projekcji określony jest tylko jeden z wymiarów (x lub y), drugi jest ustawiany na None.
            Dla przekątnych obliczane są obie współrzędne.
            """
            if proj_name == "vertical":
                return idx, None
            elif proj_name == "horizontal":
                return None, idx
            elif proj_name == "diag45":
                x = (binary_pupil.shape[1] + idx) // 2
                y = (binary_pupil.shape[0] - 1 - idx) // 2
                return x, y
            elif proj_name == "diag135":
                x = (binary_pupil.shape[1] + idx) // 2
                y = (binary_pupil.shape[0] + idx) // 2
                return x, y

        # Pobranie współrzędnych z dwóch najmocniejszych projekcji
        first_x, first_y = index_to_xy(first_name, first_idx)
        second_x, second_y = index_to_xy(second_name, second_idx)

        # Uzupełnienie brakujących współrzędnych, jeśli jedna z projekcji podała tylko x lub y
        center_x = first_x if first_x is not None else second_x
        center_y = first_y if first_y is not None else second_y

        # Zwrócenie współrzędnych środka jako liczby całkowite
        return int(center_x), int(center_y)

    def pupil_draw_detected(self, img_resized, binary_pupil):
        """
        Rysuje wykryty okrąg reprezentujący granicę źrenicy na kopii obrazu wejściowego.
        Środek i promień obliczane są na podstawie podanego obrazu binarnego źrenicy.
        Zaktualizowany obraz z narysowanym okręgiem wyświetlany jest na przygotowanym wcześniej canvasie.
        """
        # Wybór metody obliczania środka źrenicy: projekcje lub centroid (środek masy)
        if self.center_method.get() == "projection":
            center_x, center_y = self.get_pupil_center_by_projections(binary_pupil)
        else:
            coords = np.column_stack(np.where(binary_pupil == 255))
            if coords.size > 0:
                center_y, center_x = coords.mean(axis=0)
            else:
                center_x = center_y = 0

        # Obliczenie promienia jako największej odległości od środka źrenicy do punktów, w których maska źrenicy ma wartość 255
        coords = np.column_stack(np.where(binary_pupil == 255))
        if coords.size > 0:
            radius = np.max(np.linalg.norm(coords - [center_y, center_x], axis=1))
        else:
            radius = 0

        # Zapamiętanie obliczonego środka i promienia; utworzenie kopii obrazu wejściowego, na której zostanie narysowany okrąg
        self.pupil_center = (center_x, center_y)
        self.pupil_radius = radius
        marked_img = img_resized.copy()
        draw = ImageDraw.Draw(marked_img)

        # Rysowanie okręgu wokół wykrytej źrenicy (czerwony kontur)
        draw.ellipse(
            [(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)],
            outline="red", width=2
        )

        # Wyświetlanie zaktualizowanego obrazu z zaznaczoną źrenicą
        self.pupil_detected_img_tk = ImageTk.PhotoImage(marked_img)
        self.canvas_pupil_detected.delete("all")
        self.canvas_pupil_detected.create_image(0, 0, anchor=tk.NW, image=self.pupil_detected_img_tk)
        self.canvas_pupil_detected.image = self.pupil_detected_img_tk

    # ------------------------------- TĘCZÓWKA ---------------------------------
    def detect_iris(self):
        """
        Wykonuje pełną sekwencję detekcji tęczówki przy użyciu funkcji pomocniczych.
        Wyniki każdego kroku wyświetlane są na odpowiednich kanwach interfejsu graficznego.
        """
        if self.img_original is None:
            return

        X_I = self.xi_value_default.get()  # wybrany przez użytkownika współczynnik czułości binaryzacji dla tęczówki
        img_resized = self.img_original.resize((self.display_width, self.display_height))

        # Krok 1:
        # Wygenerowanie binarnego obrazu tęczówki na podstawie oryginalnego obrazu oka
        # za pomocą funkcji 'iris_detect_binary'.
        binary_iris = self.iris_detect_binary(img_resized, X_I)
        binary_img = Image.fromarray(binary_iris)
        self.iris_img_tk = ImageTk.PhotoImage(binary_img)
        self.canvas_iris.delete("all")
        self.canvas_iris.create_image(0, 0, anchor=tk.NW, image=self.iris_img_tk)
        self.canvas_iris.image = self.iris_img_tk

        # Krok 2:
        # Czyszczenie binarnego obrazu tęczówki za pomocą operacji morfologicznych,
        # przy użyciu funkcji 'iris_morphologically_clean'.
        cleaned_iris = self.iris_morphologically_clean(binary_iris)
        cleaned_img = Image.fromarray(cleaned_iris)
        self.iris_cleaned_img_tk = ImageTk.PhotoImage(cleaned_img)
        self.canvas_iris_cleaned.delete("all")
        self.canvas_iris_cleaned.create_image(0, 0, anchor=tk.NW, image=self.iris_cleaned_img_tk)
        self.canvas_iris_cleaned.image = self.iris_cleaned_img_tk

        # Krok 3:
        # Wybór największego i najlepiej dopasowanego obszaru (blob) jako kandydata na tęczówkę,
        # przy użyciu funkcji 'iris_get_largest_blob'.
        largest_iris = self.iris_get_largest_blob(cleaned_iris, self.pupil_center, self.pupil_radius)
        blob_img = Image.fromarray(largest_iris)
        self.iris_largest_blob_img_tk = ImageTk.PhotoImage(blob_img)
        self.canvas_iris_largest_blob.delete("all")
        self.canvas_iris_largest_blob.create_image(0, 0, anchor=tk.NW, image=self.iris_largest_blob_img_tk)
        self.canvas_iris_largest_blob.image = self.iris_largest_blob_img_tk

        # Krok 4:
        # Narysowanie konturu wykrytego obszaru tęczówki na oryginalnym obrazie oka
        # przy użyciu funkcji 'iris_draw_detected'.
        self.iris_draw_detected(img_resized, largest_iris)

    def iris_detect_binary(self, image, X_I):
        """
        Generuje binarny obraz tęczówki na podstawie podanego jako argument obrazu oka
        i współczynnika czułości binaryzacji (białe piksele - obszar potencjalnej tęczówki).
        Zwraca binarny obraz tęczówki w postaci tablicy NumPy (wartości 0 i 255).
        """
        # Konwersja obrazu RGB na skalę szarości
        img_np = np.array(image)
        gray_np = np.dot(img_np[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        # Obliczenie progu binaryzacji na podstawie średniej jasności obrazu i parametru X_I (współczynnika czułości)
        P = np.mean(gray_np)
        P_I = P / X_I

        # Piksele ciemniejsze od progu uznawane są za fragmenty tęczówki i kolorowane na biało
        binary_iris = np.where(gray_np < P_I, 255, 0).astype(np.uint8)

        if self.pupil_center is not None and self.pupil_radius is not None:
            center_x, center_y = self.pupil_center
            rr, cc = np.ogrid[:binary_iris.shape[0], :binary_iris.shape[1]]
            distance = np.sqrt((cc - center_x) ** 2 + (rr - center_y) ** 2)  # macierz odległości każdego piksela od środka źrenicy

            # Wykluczenie obszaru źrenicy na podstawie wcześniej wykrytego jej środka i promienia
            binary_iris[distance <= self.pupil_radius] = 0

            # Ograniczenie obszaru detekcji do koła o promieniu równym 2.5×promień źrenicy
            max_iris_radius = 2.5 * self.pupil_radius
            binary_iris[distance >= max_iris_radius] = 0

        return binary_iris

    def iris_morphologically_clean(self, binary_iris):
        """
        Czyści binarny obraz tęczówki za pomocą operacji morfologicznych (erozja, dylatacja, otwarcie, zamknięcie) -
        - przy użyciu ręcznie zaimplementowanych funkcji 'erosion' i 'dilation'.
        Zwraca binarny obraz tęczówki w postaci tablicy NumPy (wartości 0 i 255).
        """
        # Konwersja obrazu na postać logiczną (0/1)
        binary_bool = (binary_iris > 0).astype(int)
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=int)

        # Liczba iteracji otwarcia i zamknięcia pobierana jest z wartości ustawionych przez użytkownika
        open_size = self.open_strength_iris.get()
        close_size = self.close_strength_iris.get()

        # Wykonanie operacji otwarcia (erozja → dylatacja) zadaną liczbę razy (open_size) w celu usunięcia
        # drobnych zakłóceń i szumów
        opened = binary_bool.copy()
        for _ in range(open_size):
            opened = self.erosion(opened, structure)
        for _ in range(open_size):
            opened = self.dilation(opened, structure)

        # Wykonanie operacji zamknięcia (dylatacja → erozja) zadaną liczbę razy (close_size) w celu wypełnienia
        # niewielkich dziur i ujednolicenia konturów
        closed = opened.copy()
        for _ in range(close_size):
            closed = self.dilation(closed, structure)
        for _ in range(close_size):
            closed = self.erosion(closed, structure)

        return (closed * 255).astype(np.uint8)

    def iris_morphologically_clean2(self, binary_iris):
        """
        Czyści binarny obraz tęczówki za pomocą gotowych operacji z biblioteki SciPy:
        binary_closing, binary_opening, generate_binary_structure, iterate_structure.
        Zwraca binarny obraz tęczówki w postaci tablicy NumPy (wartości 0 i 255).
        """
        binary_bool = binary_iris.astype(bool)
        structure = generate_binary_structure(2, 1)

        open_size = self.open_strength_iris.get()
        close_size = self.close_strength_iris.get()

        # Struktury (maski) dla operacji otwarcia i zamknięcia na podstawie zdefiniowanej liczby iteracji
        structure_open = iterate_structure(structure, open_size)
        structure_close = iterate_structure(structure, close_size)

        # 1. Otwarcie, 2. Zamknięcie
        cleaned = binary_opening(binary_bool, structure=structure_open)
        cleaned = binary_closing(cleaned, structure=structure_close)

        return (cleaned * 255).astype(np.uint8)

    def iris_get_largest_blob(self, binary_iris, pupil_center, pupil_radius):
        """
        Wybiera największy, dobrze położony fragment (blob) z binarnego obrazu tęczówki na podstawie podanych
        współrzędnych środka i długości promienia źrenicy.
        Zwraca maskę wybranego bloba jako obraz binarny (0 i 255).
        """
        # Etykietowanie wszystkich spójnych obszarów w obrazie binarnym
        labeled_array, num_features = label(binary_iris)
        if num_features == 0:
            return binary_iris

        max_radius = 0
        selected_blob = np.zeros_like(binary_iris)

        # Dla każdego obszaru obliczany jest jego środek masy, promień oraz odległość od środka źrenicy,
        for label_idx in range(1, num_features + 1):
            mask = (labeled_array == label_idx)
            coords = np.column_stack(np.where(mask))

            if coords.size == 0:
                continue

            center_y, center_x = coords.mean(axis=0)
            radius = np.max(np.linalg.norm(coords - [center_y, center_x], axis=1))
            distance_to_pupil = np.linalg.norm([center_x - pupil_center[0], center_y - pupil_center[1]])

            # Warunki: bliskość środka źrenicy i odpowiednio duży rozmiar
            # Wybór bloba o największym promieniu spełniającego te warunki
            if distance_to_pupil < pupil_radius * 0.5 and radius > pupil_radius * 1.3:
                if radius > max_radius:
                    max_radius = radius
                    selected_blob = mask

        return selected_blob.astype(np.uint8) * 255

    def estimate_iris_radius_by_radial_projection(self, binary_iris, center, num_angles=360, min_consecutive_zeros=3):
        """
        Szacuje promień tęczówki metodą promieniowej projekcji (radial projection).
        Argumenty:
        - binary_iris - binarny obraz tęczówki,
        - center - współrzędne środka źrenicy,
        - num_angles - liczba kierunków (kątów) promieni wychodzących ze środka źrenicy, wzdłuż których wykonywane
          jest skanowanie,
        - min_consecutive_zeros - minimalna liczba kolejnych czarnych pikseli, które muszą zostać wykryte na promieniu,
          aby uznać, że granica tęczówki została osiągnięta.
        Zwraca znalezioną długość promienia tęczówki.
        """
        center_x, center_y = center
        binary_iris = binary_iris.astype(bool)

        height, width = binary_iris.shape
        angles = np.linspace(0, 2 * np.pi, num=num_angles, endpoint=False)

        radii = []  # lista do przechowywania oszacowanych długości promieni tęczówki z każdego kierunku

        for angle in angles:
            # Przeszukiwanie kolejnych pikseli wzdłuż każdego promienia w poszukiwaniu końca tęczówki
            consecutive_zeros = 0
            for r in range(int(self.pupil_radius) + 1, int(2.5 * self.pupil_radius)):
                x = int(center_x + r * np.cos(angle))
                y = int(center_y + r * np.sin(angle))

                # Przerwanie przeszukiwania danego promienia, jeśli aktualny punkt znajduje się poza granicami obrazu
                if x < 0 or x >= width or y < 0 or y >= height:
                    break

                if not binary_iris[y, x]:
                    consecutive_zeros += 1
                    # Koniec jest wykrywany, gdy napotkanych zostanie co najmniej
                    # `min_consecutive_zeros` czarnych pikseli (czyli tła)
                    if consecutive_zeros >= min_consecutive_zeros:
                        estimated_radius = r - min_consecutive_zeros // 2
                        # Zapis do listy tych długości promieni, przy których wykryto koniec tęczówki
                        radii.append(estimated_radius)
                        break
                else:
                    consecutive_zeros = 0

        # Jeśli znaleziono jakiekolwiek promienie, zwracana jest ich mediana jako szacowany promień tęczówki;
        # w przeciwnym razie zwracane jest 0 (brak detekcji).
        if radii:
            return np.median(radii)
        else:
            return 0

    def iris_draw_detected(self, img_resized, binary_iris):
        """
        Rysuje wykryty okrąg reprezentujący granicę tęczówki na kopii obrazu wejściowego.
        """
        # Sprawdzenie, czy wyznaczone są współrzędne środka źrenicy
        if self.pupil_center is None:
            return

        center_x, center_y = self.pupil_center
        coords = np.column_stack(np.where(binary_iris == 255))

        # Jeśli binarny obraz tęczówki zawiera dane, szacowany jest promień tęczówki metodą promieniowej projekcji
        if coords.size > 0:
            radius = self.estimate_iris_radius_by_radial_projection(binary_iris, self.pupil_center)
        else:
            radius = 0

        self.iris_radius = radius

        # Utworzenie kopii przeskalowanego obrazu i narysowanie na niej niebieskiego okręgu
        # reprezentującego wykrytą tęczówkę
        marked_img = img_resized.copy()
        draw = ImageDraw.Draw(marked_img)
        draw.ellipse(
            [(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)],
            outline="blue", width=2
        )

        # Utworzenie obiektu 'PhotoImage' i wyświetlenie wyniku na kanwie 'canvas_iris_detected'
        self.iris_detected_img_tk = ImageTk.PhotoImage(marked_img)
        self.canvas_iris_detected.delete("all")
        self.canvas_iris_detected.create_image(0, 0, anchor=tk.NW, image=self.iris_detected_img_tk)
        self.canvas_iris_detected.image = self.iris_detected_img_tk

    def unwrap_iris(self):
        """
        Rozwija wykrytą tęczówkę do prostokąta, wyświetla rozwinięcie na jednej kanwie oraz oryginalny obraz
        oka na drugiej, z naniesionymi okręgami wyznaczającymi granice źrenicy i tęczówki na podstawie wcześniejszych
        wyników detekcji.
        """
        if self.img_original is None or self.pupil_center is None or self.iris_radius is None:
            return

        # Pobranie wyznaczonych współrzędnych środka źrenicy, jej promienia i promienia tęczówki
        pupil_x, pupil_y = self.pupil_center
        pupil_r = self.pupil_radius
        iris_r = self.iris_radius

        img_resized = self.img_original.resize((self.display_width, self.display_height))
        img_np = np.array(img_resized)

        # Ustalenie liczby próbek w kierunku promieniowym (od źrenicy do tęczówki) i kątowym (pełen obrót wokół środka)
        num_radial_samples = 64
        num_angular_samples = 384

        # Generowanie równomiernych próbek kątów (0 do 2π) - dookoła źrenicy
        theta = np.linspace(0, 2 * np.pi, num_angular_samples, endpoint=False)
        # Generowanie równomiernych próbek promienia (od źrenicy do tęczówki)
        r = np.linspace(pupil_r, iris_r, num_radial_samples)
        # Tworzenie siatki współrzędnych promienia i kąta
        r_grid, theta_grid = np.meshgrid(r, theta)

        # Obliczenie współrzędnych X i Y na obrazie dla każdej pary (r, theta)
        x = pupil_x + r_grid.T * np.cos(theta_grid.T)
        y = pupil_y + r_grid.T * np.sin(theta_grid.T)

        unwrapped_channels = []  # Lista na rozwinięte kanały RGB

        # Iteracja przez każdy kanał (R, G, B)
        for c in range(3):
            channel = img_np[..., c]  # Wyodrębnienie kanału
            coords = np.array([y.flatten(), x.flatten()])  # Spłaszczenie współrzędnych
            # Interpolacja wartości pikseli w nowych współrzędnych
            unwrapped = map_coordinates(channel, coords, order=1, mode='reflect').reshape(
                (num_radial_samples, num_angular_samples))
            unwrapped_channels.append(unwrapped)  # Dodanie rozwiniętego kanału do listy

        # Połączenie trzech rozwiniętych kanałów w obraz RGB
        unwrapped_rgb = np.stack(unwrapped_channels, axis=2).astype(np.uint8)
        unwrapped_img = Image.fromarray(unwrapped_rgb)

        # Wyświetlanie otrzymanego rozwinięcia tęczówki na kanwie
        self.canvas_unwrapped.config(width=num_angular_samples, height=num_radial_samples)
        self.unwrapped_img_tk = ImageTk.PhotoImage(unwrapped_img)
        self.canvas_unwrapped.delete("all")
        self.canvas_unwrapped.create_image(0, 0, anchor=tk.NW, image=self.unwrapped_img_tk)
        self.canvas_unwrapped.image = self.unwrapped_img_tk

        # Naniesienie okręgów wyznaczających tęczówkę na oryginalny obraz oka
        draw_img = img_resized.copy()
        draw = ImageDraw.Draw(draw_img)
        draw.ellipse(
            [(pupil_x - pupil_r, pupil_y - pupil_r), (pupil_x + pupil_r, pupil_y + pupil_r)],
            outline="red", width=2
        )
        draw.ellipse(
            [(pupil_x - iris_r, pupil_y - iris_r), (pupil_x + iris_r, pupil_y + iris_r)],
            outline="blue", width=2
        )

        # Wyświetlenie obrazu z naniesionymi okręgami na drugiej kanwie w sekcji 'Iris Unwrapping'
        self.iris_drawn_tk = ImageTk.PhotoImage(draw_img)
        self.canvas_iris_drawn.delete("all")
        self.canvas_iris_drawn.create_image(0, 0, anchor=tk.NW, image=self.iris_drawn_tk)
        self.canvas_iris_drawn.image = self.iris_drawn_tk


if __name__ == '__main__':
    root = tk.Tk()
    app = IrisRecognitionApp(root)
    root.mainloop()


"""
--- Przykłady użycia dla zbioru tęczówek pobranego ze strony 'pages' ---

Dobre przypadki dla wykrywania źrenicy (przy opcji "centroid"):
- 76, 1, X_p=3.6, 1, 2
- 41, 2, X_p=3.4, 3, 3
- 34, 1, X_p=3.9, 1, 2

Dobre przypadki dla wykrywania tęczówki:
- 76, 1, X_i=1.3, 2, 3
- 41, 2, X_i=0.9, 1, 2
- 34, 1, X_i=0.9, 1, 2

Numery powyżej to odpowiednio:
- numer folderu, numer zdjęcia, X_p lub X_i, liczba iteracji otwarcia, liczba iteracji zamknięcia
"""