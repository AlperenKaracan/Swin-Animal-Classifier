import tkinter
import tkinter.filedialog
import tkinter.messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import timm
import os
from typing import List, Dict, Optional, Tuple, Any
import time
import webbrowser
from datetime import datetime
import sys
import cv2
import threading
import queue

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
MODEL_NAME: str = "swin_base_patch4_window7_224"
MODEL_CHECKPOINT_PATH: str = r"D:\best_checkpoint.pth"
NUM_CLASSES: int = 90
IMG_SIZE: int = 224
MEAN: List[float] = [0.485, 0.456, 0.406]
STD: List[float] = [0.229, 0.224, 0.225]
DEFAULT_MAX_HISTORY_ITEMS: int = 100
DEFAULT_TOP_K: int = 3
DEFAULT_LOW_CONFIDENCE_THRESHOLD: float = 0.50

IMAGE_DISPLAY_SIZE: Tuple[int, int] = (400, 400)
HISTORY_THUMBNAIL_SIZE: Tuple[int, int] = (64, 64)
LOW_CONFIDENCE_TEXT_COLOR: str = "#FFA500"
LOW_CONFIDENCE_PROGRESS_COLOR: str = "#FFA500"
SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
WEBCAM_PREDICTION_INTERVAL: float = 0.4
class_names: List[str] = [
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar',
    'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly',
    'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper',
    'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish',
    'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus',
    'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig',
    'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal',
    'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle',
    'whale', 'wolf', 'wombat', 'woodpecker', 'zebra'
]

if len(class_names) != NUM_CLASSES:
    print(f"Sınıf sayısı ({len(class_names)}) NUM_CLASSES ({NUM_CLASSES}) ile eşleşmiyor!")
    root_check = tkinter.Tk(); root_check.withdraw()
    tkinter.messagebox.showerror("Sınıf Hatası", f"Sınıf listesi  ({len(class_names)}) ile beklenen sınıf sayısı ({NUM_CLASSES}) eşleşmiyor.")
    root_check.destroy(); exit()

infer_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32), transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılacak cihaz: {device}")
except Exception as e:
    print(f"CPU kullanılacak {e}")
    device = torch.device("cpu")
def load_model(model_name: str, checkpoint_path: str, num_classes: int, device: torch.device) -> Optional[torch.nn.Module]:
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint dosyası bulunamadı: {checkpoint_path}")
        return None

    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        extracted_state_dict = None
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                extracted_state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                extracted_state_dict = checkpoint['state_dict']
            else:
                extracted_state_dict = checkpoint
        else:
            extracted_state_dict = checkpoint
        final_state_dict = {}
        for k, v in extracted_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            final_state_dict[name] = v

        model.load_state_dict(final_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        print(f"Model '{model_name}' başarıyla yüklendi ve '{device}' cihazına taşındı.")
        return model

    except Exception as e:
        print(f"Model yüklenirken bir sorun oluştu: {e}")
        return None


def predict_image(model: torch.nn.Module,
                  image_path: str,
                  class_names: List[str],
                  device: torch.device,
                  transform: transforms.Compose,
                  topk: int) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    if not model:
        return None, "Model yüklenemedi."
    if not os.path.exists(image_path):
        return None, f"Görüntü dosyası bulunamadı:\n{os.path.basename(image_path)}"
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Görüntü açılamadı veya dönüştürülemedi: {image_path} - Hata: {e}")
        return None, f"Görüntü dosyası okunamadı veya bozuk:\n{os.path.basename(image_path)}"
    try:
        img_t = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Görüntü dönüşümleri sırasında hata: {e}")
        return None, "Görüntü işlenirken bir hata oluştu."
    try:
        with torch.no_grad():
            outputs = model(img_t)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = probabilities.topk(topk, dim=1)
            top_probs_np = top_probs.cpu().numpy().flatten()
            top_indices_np = top_indices.cpu().numpy().flatten()

        results = []
        for idx, prob in zip(top_indices_np, top_probs_np):
            if 0 <= idx < len(class_names):
                class_name = class_names[idx].replace('_', ' ').capitalize()
                results.append({'class_name': class_name, 'probability': float(prob)})
            else:
                print(f"Geçersiz sınıf indeksi {idx}.")
                results.append({'class_name': f'Geçersiz Index ({idx})', 'probability': float(prob)})

        return results, None

    except Exception as e:
        print(f"Tahmin sırasında hata: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Model tahmini sırasında beklenmedik bir hata oluştu:\n{e}"



class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, master: 'AnimalClassifierApp', current_settings: Dict[str, Any], save_callback: callable):
        super().__init__(master)

        self.master_app = master
        self.save_callback = save_callback

        self.default_settings = {
             'max_history': DEFAULT_MAX_HISTORY_ITEMS,
             'top_k': DEFAULT_TOP_K,
             'low_conf_threshold': DEFAULT_LOW_CONFIDENCE_THRESHOLD
        }

        self.title("Uygulama Ayarları")
        self.geometry("380x250")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()

        self.grid_columnconfigure(1, weight=1)


        self.max_history_label = ctk.CTkLabel(self, text="Maksimum Geçmiş:")
        self.max_history_label.grid(row=0, column=0, padx=(20, 5), pady=(20, 5), sticky="w")
        self.max_history_entry = ctk.CTkEntry(self, width=100)
        self.max_history_entry.grid(row=0, column=1, padx=(5, 20), pady=(20, 5), sticky="e")
        self.max_history_entry.insert(0, str(current_settings.get('max_history', self.default_settings['max_history'])))


        self.top_k_label = ctk.CTkLabel(self, text="Gösterilecek Tahmin Sayısı:")
        self.top_k_label.grid(row=1, column=0, padx=(20, 5), pady=5, sticky="w")
        self.top_k_entry = ctk.CTkEntry(self, width=100)
        self.top_k_entry.grid(row=1, column=1, padx=(5, 20), pady=5, sticky="e")
        self.top_k_entry.insert(0, str(current_settings.get('top_k', self.default_settings['top_k'])))


        self.low_conf_label = ctk.CTkLabel(self, text="Güven Eşiği:")
        self.low_conf_label.grid(row=2, column=0, padx=(20, 5), pady=5, sticky="w")
        self.low_conf_entry = ctk.CTkEntry(self, width=100)
        self.low_conf_entry.grid(row=2, column=1, padx=(5, 20), pady=5, sticky="e")
        threshold_percent = current_settings.get('low_conf_threshold', self.default_settings['low_conf_threshold']) * 100
        self.low_conf_entry.insert(0, f"{threshold_percent:.1f}")


        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=(20, 20), sticky="ew")
        self.button_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.save_button = ctk.CTkButton(self.button_frame, text="Kaydet", command=self.save_settings)
        self.save_button.grid(row=0, column=2, padx=(5,0), sticky="ew")

        self.cancel_button = ctk.CTkButton(self.button_frame, text="İptal", command=self.cancel, fg_color="gray50", hover_color="gray40")
        self.cancel_button.grid(row=0, column=1, padx=5, sticky="ew")

        self.reset_button = ctk.CTkButton(self.button_frame, text="Varsayılan", command=self.reset_defaults, fg_color="transparent", border_width=1)
        self.reset_button.grid(row=0, column=0, padx=(0,5), sticky="ew")


        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self.after(50, lambda: self.max_history_entry.focus_set())

    def validate_and_get_settings(self) -> Optional[Dict[str, Any]]:
        try:
            max_history = int(self.max_history_entry.get().strip())
            if max_history <= 0:
                raise ValueError("Maksimum geçmiş pozitif bir tam sayı olmalı.")
            top_k = int(self.top_k_entry.get().strip())
            if not (0 < top_k <= self.master_app.NUM_CLASSES):
                raise ValueError(f"Top K değeri 1 ile {self.master_app.NUM_CLASSES} arasında olmalı.")

            low_conf_str = self.low_conf_entry.get().strip().replace(',', '.')
            low_conf_percent = float(low_conf_str)
            if not (0.0 <= low_conf_percent <= 100.0):
                raise ValueError("Düşük güven eşiği %0 ile %100 arasında olmalı.")
            low_conf_threshold = low_conf_percent / 100.0

            return {
                'max_history': max_history,
                'top_k': top_k,
                'low_conf_threshold': low_conf_threshold
            }

        except ValueError as e:
            tkinter.messagebox.showerror("Geçersiz Değer", str(e), parent=self)

            if "geçmiş" in str(e): self.max_history_entry.focus_set(); self.max_history_entry.select_range(0, 'end')
            elif "Top K" in str(e): self.top_k_entry.focus_set(); self.top_k_entry.select_range(0, 'end')
            elif "eşik" in str(e): self.low_conf_entry.focus_set(); self.low_conf_entry.select_range(0, 'end')
            return None
        except Exception as e:
            tkinter.messagebox.showerror("Hata", f"Beklenmedik bir hata oluştu: {e}", parent=self)
            return None

    def save_settings(self):
        new_settings = self.validate_and_get_settings()
        if new_settings:
            print("Ayarlar kaydediliyor:", new_settings)
            self.save_callback(new_settings)
            self.grab_release()
            self.destroy()

    def cancel(self):
        print("Ayarlar iptal edildi.")
        self.grab_release()
        self.destroy()

    def reset_defaults(self):
        print("Ayarlar varsayılana sıfırlanıyor.")
        self.max_history_entry.delete(0, tkinter.END)
        self.max_history_entry.insert(0, str(self.default_settings['max_history']))

        self.top_k_entry.delete(0, tkinter.END)
        self.top_k_entry.insert(0, str(self.default_settings['top_k']))

        self.low_conf_entry.delete(0, tkinter.END)
        threshold_percent = self.default_settings['low_conf_threshold'] * 100
        self.low_conf_entry.insert(0, f"{threshold_percent:.1f}")

class AnimalClassifierApp(ctk.CTk):

    NUM_CLASSES = NUM_CLASSES

    def __init__(self, model: Optional[torch.nn.Module], class_names: List[str], device: torch.device):
        super().__init__()

        self.model = model
        self.class_names = class_names
        self.device = device
        self.infer_transform = infer_transform
        self.current_image_path: Optional[str] = None
        self.current_ctk_image: Optional[ctk.CTkImage] = None
        self.current_predictions: Optional[List[Dict[str, Any]]] = None
        self.top_prediction_name: Optional[str] = None
        self.history_data: List[Dict[str, Any]] = []
        self.max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS
        self.top_k: int = DEFAULT_TOP_K
        self.low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD
        self.webcam_active = False
        self.webcam_capture: Optional[cv2.VideoCapture] = None
        self.webcam_thread: Optional[threading.Thread] = None
        self.webcam_queue = queue.Queue()
        self.default_text_color = ctk.ThemeManager.theme["CTkLabel"]["text_color"]
        self.default_progress_color = ctk.ThemeManager.theme["CTkProgressBar"]["progress_color"]
        self.default_top_pred_font = ctk.CTkFont(size=18, weight="bold")
        self.default_conf_font = ctk.CTkFont(size=14)
        self.settings_window: Optional[SettingsWindow] = None
        self.title("Yazlab3")
        self.geometry("1150x750"); self.minsize(950, 600)
        self.grid_columnconfigure(0, weight=5); self.grid_columnconfigure(1, weight=4)
        self.grid_columnconfigure(2, weight=3); self.grid_rowconfigure(0, weight=1)
        self.image_frame = ctk.CTkFrame(self, corner_radius=10, border_width=1)
        self.image_frame.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")
        self.image_frame.grid_rowconfigure(0, weight=1); self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.configure(fg_color=("gray90", "gray20"))

        self.image_label_initial_config = {
            "text": "Analiz için bir hayvan resmi yükleyin\n(JPG,PNG...)\nKlasör seçin veya Webcam'i başlatın.",
            "font": ctk.CTkFont(size=16),
            "text_color": "gray50",
            "image": None
        }
        self._recreate_image_label()

        self.controls_frame = ctk.CTkFrame(self, corner_radius=10)
        self.controls_frame.grid(row=0, column=1, padx=(10, 10), pady=20, sticky="nsew")
        self.controls_frame.grid_columnconfigure(0, weight=1)
        self.controls_frame.grid_rowconfigure(0, weight=0)
        self.controls_frame.grid_rowconfigure(1, weight=0)
        self.controls_frame.grid_rowconfigure(2, weight=1)
        self.controls_frame.grid_rowconfigure(3, weight=0)
        self.controls_frame.grid_rowconfigure(4, weight=0)

        self.title_label = ctk.CTkLabel(self.controls_frame, text="Tahmin Sonuçları", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="ew")

        self.filename_label = ctk.CTkLabel(self.controls_frame, text="Yüklenen Dosya: -", text_color="gray60", font=ctk.CTkFont(size=11), anchor="w", wraplength=300)
        self.filename_label.grid(row=1, column=0, padx=20, pady=(0, 15), sticky="ew")

        self.result_display_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.result_display_frame.grid(row=2, column=0, padx=20, pady=5, sticky="nsew")
        self.result_display_frame.grid_columnconfigure(0, weight=1); self.result_display_frame.grid_columnconfigure(1, weight=1)
        self.result_display_frame.grid_columnconfigure(2, weight=0);

        self.result_display_frame.grid_rowconfigure(0, weight=0)
        self.result_display_frame.grid_rowconfigure(1, weight=0)
        self.result_display_frame.grid_rowconfigure(2, weight=1)

        self.top_prediction_label = ctk.CTkLabel(self.result_display_frame, text="Tür: -", font=self.default_top_pred_font, anchor="w")
        self.top_prediction_label.grid(row=0, column=0, columnspan=2, pady=(10, 2), padx=(0,5), sticky="nw")

        self.search_button = ctk.CTkButton(self.result_display_frame, text="🔍", width=30, height=30,
                                           command=self.search_top_prediction_online, state="disabled",
                                           font=ctk.CTkFont(size=16), fg_color="transparent", border_width=1, text_color=("gray10", "gray90"))
        self.search_button.grid(row=0, column=2, pady=(8, 2), padx=5, sticky="ne")

        self.confidence_label = ctk.CTkLabel(self.result_display_frame, text="Güven: -%", font=self.default_conf_font, anchor="w")
        self.confidence_label.grid(row=1, column=0, pady=(0, 5), sticky="nw")

        self.confidence_bar = ctk.CTkProgressBar(self.result_display_frame, orientation="horizontal", height=12)
        self.confidence_bar.grid(row=1, column=1, columnspan=2, pady=(0, 5), padx=5, sticky="ew")
        self.confidence_bar.set(0); self.confidence_bar.configure(progress_color=self.default_progress_color)

        self.other_predictions_label = ctk.CTkLabel(self.result_display_frame, text="Diğer Olasılıklar:\n-", justify="left", anchor="nw", font=ctk.CTkFont(size=13))
        self.other_predictions_label.grid(row=2, column=0, columnspan=3, pady=(15, 5), sticky="nsew")


        self.button_subframe = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.button_subframe.grid(row=3, column=0, padx=20, pady=(15, 10), sticky="ew")

        self.button_subframe.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        self.load_button = ctk.CTkButton(self.button_subframe, text="🖼️ Resim Yükle", command=self.load_image_callback, height=40, font=ctk.CTkFont(size=14))
        self.load_button.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ew")
        self.predict_button = ctk.CTkButton(self.button_subframe, text="🧠 Tahmin Et", command=self.predict_callback, state="disabled", height=40, font=ctk.CTkFont(size=14, weight="bold"))
        self.predict_button.grid(row=0, column=1, padx=(5, 5), pady=5, sticky="ew")
        self.predict_folder_button = ctk.CTkButton(self.button_subframe, text="📁 Klasör Seç", command=self.predict_folder_callback, height=40, font=ctk.CTkFont(size=14))
        self.predict_folder_button.grid(row=0, column=2, padx=(5, 5), pady=5, sticky="ew")
        self.webcam_button = ctk.CTkButton(self.button_subframe, text="📷 Webcam Başlat", command=self.toggle_webcam_callback, height=40, font=ctk.CTkFont(size=14))
        self.webcam_button.grid(row=0, column=3, padx=(5, 5), pady=5, sticky="ew")

        self.settings_button = ctk.CTkButton(self.button_subframe, text="⚙️ Ayarlar", command=self.open_settings_window, height=40, font=ctk.CTkFont(size=14))
        self.settings_button.grid(row=0, column=4, padx=(5, 0), pady=5, sticky="ew")

        self.status_label = ctk.CTkLabel(self.controls_frame, text="Model bekleniyor...", text_color="gray", anchor='w', font=ctk.CTkFont(size=12))
        self.status_label.grid(row=4, column=0, padx=20, pady=(5, 10), sticky="ew")


        self.history_outer_frame = ctk.CTkFrame(self, corner_radius=10)
        self.history_outer_frame.grid(row=0, column=2, padx=(10, 20), pady=20, sticky="nsew")
        self.history_outer_frame.grid_rowconfigure(0, weight=0); self.history_outer_frame.grid_rowconfigure(1, weight=0)
        self.history_outer_frame.grid_rowconfigure(2, weight=1); self.history_outer_frame.grid_columnconfigure(0, weight=1)

        self.history_title_label = ctk.CTkLabel(self.history_outer_frame, text="Tahmin Geçmişi", font=ctk.CTkFont(size=18, weight="bold"))
        self.history_title_label.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="ew")
        self.clear_history_button = ctk.CTkButton(self.history_outer_frame, text="🗑️ Tüm Geçmişi Temizle", command=self._clear_all_history, height=30, font=ctk.CTkFont(size=12), fg_color="transparent", border_width=1, text_color=("red", "#FF5733"), state="disabled")
        self.clear_history_button.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="ew")
        self.history_frame = ctk.CTkScrollableFrame(self.history_outer_frame, label_text="", corner_radius=5)
        self.history_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.history_frame.grid_columnconfigure(0, weight=1)

        self._check_model_status(); self._update_history_display()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    def _recreate_image_label(self, config: Optional[Dict[str, Any]] = None):
        if hasattr(self, 'image_label') and self.image_label is not None and self.image_label.winfo_exists():
            self.image_label.destroy()

        config_to_use = (config or self.image_label_initial_config).copy()
        if 'image' not in config_to_use: config_to_use['image'] = None

        self.image_label = ctk.CTkLabel(self.image_frame, **config_to_use)
        self.image_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
    def on_closing(self):
        print("Uygulama kapatılıyor...")

        if self.settings_window is not None and self.settings_window.winfo_exists():
             print("Açık ayarlar penceresi kapatılıyor...")
             self.settings_window.destroy()

        if self.webcam_active:
            self.stop_webcam()

        if self.winfo_exists():
            self.update_idletasks()
            self.destroy()
        print("Pencere kapatıldı.")

    def _check_model_status(self):
        if self.model is None:
            self._update_status("❌ Model yüklenemedi.", "error")
            self._set_buttons_state("disabled")
        else:
            self._update_status("✅ Model hazır. Bir resim yükleyin, klasör seçin veya webcam'i başlatın.", "success")
            self._set_buttons_state("normal")

    def _update_status(self, message: str, level: str = "info"):
        color_map = {
            "info": self.default_text_color,
            "success": "#4CAF50",
            "warning": "#FFC107",
            "error": "#F44336",
            "processing": "gray60"
        }

        if self.winfo_exists() and hasattr(self, 'status_label') and self.status_label.winfo_exists():
             self.status_label.configure(text=message, text_color=color_map.get(level, self.default_text_color))
             self.update_idletasks()
    def _create_history_thumbnail(self, image_path: str) -> Optional[ctk.CTkImage]:
        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail(HISTORY_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            return ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        except FileNotFoundError:
            print(f"Geçmiş için dosya bulunamadı: {image_path}")
            return None
        except Exception as e:
            print(f"Geçmiş oluşturulamadı ({image_path}): {e}")
            return None
    def _set_buttons_state(self, state: str):
        if not self.winfo_exists(): return

        base_state = "disabled" if self.model is None else state
        non_webcam_state = "disabled" if self.webcam_active else base_state
        webcam_button_state = base_state
        settings_button_state = "disabled" if self.model is None else "normal"

        if hasattr(self, 'load_button'): self.load_button.configure(state=non_webcam_state)
        if hasattr(self, 'predict_folder_button'): self.predict_folder_button.configure(state=non_webcam_state)
        if hasattr(self, 'webcam_button'): self.webcam_button.configure(state=webcam_button_state)
        if hasattr(self, 'settings_button'): self.settings_button.configure(state=settings_button_state)
        predict_state = "disabled"
        if self.model and not self.webcam_active and self.current_image_path:
            predict_state = "normal"
        if hasattr(self, 'predict_button'): self.predict_button.configure(state=predict_state)

        search_state = "disabled"
        if self.model and self.top_prediction_name:
            search_state = "normal"
        if hasattr(self, 'search_button'): self.search_button.configure(state=search_state)


        clear_history_state = "disabled"
        if self.model and not self.webcam_active and self.history_data:
            clear_history_state = "normal"
        if hasattr(self, 'clear_history_button'): self.clear_history_button.configure(state=clear_history_state)

    def load_image_callback(self) -> None:
        if self.webcam_active: return

        try:
            file_path = tkinter.filedialog.askopenfilename(
                title="Bir Hayvan Resmi Seçin",
                filetypes=[("Resim Dosyaları", "*" + " *".join(SUPPORTED_IMAGE_EXTENSIONS)),
                           ("Tüm Dosyalar", "*.*")]
            )
            if not file_path:
                self._update_status("Resim seçimi iptal edildi.", "warning")
                return

            self.clear_results()
            self.current_image_path = file_path
            self.current_predictions = None
            self._update_status(f"Resim yükleniyor: {os.path.basename(file_path)}", "processing")

            img = Image.open(file_path)
            img_copy = img.copy()
            img_copy.thumbnail(IMAGE_DISPLAY_SIZE, Image.Resampling.LANCZOS)
            self.current_ctk_image = ctk.CTkImage(light_image=img_copy, dark_image=img_copy, size=img_copy.size)

            self.image_label.configure(image=self.current_ctk_image, text="")
            self.filename_label.configure(text=f"Yüklenen Dosya: {os.path.basename(file_path)}")

            self._update_status("Resim yüklendi. Tahmin etmeye hazır.", "success")
            self._set_buttons_state("normal")

        except FileNotFoundError:
            filename = os.path.basename(file_path) if 'file_path' in locals() else 'Bilinmeyen'
            self._update_status(f"Hata: Dosya bulunamadı - {filename}", "error")
            self._recreate_image_label(config={
                "text": "❌ Resim yüklenemedi",
                "font": self.image_label_initial_config["font"], "text_color": LOW_CONFIDENCE_TEXT_COLOR
            })
            self.filename_label.configure(text="Yüklenen Dosya: -")
            self.current_image_path = None; self.current_ctk_image = None; self.current_predictions = None
            self.clear_results()
            self._set_buttons_state("normal")
        except Exception as e:
            print(f"Resim yükleme hatası: {e}")
            self._update_status(f"Hata: Resim yüklenemedi ({type(e).__name__}). Dosya bozuk olabilir.", "error")
            self._recreate_image_label(config={
                "text": "❌ Geçersiz veya bozuk resim dosyası.",
                "font": self.image_label_initial_config["font"], "text_color": LOW_CONFIDENCE_TEXT_COLOR
            })
            self.filename_label.configure(text="Yüklenen Dosya: -")
            self.current_image_path = None; self.current_ctk_image = None; self.current_predictions = None
            self.clear_results()
            self._set_buttons_state("normal")
    def predict_callback(self) -> None:
        if self.webcam_active: return
        if not self.current_image_path or not self.model:
            self._update_status("Geçerli bir resim yükleyin.", "warning")
            return

        self._update_status("🧠 Tahmin ediliyor...", "processing")
        self._set_buttons_state("disabled")
        start_time = time.monotonic()

        predictions, error_message = predict_image(
            self.model, self.current_image_path, self.class_names, self.device, self.infer_transform, topk=self.top_k
        )
        duration = time.monotonic() - start_time

        if error_message:
            self._update_status(f"Hata: {error_message}", "error")
            self.clear_results()
            self.current_predictions = None
        elif predictions:
            self._update_status(f"✅ Tahmin başarıyla tamamlandı ({duration:.2f} saniye).", "success")
            self.current_predictions = predictions
            self.display_results(predictions)
            self._add_to_history(self.current_image_path, predictions)
        else:
            self._update_status("Tahmin yapılamadı.", "error")
            self.clear_results()
            self.current_predictions = None

        if not self.webcam_active:
            self._set_buttons_state("normal")
    def predict_folder_callback(self) -> None:
        if self.webcam_active: return
        if not self.model:
            self._update_status("Model yüklenemediği için klasörden tahmin yapılamıyor.", "error")
            return

        folder_path = tkinter.filedialog.askdirectory(title="Tahmin Edilecek Resimlerin Olduğu Klasörü Seçin")
        if not folder_path:
            self._update_status("Klasör seçimi iptal edildi.", "warning")
            return

        print(f"Seçilen klasör: {folder_path}")
        self._update_status(f"Klasör taranıyor: {os.path.basename(folder_path)}...", "processing")
        self._set_buttons_state("disabled")

        image_files = []
        try:
            all_files = os.listdir(folder_path)
            image_files = [
                os.path.join(folder_path, f) for f in all_files
                if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(folder_path, f))
            ]
            image_files.sort()
        except Exception as e:
            print(f"Klasör okunurken hata: {e}")
            self._update_status(f"Hata: Klasör okunurken sorun oluştu - {e}", "error")
            tkinter.messagebox.showerror("Klasör Hatası", f"Seçilen klasör okunurken bir hata oluştu:\n{e}")
            self._set_buttons_state("normal")
            return

        if not image_files:
            self._update_status("Seçilen klasörde desteklenen formatta resim bulunamadı.", "warning")
            tkinter.messagebox.showinfo("Klasör Boş", f"Seçtiğiniz klasörde desteklenen formatlarda ({', '.join(SUPPORTED_IMAGE_EXTENSIONS)}) resim dosyası bulunamadı.")
            self._set_buttons_state("normal")
            return

        total_images = len(image_files)
        print(f"Klasörde {total_images} adet resim bulundu. Tahminler başlıyor...")
        processed_count = 0; errors_count = 0
        start_time = time.monotonic()
        new_history_entries = []

        for i, image_path in enumerate(image_files):
            filename = os.path.basename(image_path)

            if i % 5 == 0 or i == total_images - 1:
                 self._update_status(f"🧠 Tahmin: {filename} ({i+1}/{total_images})", "processing")
                 self.update_idletasks()


            predictions, error_message = predict_image(
                self.model, image_path, self.class_names, self.device, self.infer_transform, topk=self.top_k
            )

            if predictions and not error_message:
                processed_count += 1
                thumbnail_img = self._create_history_thumbnail(image_path)
                if thumbnail_img:
                    history_entry = {
                        'timestamp': datetime.now(), 'image_path': image_path, 'thumbnail': thumbnail_img,
                        'top_prediction': predictions[0]['class_name'], 'confidence': predictions[0]['probability'],
                        'all_predictions': predictions
                    }
                    new_history_entries.append(history_entry)
                else:
                     print(f"UYARI: Tahmin ({filename}) başarılı, ancak geçmiş için resim oluşturulamadı.")
            else:
                errors_count += 1
                print(f"Hata ({filename}): {error_message or 'Bilinmeyen tahmin hatası'}")

        duration = time.monotonic() - start_time

        if new_history_entries:
            print(f"{len(new_history_entries)} yeni tahmin geçmişe ekleniyor...")
            self.history_data = new_history_entries + self.history_data

            while len(self.history_data) > self.max_history_items: self.history_data.pop()
            self._update_history_display()

        self._set_buttons_state("normal")

        success_message = f"✅ Klasör tahmini tamamlandı ({duration:.2f} saniye)."
        result_summary = f"{processed_count} resim işlendi ({len(new_history_entries)} geçmişe eklendi), {errors_count} hata oluştu."
        final_status = f"{success_message} {result_summary}"
        final_level = "success" if errors_count == 0 else ("warning" if processed_count > 0 else "error")
        self._update_status(final_status, final_level)
        print(final_status)

        if new_history_entries:
            last_entry = new_history_entries[0]
            self._load_from_history(last_entry)
        elif not new_history_entries and errors_count == total_images:

            self.clear_results()
            self._recreate_image_label()
            self.filename_label.configure(text="Yüklenen Dosya: -")
            self.current_image_path = None; self.current_ctk_image = None; self.current_predictions = None
        elif self.history_data:
             self._load_from_history(self.history_data[0])
    def display_results(self, predictions: List[Dict[str, Any]]) -> None:
        if not predictions or not self.winfo_exists():
            self.clear_results()
            return

        self.current_predictions = predictions
        top_pred = predictions[0]
        self.top_prediction_name = top_pred['class_name']
        probability = top_pred['probability']

        is_low_confidence = probability < self.low_confidence_threshold
        pred_text = f"Tür: {self.top_prediction_name}"
        conf_text = f"Güven: {probability:.2%}"
        top_pred_color = self.default_text_color
        conf_color = self.default_text_color
        progress_color = self.default_progress_color

        if is_low_confidence:
            pred_text += " (Düşük Güven!)"
            conf_text += " !"
            top_pred_color = LOW_CONFIDENCE_TEXT_COLOR
            conf_color = LOW_CONFIDENCE_TEXT_COLOR
            progress_color = LOW_CONFIDENCE_PROGRESS_COLOR

        if self.top_prediction_label.winfo_exists(): self.top_prediction_label.configure(text=pred_text, text_color=top_pred_color)
        if self.confidence_label.winfo_exists(): self.confidence_label.configure(text=conf_text, text_color=conf_color)
        if self.confidence_bar.winfo_exists():
             self.confidence_bar.configure(progress_color=progress_color)
             self.confidence_bar.set(float(probability))

        other_preds_text = "Diğer Olasılıklar:\n"
        if len(predictions) > 1:
            for pred in predictions[1:]:
                other_prob = pred['probability']
                other_name = pred['class_name']

                low_conf_marker = " *" if other_prob < self.low_confidence_threshold else ""
                other_preds_text += f" • {other_name} ({other_prob:.1%}){low_conf_marker}\n"
        else:
            other_preds_text += "-\n"

        if self.other_predictions_label.winfo_exists(): self.other_predictions_label.configure(text=other_preds_text.strip())

        self._set_buttons_state("normal")
    def clear_results(self) -> None:
        self.top_prediction_name = None
        self.current_predictions = None
        if not self.winfo_exists(): return

        if hasattr(self, 'top_prediction_label') and self.top_prediction_label.winfo_exists(): self.top_prediction_label.configure(text="Tür: -", text_color=self.default_text_color)
        if hasattr(self, 'confidence_label') and self.confidence_label.winfo_exists(): self.confidence_label.configure(text="Güven: -%", text_color=self.default_text_color)
        if hasattr(self, 'other_predictions_label') and self.other_predictions_label.winfo_exists(): self.other_predictions_label.configure(text="Diğer Olasılıklar:\n-")
        if hasattr(self, 'confidence_bar') and self.confidence_bar.winfo_exists():
             self.confidence_bar.configure(progress_color=self.default_progress_color)
             self.confidence_bar.set(0)

        self._set_buttons_state("normal")
    def search_top_prediction_online(self) -> None:
        if self.top_prediction_name:
            try:
                query_name = self.top_prediction_name.replace(" (Düşük Güven!)", "").strip()
                query = query_name.replace(" ", "+")
                url = f"https://www.google.com/search?tbm=isch&q={query}+animal"
                print(f"Web'de aranıyor: {url}")
                webbrowser.open_new_tab(url)
            except Exception as e:
                print(f"Web tarayıcısı açılamadı: {e}")
                self._update_status("Web tarayıcısı açılamadı.", "error")
                tkinter.messagebox.showerror("Tarayıcı Hatası", f"Web tarayıcısı açılamadı:\n{e}")
        else:
            print("Aranacak hayvan ismi bulunamadı.")
            self._update_status("Önce bir tahmin yapmalısınız.", "warning")
    def _add_to_history(self, image_path: str, predictions: List[Dict[str, Any]]):
        if self.webcam_active: return

        thumbnail_img = self._create_history_thumbnail(image_path)
        if thumbnail_img:
            history_entry = {
                'timestamp': datetime.now(), 'image_path': image_path, 'thumbnail': thumbnail_img,
                'top_prediction': predictions[0]['class_name'], 'confidence': predictions[0]['probability'],
                'all_predictions': predictions
            }
            self.history_data.insert(0, history_entry)

            while len(self.history_data) > self.max_history_items: self.history_data.pop()
            self._update_history_display()
        else:
            print(f"Tahmin başarılı ({os.path.basename(image_path)}) ama geçmiş için resim oluşturulamadı.")

    def _update_history_display(self) -> None:
        if not self.winfo_exists() or not hasattr(self, 'history_frame') or not self.history_frame.winfo_exists():
            return

        for widget in self.history_frame.winfo_children():
            widget.destroy()

        if not self.history_data:
            no_history_label = ctk.CTkLabel(self.history_frame, text="Henüz tahmin yapılmadı.", text_color="gray")
            no_history_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        else:
            for i, entry in enumerate(self.history_data):
                item_frame = ctk.CTkFrame(self.history_frame, corner_radius=5, border_width=1)
                item_frame.grid(row=i, column=0, padx=5, pady=(5, 0), sticky="ew")
                item_frame.grid_columnconfigure(1, weight=1)

                thumb_image = entry.get('thumbnail')
                thumb_label = ctk.CTkLabel(item_frame, text="" if thumb_image else "X", image=thumb_image, width=HISTORY_THUMBNAIL_SIZE[0], height=HISTORY_THUMBNAIL_SIZE[1])
                thumb_label.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky="nsw")

                confidence = entry['confidence']

                is_low_confidence = confidence < self.low_confidence_threshold
                pred_text = f"{entry['top_prediction']} ({confidence:.1%})"
                item_text_color = self.default_text_color
                if is_low_confidence:
                    pred_text += " (Düşük!)"
                    item_text_color = LOW_CONFIDENCE_TEXT_COLOR

                pred_label = ctk.CTkLabel(item_frame, text=pred_text, font=ctk.CTkFont(weight="bold"),
                                          anchor="w", text_color=item_text_color)
                pred_label.grid(row=0, column=1, padx=5, pady=(5, 0), sticky="new")

                time_text = entry['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                time_label = ctk.CTkLabel(item_frame, text=time_text, font=ctk.CTkFont(size=10),
                                          text_color="gray", anchor="w")
                time_label.grid(row=1, column=1, padx=5, pady=(0, 5), sticky="sew")

                click_lambda = lambda event, data=entry: self._load_from_history(data)
                item_frame.bind("<Button-1>", click_lambda)
                thumb_label.bind("<Button-1>", click_lambda)
                pred_label.bind("<Button-1>", click_lambda)
                time_label.bind("<Button-1>", click_lambda)

                context_menu_lambda = lambda event, data=entry: self._show_history_context_menu(event, data)
                item_frame.bind("<Button-3>", context_menu_lambda)
                item_frame.bind("<Button-2>", context_menu_lambda)

                item_frame.configure(cursor="hand2")
                thumb_label.configure(cursor="hand2")
                pred_label.configure(cursor="hand2")
                time_label.configure(cursor="hand2")

        self._set_buttons_state("normal")
    def _load_from_history(self, history_entry: Dict[str, Any]) -> None:
        if self.webcam_active: return

        image_path_to_load = history_entry['image_path']
        print(f"Geçmişten yükleniyor: {image_path_to_load}")

        if not os.path.exists(image_path_to_load):
            self._update_status(f"Geçmişteki dosya bulunamadı - {os.path.basename(image_path_to_load)}", "error")
            tkinter.messagebox.showwarning("Geçmiş Hatası", f"Seçilen geçmiş öğesine ait resim dosyası bulunamadı:\n{image_path_to_load}", parent=self)

            return

        try:
            self.current_image_path = image_path_to_load

            self.current_predictions = history_entry['all_predictions']
            self._update_status(f"Geçmişten resim yükleniyor: {os.path.basename(self.current_image_path)}", "processing")

            img = Image.open(self.current_image_path)
            img_copy = img.copy()
            img_copy.thumbnail(IMAGE_DISPLAY_SIZE, Image.Resampling.LANCZOS)
            self.current_ctk_image = ctk.CTkImage(light_image=img_copy, dark_image=img_copy, size=img_copy.size)

            self.image_label.configure(image=self.current_ctk_image, text="")
            self.filename_label.configure(text=f"Yüklenen Dosya: {os.path.basename(self.current_image_path)}")


            self.display_results(self.current_predictions)
            self._update_status("Geçmişten tahmin başarıyla yüklendi.", "success")
            self._set_buttons_state("normal")

        except Exception as e:
            print(f"Geçmişten resim yükleme hatası: {e}")
            self._update_status(f"Geçmişten resim yüklenemedi ({type(e).__name__}).", "error")
            self._recreate_image_label(config={
                "text": "❌ Geçmişteki resim yüklenemedi.",
                "font": self.image_label_initial_config["font"], "text_color": LOW_CONFIDENCE_TEXT_COLOR
            })
            self.filename_label.configure(text="Yüklenen Dosya: -")
            self.current_image_path = None; self.current_ctk_image = None; self.current_predictions = None
            self.clear_results()
            self._set_buttons_state("normal")

    def _show_history_context_menu(self, event: tkinter.Event, history_entry: Dict[str, Any]) -> None:
        if self.webcam_active: return

        context_menu = tkinter.Menu(self, tearoff=0, font=("Segoe UI", 9))
        context_menu.add_command(
            label="🗑️ Bu Öğeyi Sil",
            command=lambda entry=history_entry: self._delete_history_item(entry)
        )

        context_menu.add_command(
             label="📂 Dosya Konumunu Aç",
             command=lambda path=history_entry['image_path']: self._open_file_location(path)
         )
        context_menu.add_separator()
        context_menu.add_command(label="İptal")

        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()

    def _open_file_location(self, file_path: str):
        try:
            if not os.path.exists(file_path):
                 tkinter.messagebox.showwarning("Dosya Bulunamadı", "Dosya bulunamadı.", parent=self)
                 return
            folder_path = os.path.dirname(file_path)
            print(f"Klasör açılıyor: {folder_path}")
            if sys.platform == "win32":
                os.startfile(folder_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder_path])
            else:
                subprocess.Popen(["xdg-open", folder_path])
        except Exception as e:
            print(f"Dosya konumu açılamadı: {e}")
            tkinter.messagebox.showerror("Hata", f"Dosya konumu açılamadı:\n{e}", parent=self)
            self._update_status("Hata: Dosya konumu açılamadı.", "error")


    def _delete_history_item(self, entry_to_delete: Dict[str, Any]) -> None:
        if self.webcam_active: return

        confirm = tkinter.messagebox.askyesno(
            "Geçmiş Öğesini Sil",
            f"Şu tahmini geçmişten silmek istediğinizden emin misiniz?\n\n"
            f"Tür: {entry_to_delete['top_prediction']}\n"
            f"Tarih: {entry_to_delete['timestamp'].strftime('%Y-%m-%d %H:%M')}",
            icon='warning', parent=self
        )

        if confirm:
            try:

                if self.current_image_path == entry_to_delete['image_path']:
                    self.clear_results()
                    self._recreate_image_label()
                    self.filename_label.configure(text="Yüklenen Dosya: -")
                    self.current_image_path = None
                    self.current_ctk_image = None
                    self.current_predictions = None

                self.history_data.remove(entry_to_delete)
                print(f"Geçmiş öğesi silindi: {entry_to_delete['image_path']}")
                self._update_history_display()
                self._update_status("Geçmiş öğesi silindi.", "info")
            except ValueError:
                print("Silinecek geçmiş öğesi listede bulunamadı.")
                self._update_status("Geçmiş öğesi silinemedi.", "error")
            except Exception as e:
                print(f"Geçmiş öğesi silinirken hata: {e}")
                self._update_status(f"Geçmiş öğesi silinirken sorun oluştu: {e}", "error")
        else:
            self._update_status("Öğe silme işlemi iptal edildi.", "info")

    def _clear_all_history(self) -> None:
        if self.webcam_active: return
        if not self.history_data:
            self._update_status("Geçmiş zaten boş.", "info")
            return

        confirm = tkinter.messagebox.askyesno(
            "Tüm Geçmişi Temizle",
            "Tüm tahmin geçmişini silmek istediğinizden emin misiniz?\nBu işlem geri alınamaz!",
            icon='warning', default='no', parent=self
        )

        if confirm:

            if self.current_image_path and any(entry['image_path'] == self.current_image_path for entry in self.history_data):
                 self.clear_results()
                 self._recreate_image_label()
                 self.filename_label.configure(text="Yüklenen Dosya: -")
                 self.current_image_path = None
                 self.current_ctk_image = None
                 self.current_predictions = None

            self.history_data.clear()
            self._update_history_display()
            self._update_status("Tüm tahmin geçmişi temizlendi.", "info")
            print("Tüm tahmin geçmişi temizlendi.")
        else:
            self._update_status("Geçmiş temizleme işlemi iptal edildi.", "info")



    def toggle_webcam_callback(self):
        if not self.model:
            self._update_status("❌ Model yüklenemediği için webcam başlatılamıyor.", "error")
            tkinter.messagebox.showerror("Model Hatası", "Webcam'i başlatmak için önce modelin başarıyla yüklenmesi gerek.", parent=self)
            return

        if self.webcam_active:
            self.stop_webcam()
        else:
            self.start_webcam()


    def start_webcam(self):
        try:
            print("Webcam başlatılıyor...")
            self.webcam_capture = cv2.VideoCapture(0)
            if not self.webcam_capture.isOpened():
                     raise IOError("Webcam açılamadı (index 0). Başka bir uygulama kullanıyor olabilir.")

            self.webcam_active = True
            self._update_status("📷 Webcam başlatıldı...", "info")
            if self.webcam_button.winfo_exists(): self.webcam_button.configure(text="⏹️ Webcam Durdur")
            self.clear_results()
            self.current_image_path = None
            self.current_predictions = None
            if self.filename_label.winfo_exists(): self.filename_label.configure(text="Yüklenen Dosya: Webcam")

            self._recreate_image_label(config={"text": "Webcam Başlatılıyor...", "font": ctk.CTkFont(size=16), "text_color": "gray50"})
            self._set_buttons_state("disabled")

            self.webcam_thread = threading.Thread(target=self._webcam_loop, daemon=True)
            self.webcam_thread.start()
            self._process_webcam_queue()
            print("Webcam thread'i başlatıldı.")

        except Exception as e:
            print(f"Webcam başlatma hatası: {e}")
            self._update_status(f"❌ Webcam başlatma hatası: {e}", "error")
            tkinter.messagebox.showerror("Webcam Hatası", f"Webcam başlatılamadı:\n{e}", parent=self)
            if self.webcam_capture and self.webcam_capture.isOpened():
                self.webcam_capture.release()
            self.webcam_capture = None
            self.webcam_active = False
            if hasattr(self, 'image_label'):
                 self._recreate_image_label()
            if hasattr(self, 'webcam_button'): self.webcam_button.configure(text="📷 Webcam Başlat")
            self._set_buttons_state("normal")
    def stop_webcam(self):
        if self.webcam_active:
            print("Webcam durduruluyor...")
            self.webcam_active = False
            self._update_status("📷 Webcam durduruluyor...", "info")

            if self.webcam_thread and self.webcam_thread.is_alive():
                print("Webcam thread'inin bitmesi bekleniyor...")
                self.webcam_thread.join(timeout=1.5)
                if self.webcam_thread.is_alive(): print("Webcam thread'i zamanında durmadı.")
                self.webcam_thread = None

            while not self.webcam_queue.empty():
                 try: self.webcam_queue.get_nowait()
                 except queue.Empty: break
                 self.webcam_queue.task_done()

            if self.webcam_capture and self.webcam_capture.isOpened():
                self.webcam_capture.release();
                print("Webcam kaynağı serbest bırakıldı.")
            else: print("Webcam kaynağı zaten kapalıydı veya hiç açılmamış.")
            self.webcam_capture = None

            if self.winfo_exists():
                if hasattr(self, 'webcam_button') and self.webcam_button.winfo_exists():
                    self.webcam_button.configure(text="📷 Webcam Başlat")

                if hasattr(self, 'image_label') and self.image_label.winfo_exists():
                    self._recreate_image_label()

                if hasattr(self, 'filename_label') and self.filename_label.winfo_exists():
                    self.filename_label.configure(text="Yüklenen Dosya: -")

                self.clear_results()
                self._set_buttons_state("normal")
                self._update_status("✅ Webcam durduruldu.", "success")

            print("Webcam durdurma işlemi tamamlandı.")
        else:
            print("Webcam zaten aktif değildi.")
    def _webcam_loop(self):
        last_prediction_time = time.monotonic()
        frame_count = 0
        fps_start_time = time.monotonic()
        reported_fps = 0.0
        last_predictions = None

        while self.webcam_active:
            capture = self.webcam_capture
            if not capture or not capture.isOpened():
                print("Webcam döngüsü içinde capture kapalı!")
                if self.webcam_active:
                    try: self.webcam_queue.put(("error", "Webcam bağlantısı koptu."), block=False)
                    except queue.Full: print("Webcam Kuyruğu Dolu")
                break
            ret, frame = capture.read()
            if not ret:
                print("Webcam'den kare okunamadı. Döngü sonlandırılıyor.")
                if self.webcam_active:
                    try: self.webcam_queue.put(("error", "Webcam'den kare okunamadı."), block=False)
                    except queue.Full: print("Webcam Kuyruğu Dolu")
                break

            current_time = time.monotonic()
            perform_prediction = (current_time - last_prediction_time >= WEBCAM_PREDICTION_INTERVAL)

            try:

                if perform_prediction:
                    last_prediction_time = current_time
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    img_t = self.infer_transform(pil_img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = self.model(img_t)
                        probabilities = torch.softmax(outputs, dim=1)
                        top_probs, top_indices = probabilities.topk(self.top_k, dim=1)

                    top_probs_np = top_probs.cpu().numpy().flatten()
                    top_indices_np = top_indices.cpu().numpy().flatten()

                    predictions = []
                    for idx, prob in zip(top_indices_np, top_probs_np):
                        if 0 <= idx < len(self.class_names):
                            class_name = self.class_names[idx].replace('_', ' ').capitalize()
                            predictions.append({'class_name': class_name, 'probability': float(prob)})
                        else:
                            predictions.append({'class_name': f'Geçersiz Index ({idx})', 'probability': float(prob)})
                    last_predictions = predictions

                else:
                    predictions = last_predictions
                display_frame = frame
                if predictions:
                    top_pred_text = f"{predictions[0]['class_name']} ({predictions[0]['probability']:.1%})"

                    is_low_conf = predictions[0]['probability'] < self.low_confidence_threshold
                    text_color = (0, 128, 255) if is_low_conf else (0, 255, 0)
                    bg_color = (0, 0, 0)
                    (text_width, text_height), baseline = cv2.getTextSize(top_pred_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(display_frame, (5, 5), (10 + text_width, 10 + text_height + baseline), bg_color, thickness=cv2.FILLED)
                    cv2.putText(display_frame, top_pred_text, (8, 8 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

                frame_count += 1
                fps_duration = current_time - fps_start_time
                if fps_duration >= 1.0:
                    reported_fps = frame_count / fps_duration
                    frame_count = 0; fps_start_time = current_time
                cv2.putText(display_frame, f"FPS: {reported_fps:.1f}", (display_frame.shape[1] - 90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if perform_prediction and self.webcam_active:
                    frame_rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    try:
                         self.webcam_queue.put(("update", frame_rgb_display, predictions), block=False, timeout=0.1)
                    except queue.Full:
                         pass

            except Exception as e:
                print(f"Webcam döngü hatası: {e}")
                import traceback; traceback.print_exc()
                if self.webcam_active:
                    try: self.webcam_queue.put(("error", f"İşleme hatası: {type(e).__name__}"), block=False)
                    except queue.Full: print("Webcam Kuyruğu Dolu")
                time.sleep(0.5)
        print("Webcam döngüsü sona erdi.")
    def _process_webcam_queue(self):
        try:
            while not self.webcam_queue.empty():
                msg = self.webcam_queue.get_nowait()
                msg_type = msg[0]

                if msg_type == "update":

                    if self.webcam_active and self.winfo_exists():
                        _, frame_rgb, predictions = msg
                        try:
                            pil_img = Image.fromarray(frame_rgb)


                            pil_img.thumbnail(IMAGE_DISPLAY_SIZE, Image.Resampling.LANCZOS)
                            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)

                            if hasattr(self, 'image_label') and self.image_label.winfo_exists():
                                self.image_label.configure(image=ctk_img, text="")

                            self.display_results(predictions)

                        except tkinter.TclError as tk_error:
                            print(f"Webcam güncellemesi sırasında tkinter.TclError : {tk_error}")
                        except Exception as e:
                            print(f"Webcam GUI güncelleme hatası: {e}")
                            import traceback; traceback.print_exc()

                elif msg_type == "error":
                    _, error_message = msg
                    print(f"Webcam thread'inden hata mesajı alındı: {error_message}")
                    if self.webcam_active and self.winfo_exists():
                        self._update_status(f"❌ Webcam Hatası: {error_message}. Webcam durduruluyor...", "error")
                        print("Hata nedeniyle webcam durduruluyor...")
                        self.stop_webcam()
                self.webcam_queue.task_done()

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Webcam kuyruk işleme hatası: {e}")
            import traceback; traceback.print_exc()

        finally:
            if self.webcam_active and self.winfo_exists():
                self.after(35, self._process_webcam_queue)
    def open_settings_window(self):
        if not self.model:
             self._update_status("Ayarları açmak için önce modelin yüklenmesi gerekir.", "warning")
             return

        if self.settings_window is not None and self.settings_window.winfo_exists():
            print("Ayarlar penceresi zaten açık, öne getiriliyor.")
            self.settings_window.lift()
            self.settings_window.focus_set()
            return
        print("Ayarlar penceresi açılıyor...")
        current_settings = {
            'max_history': self.max_history_items,
            'top_k': self.top_k,
            'low_conf_threshold': self.low_confidence_threshold
        }

        self.settings_window = SettingsWindow(self, current_settings, self.update_settings_callback)
    def update_settings_callback(self, new_settings: Dict[str, Any]):
        try:
            old_max_history = self.max_history_items
            old_low_conf_threshold = self.low_confidence_threshold
            old_top_k = self.top_k

            self.max_history_items = int(new_settings['max_history'])
            self.top_k = int(new_settings['top_k'])
            self.low_confidence_threshold = float(new_settings['low_conf_threshold'])

            print("Ayarlar başarıyla güncellendi:")
            print(f" - Maksimum Geçmiş: {self.max_history_items}")
            print(f" - Top K: {self.top_k}")
            print(f" - Güven Eşiği: {self.low_confidence_threshold:.2%}")

            self._update_status("Ayarlar başarıyla güncellendi.", "success")

            if self.max_history_items < old_max_history:
                while len(self.history_data) > self.max_history_items:
                    self.history_data.pop()
                self._update_history_display()

            if self.current_predictions and (self.low_confidence_threshold != old_low_conf_threshold or self.top_k != old_top_k):
                 print("Ayarlar değişti, mevcut sonuçlar yeniden gösteriliyor...")
                 self.display_results(self.current_predictions)

            elif self.history_data and self.low_confidence_threshold != old_low_conf_threshold:
                 print("Güven eşiği değişti, geçmiş listesi güncelleniyor...")
                 self._update_history_display()

        except Exception as e:
            print(f"Ayarlar güncellenirken hata: {e}")
            import traceback
            traceback.print_exc()
            tkinter.messagebox.showerror("Hata", f"Ayarlar güncellenirken bir sorun oluştu:\n{e}", parent=self)
        finally:
            self.settings_window = None
            print("Ayarlar penceresi temizlendi.")

if __name__ == "__main__":
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"Model dosyası bulunamadı: {MODEL_CHECKPOINT_PATH}")
        root_err = tkinter.Tk(); root_err.withdraw()
        tkinter.messagebox.showerror("Hata", f"Model dosyası bulunamadı\n{MODEL_CHECKPOINT_PATH}\n\nLütfen `MODEL_CHECKPOINT_PATH` değişkenini kod içinde doğru ayarlayın.")
        root_err.destroy(); exit()
    else: print(f"Model dosyası bulundu: {MODEL_CHECKPOINT_PATH}")

    print("Model yükleme işlemi başlıyor...")
    loaded_model = load_model(MODEL_NAME, MODEL_CHECKPOINT_PATH, NUM_CLASSES, device)
    if loaded_model is None:
        print("Model yüklenemediği için uygulama başlatılamıyor.")
        root_err = tkinter.Tk(); root_err.withdraw()
        tkinter.messagebox.showerror("Model Yükleme Hatası", "Model yüklenirken sorun oluştu. Lütfen kontrol edin.", parent=None)
        root_err.destroy(); exit()
    else: print("Model başarıyla yüklendi.")

    import subprocess
    print("Arayüz başlatılıyor...")
    app = AnimalClassifierApp(model=loaded_model, class_names=class_names, device=device)
    app.mainloop()
    print("Uygulama kapatıldı.")
    if 'app' in locals() and hasattr(app, 'webcam_active') and app.webcam_active:
         print("Kapanışta webcam hala aktif görünüyor, durduruluyor...")
         app.stop_webcam()
    elif 'app' in locals() and hasattr(app, 'webcam_capture') and app.webcam_capture and app.webcam_capture.isOpened():
         print("Kapanışta webcam kaynağı açık kalmış, serbest bırakılıyor...")
         app.webcam_capture.release()
    print("Program sonlandı.")