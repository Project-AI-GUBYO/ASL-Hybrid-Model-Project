import cv2
import mediapipe as mp
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import StringVar
import threading
import time

# Cihaz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sınıf isimleri
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
               "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
               "Y", "Z", "del", "nothing", "space"]

# Görüntü dönüşümü
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Modeli yükle
def load_model():
    checkpoint = torch.load("asl_hybrid_model.pth", map_location=device)

    def build_model(model_type, state_dict):
        if model_type == "googlenet":
            model = models.googlenet(weights=None, aux_logits=False)
            model.fc = nn.Linear(1024, len(class_names))
        elif model_type == "resnet18":
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(512, len(class_names))
        elif model_type == "densenet121":
            model = models.densenet121(pretrained=False)
            model.classifier = nn.Linear(1024, len(class_names))
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        return model

    googlenet = build_model("googlenet", checkpoint["googlenet"])
    resnet18 = build_model("resnet18", checkpoint["resnet18"])
    densenet121 = build_model("densenet121", checkpoint["densenet121"])
    return googlenet, resnet18, densenet121

# Tahmin fonksiyonu
def predict(image_pil, models):
    image = transform(image_pil).unsqueeze(0).to(device)
    outputs = [model(image) for model in models]
    avg_output = torch.mean(torch.stack(outputs), dim=0)
    probabilities = torch.nn.functional.softmax(avg_output, dim=1)[0]
    max_prob, pred = torch.max(probabilities, 0)
    predicted_label = class_names[pred.item()]
    confidence = max_prob.item() * 100
    print(f"Tahmin: {predicted_label} ({confidence:.2f}%)")
    return predicted_label

# GUI sınıfı
class ASLApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ASL Tahmin")
        self.text_var = StringVar()
        self.text_var.set("")

        label = tk.Label(self.root, textvariable=self.text_var, font=("Helvetica", 24))
        label.pack(padx=20, pady=20)

        self.root.bind("<Delete>", self.clear_text)  # Burada Delete dinleniyor

        self.models_loaded = load_model()
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.daemon = True
        self.thread.start()

        self.root.mainloop()

    def clear_text(self, event=None):
        self.text_var.set("")
        print("Metin temizlendi.")

    def update_gui_text(self, char):
        current = self.text_var.get()
        if char == "space":
            current += " "
        elif char == "del":
            current = current[:-1]
        elif char != "nothing":
            current += char
        self.text_var.set(current)

    def video_loop(self):
        cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        mp_draw = mp.solutions.drawing_utils

        last_gui_update_time = 0
        last_label = None
        repeat_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            current_label = None

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0

                    for lm in handLms.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)

                    box_size = max(x_max - x_min, y_max - y_min)
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    half_size = box_size // 2

                    x_min_square = max(0, cx - half_size - 20)
                    y_min_square = max(0, cy - half_size - 20)
                    x_max_square = min(w, cx + half_size + 20)
                    y_max_square = min(h, cy + half_size + 20)

                    roi = frame[y_min_square:y_max_square, x_min_square:x_max_square]
                    if roi.size != 0:
                        image_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                        label = predict(image_pil, self.models_loaded)
                        current_label = label

                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS,
                                           mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                                           mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2))

            now = time.time()
            if current_label:
                if current_label == last_label:
                    repeat_count += 1
                else:
                    repeat_count = 1
                    last_label = current_label

                if repeat_count >= 5 and (now - last_gui_update_time >= 1.5):
                    self.update_gui_text(current_label)
                    print(f"✓ Onaylı Tahmin: {current_label}")
                    last_gui_update_time = now
                    repeat_count = 0
                    last_label = None

            cv2.imshow("ASL Kamera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Uygulamayı başlat
ASLApp()