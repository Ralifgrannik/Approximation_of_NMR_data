# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Отключаем лишние потоки PyTorch!
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Путь к модели
CHECKPOINT_PATH = resource_path("best_nmr_param_cnn.pth")

N_POINTS = 2048
T_MAX = 12.282
T_ECHOES = torch.linspace(0, T_MAX, N_POINTS)

class NMRParamCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=16, stride=1, padding='same'), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=16, stride=1, padding='same'), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding='same'), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding='same'), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding='same'), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 64, 4096), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(4096, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 8)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        T2_out = torch.relu(x[:, :4])
        w_out = F.softmax(x[:, 4:], dim=1)
        return torch.cat([T2_out, w_out], dim=1)

def reconstruct_signal(t2_values, w_values, time_points):
    reconstructed = torch.zeros_like(time_points)
    for t2, w in zip(t2_values, w_values):
        if w > 0.001 and t2 > 1e-6:
            reconstructed += w * torch.exp(-time_points / t2)
    return reconstructed.numpy()

class NMRAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("NMR T₂ Анализатор")
        master.geometry("1280x880")
        master.minsize(1150, 800)

        master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        if not self.model:
            messagebox.showerror("Критическая ошибка", "Модель не загружена. Приложение будет закрыто.")
            self.on_closing()  
            return

        self.create_widgets()

    def _load_model(self):
        if not os.path.exists(CHECKPOINT_PATH):
            messagebox.showerror("Ошибка", f"Модель не найдена!\nОжидался файл: best_nmr_param_cnn.pth")
            return None

        model = NMRParamCNN().to(self.device)
        try:
            state_dict = torch.load(CHECKPOINT_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", f"Не удалось загрузить модель:\n{e}")
            return None

    def create_widgets(self):
        top = ttk.Frame(self.master, padding="5 5")
        top.pack(fill='x')

        ttk.Button(top, text="Загрузить файл CPMG (.txt/.nmr)", command=self.load_and_analyze).pack(side='left', padx=(0, 5))
        self.file_label = ttk.Label(top, text="Файл не выбран", font=('Arial', 11, 'italic'), foreground="#7f8c8d")
        self.file_label.pack(side='left', padx=(5, 0))
        ttk.Label(top, text=f"Устройство: {self.device}", font=('Arial', 10), foreground="#2c3e50").pack(side='right')

        graphs_frame = ttk.Frame(self.master)
        graphs_frame.pack(fill='both', expand=True, padx=10, pady=(5, 5))

        self.fig1, self.ax1 = plt.subplots(figsize=(6.5, 5))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, graphs_frame)
        self.canvas1.get_tk_widget().pack(side='left', expand=True, fill='both', padx=(0, 5))

        self.fig2, self.ax2 = plt.subplots(figsize=(6.5, 5))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, graphs_frame)
        self.canvas2.get_tk_widget().pack(side='right', expand=True, fill='both', padx=(5, 0))

        results_frame = ttk.LabelFrame(self.master, text=" Предсказанные T₂-компоненты ", padding="10")
        results_frame.pack(fill='x', padx=15, pady=(0, 15))

        ttk.Label(results_frame, text="Распределение времён T₂-релаксации", font=('Arial', 14, 'bold')).pack(pady=(0, 8))

        columns = ("№", "T₂, с", "Доля, %", "Вклад")
        self.tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=6)
        style = ttk.Style()
        style.configure("Treeview", font=('Arial', 14), rowheight=30)
        style.configure("Treeview.Heading", font=('Arial', 13, 'bold'))

        for col in columns:
            self.tree.heading(col, text=col)
            width = 340 if col == "Вклад" else 140 if col != "№" else 60
            anchor = 'w' if col == "Вклад" else 'center'
            self.tree.column(col, width=width, anchor=anchor)
        self.tree.pack(fill='x', padx=5, pady=5)

        self.summary_label = ttk.Label(results_frame, text="Готов к анализу", font=('Arial', 11, 'italic'), foreground="#27ae60")
        self.summary_label.pack(pady=5)

        self.tree.tag_configure("доминирующий", background="#d5f4e6")
        self.tree.tag_configure("значимый", background="#fef9e7")
        self.tree.tag_configure("средний", background="#fdf3e7")
        self.tree.tag_configure("слабый", background="#fadbd8")

    def load_and_analyze(self):
        if not self.model:
            return
        path = filedialog.askopenfilename(filetypes=[("CPMG файлы", "*.txt *.nmr"), ("Все файлы", "*.*")])
        if path:
            self.file_label.config(text=f"Анализ: {os.path.basename(path)}")
            self.analyze_file(path)

    def analyze_file(self, file_path):
        self.ax1.clear()
        self.ax2.clear()
        for i in self.tree.get_children():
            self.tree.delete(i)

        try:
            data = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python')
            time_raw_s = data.iloc[:, 0].values / 1000.0
            amplitude_raw = data.iloc[:, 1].values.astype(float)
            n_raw = len(amplitude_raw)
            used_len = min(n_raw, N_POINTS)

            amp_padded = amplitude_raw[:N_POINTS] if n_raw >= N_POINTS else np.pad(amplitude_raw, (0, N_POINTS - n_raw))
            signal = torch.from_numpy(amp_padded).float()
            signal = (signal - signal.mean()) / (signal.std() + 1e-8)
            x = signal.unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.model(x).cpu().squeeze(0)
            t2_pred = pred[:4].numpy()
            w_pred = pred[4:].numpy()

            components = [(t2, w) for t2, w in zip(t2_pred, w_pred) if w > 0.001]
            components.sort(key=lambda x: x[1], reverse=True)

            for i, (t2, w) in enumerate(components, 1):
                percent = w * 100
                bar = "█" * int(percent // 3) + f" {percent:.2f}%"
                tag = ("доминирующий" if percent > 35 else "значимый" if percent > 15 else "средний" if percent > 5 else "слабый")
                self.tree.insert("", "end", values=(i, f"{t2:.5f}", f"{percent:.2f}", bar), tags=(tag,))

            reconst = reconstruct_signal(torch.tensor(t2_pred), torch.tensor(w_pred), T_ECHOES[:used_len])
            if reconst.max() > 1e-6:
                reconst = reconst * (amplitude_raw.max() / reconst.max())

            time_used = time_raw_s[:used_len]
            amp_used = amplitude_raw[:used_len]
            amp_log = np.clip(amp_used, 1e-6, None)
            reconst_log = np.clip(reconst, 1e-6, None)

            self.ax1.plot(time_used, amp_used, label="Сигнал CPMG", color='blue', linewidth=1.8)
            self.ax1.plot(time_used, reconst, label="Мультиэкспонента", color='red', linestyle='--', linewidth=2.2)
            self.ax1.grid(True, which='major', ls="--", alpha=0.7)
            self.ax1.legend(fontsize=11)
            self.ax1.set_xlabel("Время, с"); self.ax1.set_ylabel("Амплитуда")
            self.ax1.set_title("1. Сигнал CPMG (Линейная шкала Y)")

            self.ax2.plot(time_used, amp_log, label="Сигнал CPMG", color='blue', linewidth=1.8)
            self.ax2.plot(time_used, reconst_log, label="Мультиэкспонента", color='red', linestyle='--', linewidth=2.2)
            self.ax2.set_yscale('log')
            self.ax2.grid(True, which='both', ls="--", alpha=0.6)
            self.ax2.legend(fontsize=11)
            self.ax2.set_xlabel("Время, с"); self.ax2.set_ylabel("Амплитуда (log)")
            self.ax2.set_title("2. Сигнал CPMG (Логарифмическая шкала Y)")

            y_min = max(1e-6, min(amp_log.min(), reconst_log.min()) * 0.5)
            self.ax2.set_ylim(bottom=y_min)

            self.canvas1.draw()
            self.canvas2.draw()
            self.summary_label.config(text=f"Готово • Компонент: {len(components)}", foreground="#27ae60")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обработать файл:\n{e}")
            self.summary_label.config(text="Ошибка", foreground="red")

    def on_closing(self):
        print("Закрытие приложения... Очистка ресурсов...")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        try:
            self.master.destroy()
        except:
            pass
        os._exit(0)  

if __name__ == '__main__':
    root = tk.Tk()
    app = NMRAnalyzerApp(root)

    try:
        root.mainloop()
    finally:
        os._exit(0)