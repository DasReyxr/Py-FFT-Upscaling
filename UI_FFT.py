"""
------ Iker Garcia  ------
--------- Auf Das ---------
--------- FFT II ----------
-------- 08/11/2025 -------
"""
# ------- Main Library -------
import customtkinter as ctk
from PIL import Image

from tkinter import filedialog, messagebox
import threading
import os
from ProyectoFFT import generate_fft_report_safe

PINK_PATH_PHOTO = r"C:\Users\dasre\Wkn\Python\PY1 Upscaling\HK.jpg"
PINK_PATH_THEME = r"C:\Users\dasre\Wkn\Python\PY1 Upscaling\pink.json"

# ---------- Class ----------
class FFTGui(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window setup ---
        self.title("FFT Resizer GUI")
        self.geometry("520x560")
        ctk.set_appearance_mode("dark")  # initial mode
        
        ctk.ThemeManager.load_theme(PINK_PATH_THEME)
        # --- Track appearance mode ---
        self.current_mode = "dark"
        self.configure(fg_color="#272727") 


        # --- Input folder ---
        self.label_input = ctk.CTkLabel(self, text="Carpeta de entrada:",fg_color="transparent")
        self.label_input.pack(pady=(15, 5))

        self.input_entry = ctk.CTkEntry(self, width=400, placeholder_text="Selecciona una carpeta...")
        self.input_entry.pack(pady=(0, 5))

        self.btn_input = ctk.CTkButton(self, text="Seleccionar carpeta", command=self.select_input)
        self.btn_input.pack(pady=5)

        # --- Output file ---
        self.label_output = ctk.CTkLabel(self, text="PDF de salida:",fg_color="transparent")
        self.label_output.pack(pady=(15, 5))

        self.output_entry = ctk.CTkEntry(self, width=400, placeholder_text="Selecciona destino del PDF...")
        self.output_entry.pack(pady=(0, 5))

        self.btn_output = ctk.CTkButton(self, text="Seleccionar destino", command=self.select_output)
        self.btn_output.pack(pady=5)

        # --- Scales ---
        self.label_scales = ctk.CTkLabel(self, text="Escalas:",fg_color="transparent")
        self.label_scales.pack(pady=(15, 5))

        self.scale_entry = ctk.CTkEntry(self, width=200, placeholder_text="Ej. 0.1")
        self.scale_entry.pack(pady=(0, 5))

        self.btn_add_scale = ctk.CTkButton(self, text="Agregar escala", command=self.add_scale)
        self.btn_add_scale.pack(pady=(0, 5))

        self.scales_list_label = ctk.CTkLabel(self, text="Escalas agregadas: []",fg_color="transparent")
        self.scales_list_label.pack(pady=(0, 10))

        # Internal list to store scales
        self.scales = []


        # --- Run button ---
        self.run_button = ctk.CTkButton(self, text="Generar PDF", command=self.run)
        self.run_button.pack(pady=(20, 10))

        # --- Progress label ---
        self.progress = ctk.CTkLabel(self, text="Listo",fg_color="transparent")
        self.progress.pack(pady=10)

        # --- Dark/Light Mode Switch ---
        self.mode_switch = ctk.CTkSwitch(
            self,
            text="Modo oscuro",
            command=self.toggle_mode,
            onvalue="dark",
            offvalue="light",
        )
        self.mode_switch.select()  # start in dark mode
        self.mode_switch.pack(pady=(10, 15))

        # --- Hello Kitty background image (hidden by default) ---
        image_path = os.path.join(os.path.dirname(__file__), PINK_PATH_PHOTO) 
        if os.path.exists(image_path):
            self.bg_image = ctk.CTkImage(light_image=Image.open(image_path),
                                        dark_image=Image.open(image_path),
                                        size=(520, 560))
            self.bg_label = ctk.CTkLabel(self, image=self.bg_image, text="")
            # Don’t place it yet — we’ll only show it in Modo Rosa
        else:
            self.bg_label = None
            print("⚠️ Hello Kitty image not found — background disabled.")


    # ===================== Logic =====================
    def select_input(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, folder)

    def select_output(self):
        file = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")]
        )
        if file:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, file)

    def add_scale(self):
        value = self.scale_entry.get().strip()
        try:
            val = float(value)
            if val <= 0:
                raise ValueError
            self.scales.append(val)
            self.scales_list_label.configure(text=f"Escalas agregadas: {self.scales}")
            self.scale_entry.delete(0, "end")
        except ValueError:
            messagebox.showerror("Error", "Introduce un número válido mayor que 0")

    def run(self):
        inp = self.input_entry.get().strip()
        outp = self.output_entry.get().strip()
        if not self.scales:
            messagebox.showerror("Error", "Agrega al menos una escala")
            return

        scales = self.scales

        if not os.path.isdir(inp):
            messagebox.showerror("Error", "Carpeta de entrada inválida")
            return
        if not outp:
            messagebox.showerror("Error", "Archivo de salida inválido")
            return

        threading.Thread(target=self.process, args=(inp, outp, scales), daemon=True).start()
        self.progress.configure(text="Procesando...", text_color="orange")

    def process(self, inp, outp, scales):
        try:
            generate_fft_report_safe(inp, outp, scales)
            self.progress.configure(text="Listo!")
            messagebox.showinfo("Éxito", "PDF generado correctamente")
        except Exception as e:
            self.progress.configure(text="Error", text_color="red")
            messagebox.showerror("Error", str(e))   
                
    def toggle_mode(self):
        """Switch between dark and pink (light) modes."""
        new_mode = self.mode_switch.get()
        ctk.set_appearance_mode(new_mode)
        self.current_mode = new_mode

        if new_mode == "light":  # Modo Rosa
            if self.bg_label:
                self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # Show image
                self.bg_label.lower()  # Send image behind widgets
            self.configure(fg_color="white")  # optional light background under image
            self.mode_switch.configure(text="Modo Puto")
        else:  # Modo Darks
            if self.bg_label:
                self.bg_label.place_forget()  # Hide image
            self.configure(fg_color="#272727")
            self.mode_switch.configure(text="Modo Darks")



if __name__ == "__main__":
    app = FFTGui()
    app.mainloop()
