import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps, ImageTk

from model import load_model, predict_top5

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ARCH = "resnet18"
WEIGHTS = BASE_DIR / "models" / "t_model.pth"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pet Breed Classifier")
        self.model, self.device = load_model(ARCH, WEIGHTS)

        tk.Button(self, text="Upload Image", command=self.upload).pack(pady=8)
        self.image_label = tk.Label(self)
        self.image_label.pack()
        self.result_var = tk.StringVar(value="No image loaded")
        tk.Label(self, textvariable=self.result_var, justify="left",
                 font=("Courier", 11)).pack(padx=12, pady=10)

    def upload(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return
        try:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{e}")
            return

        preview = img.copy()
        preview.thumbnail((320, 320))
        self.photo = ImageTk.PhotoImage(preview)  # keep a reference
        self.image_label.configure(image=self.photo)

        results = predict_top5(self.model, img, ARCH, self.device)
        self.result_var.set("\n".join(
            f"{i}. {name:<28}{p * 100:5.1f}%" for i, (name, p) in enumerate(results, 1)
        ))


if __name__ == "__main__":
    App().mainloop()