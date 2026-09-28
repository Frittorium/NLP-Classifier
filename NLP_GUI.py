# NLP_GUI.py
import tkinter as tk
from tkinter import ttk, messagebox
from model import SentimentModel

GREEN, RED = "#2e7d32", "#c62828"
SAMPLES = {
    "Sample: positive": "An absolute delight from start to finish. The performances are warm and "
                        "the script is sharp, and I left the cinema smiling. Highly recommended!",
    "Sample: negative": "I really wanted to like this, but it was a slow, incoherent mess. "
                        "The acting was wooden and the ending made no sense. Not worth your time.",
    "Sample: mixed":    "Beautiful cinematography, but the story is predictable and the pacing "
                        "drags. Not terrible, not great either.",
}


class App(tk.Tk):
    def __init__(self, clf: SentimentModel):
        super().__init__()
        self.clf = clf
        self.title("IMDB Review Sentiment Analyzer")
        self.geometry("820x680")
        self.minsize(700, 620)
        pad = {"padx": 14, "pady": 6}

        ttk.Label(self, text="Enter a movie review", font=("Helvetica", 14, "bold")).pack(anchor="w", **pad)

        box = ttk.Frame(self)
        box.pack(fill="both", expand=True, **pad)
        self.txt = tk.Text(box, wrap="word", height=10, font=("Helvetica", 11), undo=True)
        sb = ttk.Scrollbar(box, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        self.txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.txt.bind("<Control-Return>", lambda e: (self.analyze(), "break")[1])

        bar = ttk.Frame(self)
        bar.pack(fill="x", **pad)
        ttk.Button(bar, text="Analyze (Ctrl+Enter)", command=self.analyze).pack(side="left")
        ttk.Button(bar, text="Clear", command=self.clear).pack(side="left", padx=8)
        self.sample = ttk.Combobox(bar, values=list(SAMPLES), state="readonly", width=20)
        self.sample.set("Load a sample...")
        self.sample.pack(side="right")
        self.sample.bind("<<ComboboxSelected>>", self.load_sample)

        res = ttk.LabelFrame(self, text="Result")
        res.pack(fill="x", **pad)
        self.lbl_result = ttk.Label(res, text="-", font=("Helvetica", 22, "bold"))
        self.lbl_result.pack(pady=(8, 2))
        self.bar = ttk.Progressbar(res, maximum=100, length=520)
        self.bar.pack(pady=4)
        self.lbl_prob = ttk.Label(res, text="")
        self.lbl_prob.pack()
        self.lbl_pos = ttk.Label(res, text="", foreground=GREEN, wraplength=740, justify="left")
        self.lbl_pos.pack(anchor="w", padx=10, pady=(8, 0))
        self.lbl_neg = ttk.Label(res, text="", foreground=RED, wraplength=740, justify="left")
        self.lbl_neg.pack(anchor="w", padx=10, pady=(0, 8))

        m = clf.info["test_metrics"]
        ttk.Label(self, foreground="gray",
                  text=f"Model: {clf.info['model']} on {clf.info['features']} | "
                       f"test accuracy {m['accuracy']:.3f}, F1 {m['f1']:.3f}, ROC-AUC {m['roc_auc']:.3f}"
                  ).pack(anchor="w", padx=14, pady=(0, 10))

    def load_sample(self, _=None):
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", SAMPLES[self.sample.get()])
        self.analyze()

    def clear(self):
        self.txt.delete("1.0", "end")
        self.lbl_result.config(text="-", foreground="black")
        self.bar["value"] = 0
        for l in (self.lbl_prob, self.lbl_pos, self.lbl_neg):
            l.config(text="")

    def analyze(self):
        text = self.txt.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("No input", "Type or paste a review first.")
            return
        try:
            r = self.clf.predict(text)
        except ValueError as e:
            messagebox.showwarning("Cannot analyze", str(e))
            return
        color = GREEN if r["label"] == "Positive" else RED
        self.lbl_result.config(text=r["label"], foreground=color)
        self.bar["value"] = r["p_pos"] * 100
        self.lbl_prob.config(text=f"P(positive) = {r['p_pos']:.1%}   |   confidence {r['confidence']:.1%}")
        fmt = lambda ws: ", ".join(f"{w} ({s:+.2f})" for w, s in ws) or "none"
        self.lbl_pos.config(text="Pushes positive: " + fmt(r["pos_words"]))
        self.lbl_neg.config(text="Pushes negative: " + fmt(r["neg_words"]))


def main():
    try:
        clf = SentimentModel()
    except Exception as e:
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Model not found",
                             f"Could not load artifacts. Run 1.py first.\n\n{e}")
        return
    App(clf).mainloop()


if __name__ == "__main__":
    main()
