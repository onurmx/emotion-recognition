import tkinter as tk
from gui import gui

if __name__ == "__main__":
    root = tk.Tk()
    app = gui.EmotionRecognitionApplication(root)
    app.mainloop()