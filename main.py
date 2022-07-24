from gui import gui
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = gui.EmotionRecognitionApplication(root)
    app.mainloop()