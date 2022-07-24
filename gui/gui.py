import tkinter as tk

class EmotionRecognitionApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.parent.geometry("500x500")
        self.parent.title("Emotion Recognition Application")