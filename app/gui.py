import tkinter as tk

class EmotionRecognitionApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # center the window
        width = 900
        height = 750
        x = (self.parent.winfo_screenwidth() // 2) - (width // 2)
        y = (self.parent.winfo_screenheight() // 2) - (height // 2)
        self.parent.geometry('{}x{}+{}+{}'.format(width, height, x, y))

        # set the title
        self.parent.title("Emotion Recognition")
