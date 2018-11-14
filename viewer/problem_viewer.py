# In order to be able to import tkinter for either in python 2/3
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk

from PIL import ImageTk, Image

root = tk.Tk()
img = ImageTk.PhotoImage(Image.open("background.jpg"))
panel = tk.Label(root, image = img)
panel.pack()
root.mainloop()

