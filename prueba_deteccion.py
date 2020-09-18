import tkinter as tk
from tkinter import filedialog, Text
import os
def reset_all():
    popup.destroy()
popup = tk.Tk()
canvas = tk.Canvas(popup,height=400, width= 600, bg="#263d42")
canvas.pack()
frame = tk.Frame(popup,bg="white")
frame.place(relwidth=0.8,relheight=0.8,relx=0.1,rely=0.1)
popup.title("EMERGENCIA")

b1 = tk.Button(popup, text="Okay", padx=10, pady=5, fg="gray", bg="#263d42", command=reset_all)
b1.pack()

popup.mainloop()