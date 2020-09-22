import tkinter as tk
from sys import exit


def show_notification_message(message):
    popupRoot = tk.Tk()
    popupRoot.after(2000, exit)
    popupButton = tk.Button(popupRoot, text=message, font=("Verdana", 12), bg="yellow", command=exit)
    popupButton.pack()
    popupRoot.geometry('400x50+700+500')
    popupRoot.mainloop()

show_notification_message("HOLA!")
