import tkinter as tk
from sys import exit


def show_notification_message(message):
    popupRoot = tk.Tk()
    popupRoot.configure(bg='gray')
    image = "/Users/rodrigomoralesrivas/PycharmProjects/proyecto_tesis/yolov4-deepsort/data/time-is-up.png"
    photo = tk.PhotoImage(file=image)
    label = tk.Label(image=photo)
    label.pack()
    popupRoot.after(5000, exit)
    popupRoot.title("EMERGENCIA")
    popupButton = tk.Button(popupRoot, text=message, font=("Verdana", 25), bg="gray", command=exit, wraplength=390)
    popupButton.pack()
    popupRoot.geometry('555x300+700+500')
    popupRoot.mainloop()
show_notification_message("Se súpero el tiempo de espera maximo y se recomienda abrir una nueva caja")
