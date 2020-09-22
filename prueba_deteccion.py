import tkinter as tk
from sys import exit


def show_notification_message(message):
    popupRoot = tk.Tk()
    popupRoot.configure(bg='gray')
    image = "/Users/rodrigomoralesrivas/PycharmProjects/proyecto_tesis/yolov4-deepsort/data/danger.gif"
    photo = tk.PhotoImage(file=image)
    label = tk.Label(image=photo)
    label.pack()
    popupRoot.after(5000, exit)
    popupRoot.title("EMERGENCIA")
    popupButton = tk.Button(popupRoot, text=message, font=("Verdana", 12), bg="yellow", command=exit, wraplength=390)
    popupButton.pack()
    popupRoot.geometry('400x200+700+500')
    popupRoot.mainloop()
show_notification_message("Se supero el tiempo de espera maximo y se recomienda abrira una nueva caja")
