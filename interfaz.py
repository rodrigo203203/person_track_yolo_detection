import tkinter as tk
import os
from PIL import ImageTk, Image
root = tk.Tk()
video = ""
salida = ""
root.title("VisionRex")
root.geometry("650x400")
#"#32afb5"
root.configure(bg="white")
imagen = Image.open("./data/logo-visionrex.png")
imagen = imagen.resize((200,100))
#imagen.save("ProyectosconPython_resize.png")
img = ImageTk.PhotoImage(imagen)
label1 = tk.Label(root, image=img)
label1.place(x=240, y=0)
def start():
    global video
    global salida
    def seleccionar_video():
        global video
        a = serialEntry.get()
        video = a
        print(video)
    def guardar_video():
        global salida
        a = serialEntry2.get()
        salida = a
        print(salida)

    def iniciar():
        virtual_session = os.system("python object_tracker_modificado.py --weights ./checkpoints/yolov4-custom-best-416 --model yolov4 --video ./data/video/"+video+" --output ./data/video/output/"+salida+".avi --info")
        print("activate")
    texto=tk.Label(root,text="Introduce la direccion y el video para analizar:",bg ="white")
    texto.place(x=180,y=170)
    serialEntry=tk.Entry(root)
    serialEntry.place(x=170,y=210)
    b1 = tk.Button(text='Guardar', command=seleccionar_video)
    b1.place(x=390,y=210)
    texto2=tk.Label(root,text="Introduce el nombre del video resultante:",bg ="white")
    texto2.place(x=190,y=250)
    serialEntry2 = tk.Entry(root)
    serialEntry2.place(x=170,y=290)
    b2 = tk.Button(text='Guardar', command=guardar_video)
    b2.place(x=390,y=290)
    b3 = tk.Button(text='Iniciar', command=iniciar,bg ="blue")
    b3.place(x=300,y=330)
    root.mainloop()
if __name__ == '__main__':
    start()