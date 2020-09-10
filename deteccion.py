import subprocess
subprocess.Popen(
        "python /Users/rodrigomoralesrivas/PycharmProjects/proyecto_tesis/yolov4-deepsort/object_tracker_modificado.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video 0 --output ./outputs/tiny.avi --tiny",
        shell=True)