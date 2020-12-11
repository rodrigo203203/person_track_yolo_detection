import os
import subprocess
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import argparse
import imutils

import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#instalar mongo para la base de datos
import pymongo
from pymongo import MongoClient

# deep sort imports
from deep_sort import preprocessing, nn_matching, linear_assignment
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import track
# from deep_sort import linear_assignment
from tools import generate_detections as gdet

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')

DETECTION_EVENTS = []
MAXIMUM_AVERAGE_WAITING_TIME_IN_SECONDS = 1000000
MAXIMUM_NUMBER_OF_PEOPLE_DETECTED = 0
MAXIMUM_TIME_WAITING_IN_SECONDS = 1000000
LIMIT_LINE_OF_DETECTION_MIN = 350
LIMIT_LINE_OF_DETECTION_MAX = 1550
client = MongoClient()
db = client.neo_database


def main(_argv):
    # se definen parametros
    global MAXIMUM_NUMBER_OF_PEOPLE_DETECTED
    global COUNTER_OF_PEOPLE_DETECTED
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    t = time.time()

    # inilizacion de deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # configuracion para la deteccion
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    #video_path = 0
    #video_path = 'rtsp://admin:Admin123@172.30.3.71'
    # cargar el modelo tyny
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # carga el modelo yolo normal
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # comienza la carga del video/camara
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # parametro para guardar el video
    if FLAGS.output:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    # ciclo al comenzar el programa
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video finalizado o el formato del video no se puede leer')
            break

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        begin = time.time()
        limit = begin + 20

        # se inicia la deteccion
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.1,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)
        pred_bbox = [bboxes, scores, classes, num_objects]
        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # Activar todas las clases
        #allowed_classes = list(class_names.values())
        # Se elige que clases se va a detectar
        allowed_classes = ['person']

        # se inicia el filtro para separar las clases a detectar
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        # eliminar detecciones que no esten permitidas
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        # yolo + deepsort
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # se inicia color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # incia non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # se llama al tracker
        tracker.predict()

        tracker.update(detections)

        # se actualiza los tracks y se dibujan los cuadros

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            print(bbox)
            print(bbox[0])
            if bbox[1] < LIMIT_LINE_OF_DETECTION_MIN or bbox[1] > LIMIT_LINE_OF_DETECTION_MAX:
                track.is_deleted()
                continue
            class_name = track.get_class()
            # se empieza a dibujar los cuadros
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            font = cv2.FONT_HERSHEY_COMPLEX
            if MAXIMUM_NUMBER_OF_PEOPLE_DETECTED < track.track_id:
                MAXIMUM_NUMBER_OF_PEOPLE_DETECTED = MAXIMUM_NUMBER_OF_PEOPLE_DETECTED + 1
                DETECTION_EVENTS.append({"number_of_people_detected": track.track_id,
                                         "datetime_event": datetime.now(), "detection_position": bbox[-1]})
            cv2.putText(frame, "Personas detectadas: {}".format(count), (5, 35), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (255, 0, 0), 2)
        else:
            print("no hay nada")

        # calculo de fps
        cv2.line(frame, (1400, 0),
                 (350, 1300),
                 (255, 0, 0), 2)
        cv2.line(frame, (1550, 0),
                 (1900, 1300),
                 (255, 0, 0), 2)
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # salvar video
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        if DETECTION_EVENTS:
            waiting_time_in_seconds = calculate_average_waiting_time_in_seconds()
            last_number_of_people_detected = get_last_number_of_people_detected()
            average_waiting_time_in_seconds = waiting_time_in_seconds / last_number_of_people_detected
            print("AVG: {}".format(average_waiting_time_in_seconds))
            diff2 = DETECTION_EVENTS[0]["datetime_event"] - datetime.now()
            max_time = abs(diff2.total_seconds())
            if average_waiting_time_in_seconds >= MAXIMUM_AVERAGE_WAITING_TIME_IN_SECONDS or max_time >= MAXIMUM_TIME_WAITING_IN_SECONDS:
                print("HOLA!")
                max_time = 0
                post = {"date": datetime.now()}
                alarms = db.alarms
                alarms.insert_one(post)
                DETECTION_EVENTS.clear()
                subprocess.run(
                    'python /Users/rodrigomoralesrivas/PycharmProjects/proyecto_tesis/yolov4-deepsort/prueba_deteccion.py',
                    shell=True)

    cv2.destroyAllWindows()


def get_last_number_of_people_detected():
    return DETECTION_EVENTS[-1]['number_of_people_detected']


def create_list_of_closest_person_detected():
    list_closest_person_detected = [0, 0]
    for closest_detecttion in DETECTION_EVENTS:
        person_closes = closest_detecttion[0]['detection_position', 0]
        list_closest_person_detected.append(person_closes)
        max_value = np.max(list_closest_person_detected)
    return max_value


def calculate_average_waiting_time_in_seconds():
    datetime_diff_in_seconds = []
    for oldest_event, newest_event in zip(DETECTION_EVENTS, DETECTION_EVENTS[1:]):
        diff = oldest_event['datetime_event'] - newest_event['datetime_event']
        datetime_diff_in_seconds.append(abs(diff.total_seconds()))
    return sum(datetime_diff_in_seconds)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
