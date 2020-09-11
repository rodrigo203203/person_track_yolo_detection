import sched, time
import object_tracker_modificado
s = sched.scheduler(time.time, time.sleep)
global MAXIMUM_NUMBER_OF_PEOPLE_DETECTED
def preguntar(sc):
       cant_person = 1
       print(MAXIMUM_NUMBER_OF_PEOPLE_DETECTED)
       if object_tracker_modificado.track.cant_actual == cant_person or object_tracker_modificado.cant_actual > cant_person:
              print(10+5)
              object_tracker_modificado.max_person_counter = 0
       else:
              object_tracker_modificado.max_person_counter = 0
              print(10)
       s.enter(10, 1, preguntar, (sc,))

s.enter(10, 1, preguntar, (s,))
s.run()