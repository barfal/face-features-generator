import datetime
import dlib

DETECTOR = dlib.get_frontal_face_detector() # instancja detektora z bilbioteki dlib
PREDICTOR = dlib.shape_predictor('utils/shape_predictor_68_face_landmarks.dat') # instancja predyktora - funkcja z dlib, a parametrem jest pobrany wytrenowany model 68_poinst_costam...

def customFormatVideoTimeToTimeWithMiliseconds(val):
  time = datetime.datetime.strptime(val, '%H:%M:%S.%f').time()
  return time

def formatToFloat(val):
  try:
    return float(val)
  except ValueError:
    return 0

def save_frame_coords(tmp_json, time_stamp, coords_file_path):
  with open(coords_file_path, 'a') as outfile:
    line = "\"%s\": {\n" % (time_stamp)
    outfile.write(line)
    for key in tmp_json.keys():
      line = "\"f_%s\": {\"x\": %d, \"y\": %d}" % (key, tmp_json[key]['x'], tmp_json[key]['y'])
      if (int(key) < 67): 
        line += ", \n"
      outfile.write(line)
    line = "},"
    outfile.write(line)