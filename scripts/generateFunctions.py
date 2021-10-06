import pandas as pd
import cv2
import time
import tools

''' [ 1 ] '''
def map_emotion_analyse_to_flow_or_not_flow(filePath, precision = 1.6):
  
  # kilka pierwszych wierszy zostanie pominiętych, gdyż interesująca nas tabela zaczyna się od wiersza 8
  skiprows = [0,1,2,3,4,5,6,7] 
  
  # Wczytany zostanie plik, z którego wyodrębniana jest tabela z pominięciem zadeklarowanych wierszy i wycięciem pierwszych siedmiu kolumn
  df = pd.read_csv(filePath, skiprows=skiprows, sep='\t').iloc[:, 0:8]  
  
  # Deklaracja tablicy z dostępnymi stanami emocji
  emotions = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Scared", "Disgusted"]
  
  # Formatowanie wartości w DataFrame do typu ‘float’
  for em in emotions:
    df[em] = df[em].apply(lambda x: tools.formatToFloat(x))
  
  # Inicjalizacja nowego DataFrame z jedną kolumną ‘time’, która jest wynikiem mapowania kolumny ‘Video Time’ z ciągów znaków na czas
  df_return = pd.DataFrame({'time': df['Video Time'].apply(lambda x: tools.customFormatVideoTimeToTimeWithMiliseconds(x))})
  
  # dodawana jest nowa kolumna do wynikowego DataFrame ‘flow’ z domyślną wartość 1, domyślna wartość wynika z następującej heurystyki: potencjalny klasyfikator nie powinien nadużywać komunikatów ‘no flow’, gdyż naturalną etykietą powinno być ‘flow’
  df_return['flow'] = 1
  
  # warunki, które określają etykietę ‘not flow’. Metoda loc() wykonana na DataFrame działa jak referencja
  df_return.loc[((df['Neutral'] ) < df["Sad"] * precision), 'flow'] = 0
  df_return.loc[((df['Neutral'] ) < df["Angry"] * precision), 'flow'] = 0
  df_return.loc[((df['Neutral'] ) < df["Surprised"] * precision), 'flow'] = 0
  df_return.loc[((df['Neutral'] ) < df["Scared"] * precision), 'flow'] = 0
  df_return.loc[((df['Neutral'] ) < df["Disgusted"] * precision), 'flow'] = 0

  df_return.loc[((df['Happy'] ) < df["Sad"] * precision), 'flow'] = 0
  df_return.loc[((df['Happy'] ) < df["Angry"] * precision), 'flow'] = 0
  df_return.loc[((df['Happy'] ) < df["Surprised"] * precision), 'flow'] = 0
  df_return.loc[((df['Happy'] ) < df["Scared"] * precision), 'flow'] = 0
  df_return.loc[((df['Happy'] ) < df["Disgusted"] * precision), 'flow'] = 0
  
  # Funkcja zwraca DataFrame z kolumnami czas oraz flow, który może zostać wykorzystany do korelacji z punktami twarzy, lub bezpośrednio zapisany jako plik csv
  return df_return 


''' [2] '''
def extract_coordinates_from_face_video(video_path, save_file):

  cap = cv2.VideoCapture(video_path)
  #video_frame = 0
  start_time = time.time()
  tmp_json = {}

  open(save_file, 'w').write('{\n')

  while(cap.isOpened()): 
    try:
      _, frame = cap.read()
      # Convert image into grayscale
      gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    except:
      break

    faces = tools.DETECTOR(gray)

    for face in faces:
      x1 = face.left()
      y1 = face.top()
      x2 = face.right()
      y2 = face.bottom()

      landmarks = tools.PREDICTOR(image=gray, box=face)
      tmp_json = {}  
      for n in range(0, 68): # loop przez punkty 
        x = landmarks.part(n).x
        y = landmarks.part(n).y  
        tmp_json[str(n)] = {'x': x, 'y': y}  # alternatywnie mozna zapisać względem wykrytego obszaru {'x': x-x1, 'y': y-y1}
              
        # Rysowanie punktu na klatce wideo
        cv2.circle(img=frame, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)

    cv2.imshow(winname="Face", mat=frame)  # Pokazywanie klatki wideo

    time_break = time.time()
    time_stamp = time_break-start_time
    milliseconds = int(round(time_stamp * 1000))

    tools.save_frame_coords(tmp_json, str(milliseconds), save_file) #alternatywnie można przekazać klatkę wideo zamiast video_time jako klucz 
    #video_frame += 1

    # Wyłącz jak wciśniesz escape
    if cv2.waitKey(delay=1) == 27:
      break

  cap.release()
  cv2.destroyAllWindows()

  # Usuń ostatni przecinek
  with open(save_file, 'rb+') as f:
    f.seek(0,2)           
    size=f.tell()         
    f.truncate(size-1)     

  # i ostatni znak '}'
  with open(save_file, 'a') as f:
    f.write('}')

  print('Extracted points from video: ' + video_path)


