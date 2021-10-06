'''
Map FaceReader outs into FLOW / NOT_FLOW 
'''

import pandas as pd
import datetime
import os
from pathlib import Path


def customFormatVideoTimeToTimeWithMiliseconds(val):
  time = datetime.datetime.strptime(val, '%H:%M:%S.%f').time()
  return time

def formatToFloat(val):
  try:
    return float(val)
  except ValueError:
    return 0


def map_emotion_analyse_to_flow_or_not_flow(filePath, precision = 1.6):
  
  # kilka pierwszych wierszy zostanie pominiętych, gdyż interesująca nas tabela zaczyna się od wiersza 8
  skiprows = [0,1,2,3,4,5,6,7] 
  
  # Wczytany zostanie plik, z którego wyodrębniana jest tabela z pominięciem zadeklarowanych wierszy i wycięciem pierwszych siedmiu kolumn
  df = pd.read_csv(filePath, skiprows=skiprows, sep='\t').iloc[:, 0:8]  
  
  # Deklaracja tablicy z dostępnymi stanami emocji
  emotions = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Scared", "Disgusted"]
  
  # Formatowanie wartości w DataFrame do typu ‘float’
  for em in emotions:
    df[em] = df[em].apply(lambda x: formatToFloat(x))
  
  # Inicjalizacja nowego DataFrame z jedną kolumną ‘time’, która jest wynikiem mapowania kolumny ‘Video Time’ z ciągów znaków na czas
  df_return = pd.DataFrame({'time': df['Video Time'].apply(lambda x: customFormatVideoTimeToTimeWithMiliseconds(x))})
  
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

if __name__ == '__main__':

    saveRootPath = '../DATA_OUT'

    # simulate classes for videos
    facereaderPath = '../DATA_OUT_FACEREADER'
    facereaderFolders = next(os.walk(facereaderPath))[1]
    for faceFolder in facereaderFolders:
      print(faceFolder)

      for f in os.listdir(facereaderPath+"/"+faceFolder):
        print( os.listdir(facereaderPath+"/"+faceFolder))
        # czasami w srodku jest folder "logs zamiast od razu pliki z _details"
        if f == 'Logs':
          for logs in f:
            if logs.endswith("_detailed.txt"):
              fileName = facereaderPath+"/"+faceFolder+"/"+f             
              flowNotFlowDF = map_emotion_analyse_to_flow_or_not_flow(
                filePath= fileName
              )
              lines_to_read = [5]
              with open(fileName, 'r') as a_file:
                for position, line in enumerate(a_file):      
                  if position in lines_to_read:              
                    splitLine = line.split('\\')            
                    recordTypeName = splitLine[-2]
              # save classes of records
              savePath = saveRootPath+"/"+faceFolder+"/"+recordTypeName
              # Utwórz ścieżkę
              Path(savePath).mkdir(parents=True, exist_ok=True)
              flowNotFlowDF.to_csv(savePath+"/flowNotFlow.csv")    

        if f.endswith("_detailed.txt"):
          fileName = facereaderPath+"/"+faceFolder+"/"+f             
          flowNotFlowDF = map_emotion_analyse_to_flow_or_not_flow(
            filePath= fileName
          )
          lines_to_read = [5]
          with open(fileName, 'r') as a_file:
            for position, line in enumerate(a_file):      
              if position in lines_to_read:              
                splitLine = line.split('\\')            
                recordTypeName = splitLine[-2]
          # save classes of records
          savePath = saveRootPath+"/"+faceFolder+"/"+recordTypeName
          # Utwórz ścieżkę
          Path(savePath).mkdir(parents=True, exist_ok=True)
          flowNotFlowDF.to_csv(savePath+"/flowNotFlow.csv")