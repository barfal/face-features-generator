from datetime import time
import json
from numpy import core
import pandas as pd

def checkPathsCompatibility(face_points_coords, flow_not_flow):
  print('[PATH] face points:', face_points_coords.split('/')[0:4])
  print('[PATH] flow / no flow:', flow_not_flow.split('/')[0:4])
  if (face_points_coords.split('/')[0:4] != flow_not_flow.split('/')[0:4]):
    raise Exception('Paths are not compatible with each other!' + face_points_coords.split('/')[0:4] + ' !! ' + flow_not_flow.split('/')[0:4])


''' Zwraca dataframe z punktami skorelowanymi z etykieta, czasem i precyzja '''
def create_dataframe_points_with_flow_no_flow(face_points_coords, flow_not_flow, corelate_precision_ms = 10):
  
  checkPathsCompatibility(face_points_coords, flow_not_flow)
  ret = pd.DataFrame()

  # konwersja pliku JSON na ciąg znaków, a następnie na słownik w języku Python
  try:
    tmp = json.load(open(face_points_coords))
    tmp2=json.dumps(tmp) 
    face_points_coords=json.loads(tmp2)
  except:
    print("JSON LOAD FAILED FROM FILE: ", face_points_coords)
    return ret

  flowNotFlowDF = pd.read_csv(flow_not_flow)  

  # iteracja po słowniku zawierającym w kluczach czas, a w wartościach koordynaty wykrytych punktów
  for face_points in face_points_coords:

    timeMS = int(face_points)
    #print(face_points_coords[face_points])

    # jesli twarz zostala wykryta to korelacja, jesli nie to zapisz tylko czas
    if (face_points_coords[face_points] != {}):
      #print(face_points_coords[face_points])
    
      # na podstawie oreslenie wspolnego czasu klatkiwideo i analizay facereadera z pewna precyzja wyznacz klase dla danej iteracji ukladu punktow twarzy
      classInThisTime = getClassOfThisTime(
        timeMS=timeMS, 
        precision=corelate_precision_ms, 
        df=flowNotFlowDF) 

      if (classInThisTime != None):

        pointsDF = pd.DataFrame(face_points_coords[face_points])
      
        # Splaszcz dataframe z punktami
        addToReturn = pointsDF.unstack().to_frame().T
        addToReturn.columns = addToReturn.columns.map('{0[0]}_{0[1]}'.format) 

        # Dodaj koluny do dataframe
        addToReturn['time_ms'] = timeMS
        addToReturn['flow'] = classInThisTime
        addToReturn['time_precision'] = corelate_precision_ms

        # dodaj wiersz do koncowego zwracanego dataframe
        ret = ret.append(addToReturn)
      else:
        pass

    else:
      ''' FACE WAS NOT DETECTED - do not need this information '''
      
  return ret


def flatDF(df):
  df.index = df.index + 1
  df_out = df.stack()
  df_out.index = df_out.index.map('{0[1]}_{0[0]}'.format)
  df_out.to_frame().T
  return df_out


''' Kowertuje czas podany w wierszu FaceReaderowych plikow na czas zapisywany w facepoointach '''
def convertToCustomMilisecnd(msf):
  hours, minutes, seconds = msf.split(':')
  secSplit = seconds.split('.')
  if len(secSplit) == 2:
    seconds, milliseconds = secSplit[0], secSplit[1]
  else:
    milliseconds = 0
    seconds = secSplit[0]
  hours, minutes, seconds, milliseconds = map(int, (hours, minutes, seconds, milliseconds))
  milliseconds = milliseconds // 1000 # tak na prawde w plikach z etykietami nie bylo milisekund tylko mikrosekundy
  ret = (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds
  return ret


def getClassOfThisTime(timeMS, precision, df):

    # Zamiana czasu w formacie H:mm:ss.msc na milisekundy
  df['miliseconds'] = df['time'].apply(lambda x: convertToCustomMilisecnd(x))
  
  # Wyciagnij wiersze gdzie czas jest pomiedzy zakresem precyzji i czasu z FaceReadera
  tmp = df[(df['miliseconds'] >= timeMS - precision) & (df['miliseconds'] <= timeMS + precision)]

  # Moze byc taka sytuacja ze beda 2, wiec wtedy wez ten pierwszy (bardzo rzadko)
  # Jak jest jeden, to go uwzgledniaj i skoreluj
  # Jak zero, to zwroc None
  if (len(tmp['flow']) >= 1):
    return tmp['flow'].values[0]

  return None


