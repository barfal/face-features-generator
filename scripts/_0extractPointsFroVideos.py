'''
Extracts face points location from videos
'''

import os
from pathlib import Path
import generateFunctions as generate

rootPath = '../DATA'
saveRootPath = '../DATA_OUT'


rootFolders = next(os.walk(rootPath))[1]
# slice when you want continue from 
# rootFolders = rootFolders[46:]

for rFolder in rootFolders:
  inFolders = os.listdir(rootPath + '/' + rFolder)
  for inFolder in inFolders:

    # Scieżka pliku z nagraniem wideo
    videoFile = rootPath+'/'+rFolder+'/'+inFolder+'/av.mp4'

    # Ścieżka do której będą zapisane wyodrębnione punkty
    savePath = saveRootPath+'/'+rFolder+'/'+inFolder

    # Utwórz ścieżkę
    Path(savePath).mkdir(parents=True, exist_ok=True)

    # W utworzone ścieżce utwórz pusty plik
    saveFile = savePath+'/points.json'
    print(saveFile)
    open(saveFile, 'a').close()

    generate.extract_coordinates_from_face_video(
      video_path = videoFile,
      save_file = saveFile
    )

