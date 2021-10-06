'''
Colerate classes FLOW / NOT FLOW and face points 
'''


import corealateData as corelate
import glob, os

POSSIBLE_FOLDERS = ['Fibonacci', 'GetHello', 'Palindrome', 'SortInOrder', 'TransposeMatrix']

# Wez jeden z folderów DATA_OUT
rootPath = '../DATA_OUT'
rootFolders = next(os.walk(rootPath))[1]

# slice if you want to skip smth
#rootFolders = rootFolders[1:]

for rFolder in rootFolders:
    
    inFolders = os.listdir(rootPath + '/' + rFolder)

    inFacereaderFolders = {}
    for oneFolder in inFolders:
      # Zabezpieczenie przed wystepujacym folderem lub plikiem, ktorego nie chcemy korelowac (np. wystepowal ukryty plik 'Icon?')
      if oneFolder in POSSIBLE_FOLDERS: 

        oneFolderDir = rootPath + '/' + rFolder+"/"+oneFolder
        files = os.listdir(oneFolderDir)

        # jesli zawiera komplet punkty oraz klasy flow/not flow to koreluj
        if 'points.json' in files and 'flowNotFlow.csv' in files:

          corelatedDF = corelate.create_dataframe_points_with_flow_no_flow(
            face_points_coords= oneFolderDir+'/points.json',
            flow_not_flow= oneFolderDir+"/flowNotFlow.csv",
            corelate_precision_ms = 10,
          )      

          if not corelatedDF.empty:
            print("save corelated df to location: ", oneFolderDir+'/corelated.csv')       
            corelatedDF.to_csv(oneFolderDir+'/corelated.csv')  
          
    