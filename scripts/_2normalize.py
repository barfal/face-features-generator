import sys
import utilsForModels as ufm
import os 

''' Split corelated csv for two classes '''

# Wez jeden z folderów DATA_OUT
rootPath = '../DATA_OUT'
rootFolders = next(os.walk(rootPath))[1]
POSSIBLE_FOLDERS = ['Fibonacci', 'GetHello', 'Palindrome', 'SortInOrder', 'TransposeMatrix']


for rFolder in rootFolders:
    inFolders = os.listdir(rootPath + '/' + rFolder)

    inFacereaderFolders = {}
    for oneFolder in inFolders:
      # Zabezpieczenie przed wystepujacym folderem lub plikiem, ktorego nie chcemy korelowac (np. wystepowal ukryty plik 'Icon?')
      if oneFolder in POSSIBLE_FOLDERS: 

        oneFolderDir = rootPath + '/' + rFolder+"/"+oneFolder
        files = os.listdir(oneFolderDir)

        ''' split corelated on two classes '''
        noflowPath, flowPath = ufm.split_corelated_csv_for_classes(
          corelated_csv_dir=oneFolderDir+'/corelated.csv'
        )   

        ''' And generate bitmaps from it '''
        ufm.get_bitmaps_from_csvs(
          flow_file_dir=flowPath,
          noflow_file_dir=noflowPath,
        )



