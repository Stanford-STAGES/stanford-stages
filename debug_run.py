from PyQt5.QtWidgets import QApplication
import run_stanford_stages

# narco_gui: C:\Users\hyatt\AppData\Local\Programs\Python\Python36\python.exe C:/Git/narco_class_app/narco_gui.py
# debug_run: C:\Users\hyatt\AppData\Local\Programs\Python\Python36\python.exe C:/Git/stanford-stages/debug_run.py
json_file='C:\\Data\\tmp\\wsc_edf\\stanford_stages.json'
json_file='E:\\validation_set\\edf\\stanford_stages.json'
run_stanford_stages.run_using_json_file(json_file=json_file)
