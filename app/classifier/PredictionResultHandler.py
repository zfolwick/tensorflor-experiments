import shutil
from os import path
import numpy as np

class PredictionResultHandler:
    def __init__(self):
        self.__certain_destination_path = path.abspath("human_processing_queue")
        self.__uncertain_destination_path = path.abspath("human_processing_queue")
        
    def move_file(self, src, dest):
        try:
            shutil.move(src, dest)
        except Exception as e:
            print(f"Error: {e}")
            
    # currently, just moving a file from one place on the local file system to another
    def handle(self, src_result):
        fullpath, score_value = src_result
        probability_modified_score = np.array(score_value)[0]
        
        if ( probability_modified_score > .50):
            self.move_file(
                fullpath, 
                path.join(self.__certain_destination_path, "proposed_modified")
            )
        else:
            self.move_file(
                fullpath, 
                path.join(self.__uncertain_destination_path, "proposed_original")
            )

