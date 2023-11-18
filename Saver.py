#Name of files
import json
import datetime
import torch

class Saver():
    def __init__(self, title, path): 
        super(Saver, self).__init__() 

        x = datetime.datetime.now()
        strx = x.strftime("%Y")+x.strftime("%m")+x.strftime("%d")
        self.title = title
        self.path = path
        self.path_model = f"{path}/model_{title}_{strx}"
    
    def SaveAll(self, model, accuracy):
        #сохранение модели
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(self.path_model + ".pt") # Save

        json_object = json.dumps(accuracy)

        with open(self.path_model + "_accuracy.json", "w") as outfile:
            outfile.write(json_object)
    
    def SaveModel(self, model):
        #сохранение модели
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(self.path_model + ".pt") # Save

    def SaveAccuracy(self, accuracy):
        
        json_object = json.dumps(accuracy)

        with open(self.path + self.title +"_accuracy.json", "w") as outfile:
            outfile.write(json_object)