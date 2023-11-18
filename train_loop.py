
#Общий модуль для тренировок моделей

import torch

def train_loop(_device, _max_epochs, _model, 
_criterion, _optimizer, _loaders, _is_fc, _is_break):

    

    accuracy = {"train": [], "valid": []}
    history = []
    i = 0
    for epoch in range(_max_epochs):
        epoch_correct = 0
        epoch_all = 0
        print("")

        for k, dataloader in _loaders.items():
            acc = 0
            i = 0
            total = len(dataloader)
            for x_batch, y_batch in dataloader:
                if (_is_break):
                    if (i>100):
                        break

                if k == "train":  
                    _model.train()
                    if (_is_fc == True):
                        x_batch = x_batch.view(x_batch.shape[0], -1).to(_device)
                    y_batch = y_batch.to(_device)
                    outp = _model(x_batch)
                    loss = _criterion(outp, y_batch)   
                    _optimizer.zero_grad()
                    loss.backward()   
                    _optimizer.step()
                    preds = torch.argmax(_model(x_batch),dim=1)             
                else:
                    _model.eval()
                    with torch.no_grad():
                        outp = _model(x_batch)
                        preds = torch.argmax(_model(x_batch), dim=1)    
                        
                epoch_correct = (preds==y_batch).cpu().numpy().sum()/len(preds)
                if (i % 100 == 0):
                    print(f"{k} Epoch: {epoch+1}/{_max_epochs} \t Iteration:{i}/{total} \t Accuracy: {epoch_correct}")
                i+=1

            print(f"Loader: {k} \t Accuracy: {epoch_correct}")    

            accuracy[k].append(epoch_correct)
    return accuracy