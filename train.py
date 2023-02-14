import torch
from YOLO import YOLO
import configs as cfg
from dataloader import get_train_test_data_loaders
#from mpii_dataloader import get_train_test_data_loaders
from YOLOLoss import YOLOLoss
import os
from apex import amp
import apex

def train(model, train_dataloader,test_dataloader= None, epochs=50, mp = True):
    

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    #optimizer = apex.optimizers.FusedSGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    loss_func = YOLOLoss(cfg.GRID_SIZE,cfg.BOXES_PER_CELL,5,.5)
    if mp :
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    save_path = cfg.SAVE_DIR
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_file = open(os.path.join(save_path, cfg.BASE_MODEL+'_log.txt'),'w')

    best_test_loss = 1e9
    for epoch in range(epochs):
        model.train()
        epcoh_loss = 0
        i = 0
        for x,y in train_dataloader:

            x_batch = x.cuda()
            y_batch = y.cuda()

            optimizer.zero_grad()

            out = model(x_batch)

            loss = loss_func(out,y_batch)
            if mp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            step_loss = loss.item()
            epcoh_loss+=step_loss

            print('epoch ',epoch,', step ',i,', with loss',  step_loss)
            i +=1

        test_loss = 0
        if test_dataloader:
            model.eval()
            j=0
            with torch.no_grad():
                for x, y in test_dataloader:
                    x_batch = x.cuda()
                    y_batch = y.cuda()
                    out = model(x_batch)
                    loss= loss_func(out, y_batch)
                    step_loss = loss.item()
                    test_loss += step_loss
                    j+=1
            test_loss/=j

        if test_loss<best_test_loss:
            print('saveing model')
            torch.save(model, os.path.join(save_path,cfg.BASE_MODEL+'.pth'))
            best_test_loss = test_loss

        print('best test loss', best_test_loss)
        log_file.write(str(epcoh_loss/i)+" "+str(test_loss)+'\n')
        log_file.flush()
        print(epcoh_loss/i,test_loss)

        
if __name__ == '__main__':
  
    model = YOLO(cfg.BASE_MODEL,cfg.GRID_SIZE, len(cfg.CLASSES), cfg.BOXES_PER_CELL).cuda()
    #model = torch.load(cfg.MODEL_PATH)
    train_dataloader, test_dataloader = get_train_test_data_loaders(cfg.DATASET_PATH, cfg.BATCH_SIZE)
    model.train()

    train(model,train_dataloader,test_dataloader,epochs = cfg.EPOCHS, mp=False)


