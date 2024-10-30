from model import *
import torch 
import gzip
import pickle
import os
import random

flist_train = os.listdir('./data/train')
flist_valid = os.listdir('./data/valid')


x_size = 4
y_size = 2
feat_sizes=[64,64,64]

mdl = pdhg_net(x_size,y_size,feat_sizes)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mdl.parameters(), lr=1e-4)

max_epoch = 10000

flog = open('./logs/train_log.log','w')

best_loss = 1e+20

for epoch in range(max_epoch):
    avg_loss_x=0.0
    avg_loss_y=0.0
    random.shuffle(flist_train)
    for fnm in flist_train:
        # train
        f = gzip.open(f'./data/train/{fnm}','rb')
        A,b,c,x,y,sol,dual,obj = pickle.load(f)
        A = torch.as_tensor(A,dtype=torch.float32)
        AT = A.T
        x = torch.as_tensor(x,dtype=torch.float32)
        y = torch.as_tensor(y,dtype=torch.float32)
        b = torch.as_tensor(b,dtype=torch.float32)
        c = torch.as_tensor(c,dtype=torch.float32)
        x_gt = torch.as_tensor(sol,dtype=torch.float32)
        y_gt = torch.as_tensor(dual,dtype=torch.float32)
        f.close()

        #  apply gradient 
        optimizer.zero_grad()
        x,y = mdl(x,y,A,AT,c,b)
        loss_x = loss_func(x, x_gt)
        loss_y = loss_func(y, y_gt) 
        loss = loss_x + loss_y
        avg_loss_x += loss_x.item()
        avg_loss_y += loss_y.item()
        loss.backward()
        optimizer.step()
    avg_loss_x /= round(len(flist_train),2)
    avg_loss_y /= round(len(flist_train),2)
    print(f'Epoch {epoch} Train:::: primal loss:{avg_loss_x}    dual loss:{avg_loss_y}')
    st = f'epoch{epoch}train: {avg_loss_x} {avg_loss_y}'
    flog.write(st)



    avg_loss_x=0.0
    avg_loss_y=0.0
    for fnm in flist_valid:
        # valid
        #  reading
        f = gzip.open(f'./data/valid/{fnm}','rb')
        A,b,c,x,y,sol,dual,obj = pickle.load(f)
        A = torch.as_tensor(A,dtype=torch.float32)
        AT = A.T
        x = torch.as_tensor(x,dtype=torch.float32)
        y = torch.as_tensor(y,dtype=torch.float32)
        b = torch.as_tensor(b,dtype=torch.float32)
        c = torch.as_tensor(c,dtype=torch.float32)
        x_gt = torch.as_tensor(sol,dtype=torch.float32)
        y_gt = torch.as_tensor(dual,dtype=torch.float32)
        f.close()
        #  obtain loss
        x,y = mdl(x,y,A,AT,c,b)
        loss_x = loss_func(x, x_gt)
        loss_y = loss_func(y, y_gt) 
        loss = loss_x + loss_y
        avg_loss_x += loss_x.item()
        avg_loss_y += loss_y.item()
    avg_loss_x /= round(len(flist_train),2)
    avg_loss_y /= round(len(flist_train),2)
    print(f'Epoch {epoch} Valid:::: primal loss:{avg_loss_x}    dual loss:{avg_loss_y}')
    st = f'epoch{epoch}train: {avg_loss_x} {avg_loss_y}'
    flog.write(st)

    if best_loss > avg_loss_x+avg_loss_y:
        best_loss = avg_loss_x+avg_loss_y
        torch.save(mdl.state_dict(), f'./model/best_model.mdl')
        print(f'Saving new best model with valid loss: {best_loss}')

    flog.flush()


flog.close()