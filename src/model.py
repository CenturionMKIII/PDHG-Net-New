import torch
import torch.nn as nn


class pdhg_net(torch.nn.Module):
    
    def __init__(self,x_size,y_size,feat_sizes):
        super(pdhg_net,self).__init__()
        
        self.cons_embedding = nn.Sequential(
            nn.Linear(in_features=y_size, out_features=feat_sizes[0], bias=True),
            nn.ReLU()
        )
        self.var_embedding = nn.Sequential(
            nn.Linear(in_features=x_size, out_features=feat_sizes[0], bias=True),
            nn.ReLU()
        )

        self.layers = nn.ModuleList()
        for indx in range(len(feat_sizes)-1):
            self.layers.append(pdhg_layer(feat_sizes[indx],feat_sizes[indx],feat_sizes[indx+1]))

            
        self.output_module1 = nn.Sequential(
            nn.Linear(in_features=feat_sizes[-1], out_features=feat_sizes[-1], bias=True),
            nn.ReLU(),
            nn.Linear(in_features=feat_sizes[-1], out_features=1, bias=False)
        )
        self.output_module2 = nn.Sequential(
            nn.Linear(in_features=feat_sizes[-1], out_features=feat_sizes[-1], bias=True),
            nn.ReLU(),
            nn.Linear(in_features=feat_sizes[-1], out_features=1, bias=False)
        )
    
    def update(self,x,y,A,AT,c,b):

        x = self.var_embedding(x)
        y = self.cons_embedding(y)

        
        for index, layer in enumerate(self.layers):
            x,y = layer(x,y,A,AT,c,b)

        x = self.output_module1(x)
        y = self.output_module1(y)

        return x,y



class pdhg_layer(torch.nn.Module):
    
    def __init__(self,x_size,y_size,out_size):
        super(pdhg_layer,self).__init__()
        self.left = pdhg_layer_x(x_size,out_size)
        self.right = pdhg_layer_y(y_size,out_size)
    
    def update(self,x,y,A,AT,c,b):
        x = self.left(x,y,AT,c)
        y = self.right(x,y,A,b)
        return x,y
    

    

class pdhg_layer_x(torch.nn.Module):
    
    def __init__(self,in_size,out_size):
        super(pdhg_layer,self).__init__()
        self.Ukx = nn.Linear(in_features=in_size, out_features=out_size)
        self.Uky = nn.Linear(in_features=in_size, out_features=out_size)
        self.tau= nn.Parameter(torch.randn(size=(1, ),requires_grad=True))
        self.act = nn.ReLU()
        self.one = torch.ones(1,out_size)
    
    def update(self,x,y,AT,c):
        return self.act(self.Ukx(x) - self.tau * 
                        (torch.matmul(c,self.one) - 
                         torch.matmul(AT, self.Uky(y))))
    
    

class pdhg_layer_y(torch.nn.Module):
    
    def __init__(self,in_size,out_size):
        super(pdhg_layer,self).__init__()
        self.Vky = nn.Linear(in_features=in_size, out_features=out_size)
        self.Wkx = nn.Linear(in_features=in_size, out_features=out_size)
        self.Vkx = nn.Linear(in_features=in_size, out_features=out_size)
        self.sigma= nn.Parameter(torch.randn(size=(1, ),requires_grad=True))
        self.act = nn.ReLU()
        self.one = torch.ones(1,out_size)
    
    def update(self,x,y,A,b):
        
        return self.act(self.Vky(y) - self.sigma * 
                        (torch.matmul(b,self.one) - 
                         2 * torch.matmul(A, self.Wkx(x)) +
                          torch.matmul(A, self.Vkx(x))))

