import torch
from torch import nn

class FC_origin(nn.Module):
    def __init__(self,D_in):
        super(FC_origin,self).__init__()

        self.fc0 = nn.Linear(D_in,19)
        self.relu0 = nn.ReLU()
        self.fc1 = nn.Linear(19,1)

    def forward(self,x):
        x = self.fc0(x)
        x = self.relu0(x)
        x = self.fc1(x)
        x = torch.flatten(x)

        return x

class FC_plus(nn.Module):
    def __init__(self,D_in):
        super(FC_plus,self).__init__()
        self.fc0 = nn.Linear(D_in,64)
        self.relu0 = nn.ReLU()
        self.fc1 = nn.Linear(64,32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32,1)

    def forward(self,x):
        x = self.fc0(x)
        x = self.relu0(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.flatten(x)
        return x

class FC_base(nn.Module):
    def __init__(self,D_in):
        super(FC_base,self).__init__()
        self.fc0 = nn.Linear(D_in,1024)
        self.ln1 = nn.LayerNorm(1024)
        self.relu0 = nn.ReLU()
        self.fc1 = nn.Linear(1024,512)
        self.ln2 = nn.LayerNorm(512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512,1)

    def forward(self,x):
        x = self.fc0(x)
        x = self.ln1(x)
        x = self.relu0(x)
        x = self.fc1(x)
        x = self.ln2(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.flatten(x)
        return x

if __name__ == '__main__':
    input = torch.rand(24).cuda()
    model = FC_origin(24).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('myfc', output.size())

    import datetime
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

