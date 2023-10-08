import torch
import torch.nn.functional as F
import torch.utils.data as Data

class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_of_labels):
        super(LinearClassifier, self).__init__()
        self.layer = torch.nn.Linear(input_dim, num_of_labels)

    def forward(self, x):
        x = self.layer(x)
        return F.softmax(x)

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_of_labels):
        super(MLPClassifier, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, 2048)
        self.layer2 = torch.nn.Linear(2048, 1024)
        self.layer3 = torch.nn.Linear(1024, 512)
        self.layer4 = torch.nn.Linear(512, num_of_labels)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return F.softmax(x)

def train(model, X, y):
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fun = torch.nn.CrossEntropyLoss()

    train_dataset = Data.TensorDataset(X, y)
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    for epoch in range(15):
        for i, (x, y) in enumerate(train_dataloader):
            optim.zero_grad()
            out = model(x.cuda())
            loss = loss_fun(out, y.cuda())
            loss.backward()
            optim.step()
    
    return model