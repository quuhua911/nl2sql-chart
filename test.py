import torch
import torch.nn as nn
def celoss():
    batch_size = 5
    nb_classes = 10
    model = nn.Linear(20, nb_classes)

    x = torch.randn(batch_size, 20)
    target = torch.empty(batch_size, dtype=torch.long).random_(nb_classes)

    criterion = nn.CrossEntropyLoss()

    output = model(x)
    loss = criterion(output, target)

a = []
a.append([])
a.append("b")
print(a)