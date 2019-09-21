import torch
import torch.nn as nn
import random
import numpy as np
import os
import matplotlib.pyplot as plt

def get_data():
    status = 0

    input = np.zeros(100, dtype=np.float32)
    target = np.zeros(100, dtype=np.float32)
    pointer = 0

    while(pointer<100):
        if status == 0:
            n = random.randint(1,10)
            pointer += n
            status = 1
        else:
            n = random.randint(1,10)
            target[pointer:pointer+n] = 1
            input[pointer] = n
            pointer += n
            status = 0

    return input, target
        
class MODEL(nn.Module):
    def __init__(self, input_size):
        super(MODEL, self).__init__()
        self.rnn = nn.RNN(1, 10, 1, batch_first=True)
        self.out = nn.Linear(10, 1)
        self.h_n = None

    def forward(self, x):
        if self.training:
            r_out, self.h_n = self.rnn(x, None) 
        else:
            r_out, self.h_n = self.rnn(x, self.h_n) 
        out = self.out(r_out)
        return out


net = MODEL(1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if os.path.exists('model.pth'):
    net.load_state_dict(torch.load('model.pth'))
else:
    net.train()
    for i in range(10000):
        input,target = get_data()

        input = torch.from_numpy(input).reshape([1,100,1])
        target = torch.from_numpy(target).reshape([1,100,1])

        h0 = torch.randn(1, 1, 20)
        output = net(input)

        loss = torch.sum(torch.pow(target - output, 2))
        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(net.state_dict(), 'model.pth')



net.eval()
input,target = get_data()
print(input)
print(target)
output = []

for n in range(100):
    i = input[n]
    i = torch.tensor([i]).reshape([1,1,1])

    out = net.forward(i)

    output.append(out.detach().numpy()[0][0][0])


print(output)

plt.plot(input, 'r')
plt.plot(output, 'g')
plt.show()
