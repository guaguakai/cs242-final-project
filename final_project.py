# -*- coding: utf-8 -*-
"""Final Project (CS242).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RdkLhYVtekDb95sM_pfAdA0puONt3l2U

# CS242: Final Project
Scalable Newton's Method

Kai Wang, Aditya Mate, Han Ching Ou

> Harvard CS 242: Computing at Scale (Spring 2020)
> 
> Instructor: Professor HT Kung
"""

## Code Cell 1.1

import sys
import time
import os
import math
import tqdm
import scipy
import numpy as np
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

"""**Loading the Train and Test Datasets**

For this assignment, we will use the CIFAR-10 dataset. It contains 10 object classes, where each sample is a color image (RGB channels) with a spatial resolution of 32 x 32 pixels. More details here: https://www.cs.toronto.edu/~kriz/cifar.html. *Code Cell 1.2* will take 1-2 minutes to execute as it downloads the train and test datasets.
"""

## Code Cell 1.2

# Load training data
transform_train = transforms.Compose([                                   
    transforms.RandomCrop(32, padding=4),                                       
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True,
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=8)

# Load testing data
transform_test = transforms.Compose([                                           
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False,
                                         num_workers=2)
print('Finished loading datasets!')

"""**Constructing our Convolutional Neural Network (CNN)**

For this assignment, we will use a 10-layer CNN which we call `ConvNet` that is provided in *Code Cell 1.3*. It has 9 convolutional layers (`nn.Conv2d`) followed by 1 fully connected (`nn.Linear`) layer. The Batch Normalization layers (`nn.BatchNorm2d`) help make the training process more stable and the ReLU layers (`nn.ReLU`) are the non-linear activation functions required for learning when stacking multiple convolutional layers together.

In this assignment, you will modify `ConvNet` to implement Shift-Convolution (Section 2) and Weight Pruning (Section 3).
"""

## Code Cell 1.3

def conv_block(in_channels, out_channels, kernel_size=3, stride=1,
               padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32),
            conv_block(32, 64, stride=2),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 128, stride=2),
            conv_block(128, 128),
            conv_block(128, 256),
            conv_block(256, 256),
            nn.AdaptiveAvgPool2d(1)
            )

        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        h = self.model(x)
        B, C, _, _ = h.shape
        h = h.view(B, C)
        return self.classifier(h)

class SmallConvNet(nn.Module):
    def __init__(self):
        super(SmallConvNet, self).__init__()
        self.model = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64, stride=2),
            conv_block(64, 128),
            conv_block(128, 128, stride=2),
            conv_block(128, 128),
            conv_block(128, 128, stride=2),
            nn.AdaptiveAvgPool2d(1)
            )

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        h = self.model(x)
        B, C, _, _ = h.shape
        h = h.view(B, C)
        return self.classifier(h)

# tracks the highest accuracy observed so far
best_acc = 0

def moving_average(a, n=100):
    '''Helper function used for visualization'''
    ret = torch.cumsum(torch.Tensor(a), 0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def train(epoch, train_loss_tracker, train_acc_tracker):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # update optimizer state
        optimizer.step()
        # compute average loss
        train_loss += loss.item()
        train_loss_tracker.append(loss.item())
        loss = train_loss / (batch_idx + 1)
        # compute accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        # Print status
        sys.stdout.write(f'\rEpoch {epoch}: Train Loss: {loss:.3f}' +  
                         f'| Train Acc: {acc:.3f}')
        sys.stdout.flush()
    train_acc_tracker.append(acc)
    sys.stdout.flush()

def test(epoch, test_loss_tracker, test_acc_tracker):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            test_loss_tracker.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loss = test_loss / (batch_idx + 1)
            acc = 100.* correct / total
    sys.stdout.write(f' | Test Loss: {loss:.3f} | Test Acc: {acc:.3f}\n')
    sys.stdout.flush()
    
    # Save checkpoint.
    acc = 100.*correct/total
    test_acc_tracker.append(acc)
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

"""Newton method implementation."""

def newton_train(epoch, train_loss_tracker, train_acc_tracker):
    from scipy.sparse.linalg import LinearOperator
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    regularization_const = 0.1

    for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward(retain_graph=True)
        for parameter in net.parameters():
            parameter_size = parameter.nelement()
            grad = torch.autograd.grad(loss, parameter, create_graph=True)[0].flatten()
            def mv(v):
                z = grad @ torch.Tensor(v).to(device)
                return torch.autograd.grad(z, parameter, retain_graph=True)[0].cpu().detach().flatten().numpy() + regularization_const * v
            A = LinearOperator((parameter_size, parameter_size), matvec=mv)
            x, info = scipy.sparse.linalg.cg(A, parameter.grad.cpu().detach().flatten(), maxiter=100)
            # x, info = scipy.sparse.linalg.cg(A, parameter.grad.cpu().detach().flatten())
            parameter.grad = torch.Tensor(x.reshape(parameter.grad.shape)).to(device)

        # update optimizer state
        optimizer.step()
        # compute average loss
        train_loss += loss.item()
        train_loss_tracker.append(loss.item())
        loss = train_loss / (batch_idx + 1)
        # compute accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        # Print status
        sys.stdout.write(f'\rEpoch {epoch}: Train Loss: {loss:.3f}' +
                         f'| Train Acc: {acc:.3f}')
        # sys.stdout.flush()
    train_acc_tracker.append(acc)
    sys.stdout.flush()

"""Block Newton's method implementation."""

def explicit_block_newton_train(epoch, train_loss_tracker, train_acc_tracker):
    from scipy.sparse.linalg import LinearOperator
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    num_clients = 5
    regularization_const = 0.1
    minimum_parameter_size = 2000
    update_indices_list = [[]] * num_clients
    A_list = [[]] * num_clients
    with tqdm.tqdm(trainloader) as tqdm_loader:
        for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward(create_graph=True, retain_graph=True)

            for client in range(num_clients):
                if batch_idx % number_batches_recompute == 0:
                    batch_size = len(inputs)
                    # fixed_size_list = [int(parameter.nelement() * fixed_ratio) for parameter in net.parameters()]
                    fixed_size_list = [block_size for parameter in net.parameters()]
                    A_list[client] = [torch.zeros((fixed_size, fixed_size)).to(device) for parameter, fixed_size in zip(net.parameters(), fixed_size_list) if parameter.nelement() >= minimum_parameter_size]
                    update_indices_list[client] = [np.random.choice(parameter.nelement(), fixed_size, p=torch.abs(parameter.grad.flatten().detach()).cpu().numpy()/np.sum(torch.abs(parameter.grad.flatten().detach()).cpu().numpy())) for parameter, fixed_size in zip(net.parameters(), fixed_size_list) if parameter.nelement() >= minimum_parameter_size]

                    parameter_idx = 0
                    for parameter, fixed_size in zip(net.parameters(), fixed_size_list):
                        if parameter.nelement() < minimum_parameter_size:
                            continue
                        for i in range(fixed_size):
                            v = torch.zeros(fixed_size).to(device)
                            v[i] = 1
                    
                            update_indices = update_indices_list[client][parameter_idx]
                            grad = parameter.grad.flatten()[update_indices]
                            grad_v = torch.autograd.grad(grad @ v, parameter, retain_graph=True)[0].flatten()[update_indices] + regularization_const * v # normalization
                            A_list[client][parameter_idx][:,i] = grad_v
                        parameter_idx += 1

                b_list = []
                x = []
                parameter_idx = 0
                for parameter in net.parameters():
                    if parameter.nelement() < minimum_parameter_size:
                        continue
                    update_indices = update_indices_list[client][parameter_idx]
                    b = parameter.grad.detach().flatten()[update_indices]
                    # b_list.append(b[:,None])
                    new_x, LU = torch.solve(b[:,None], A_list[client][parameter_idx])
                    x.append(new_x)
                    # x, info = scipy.sparse.linalg.cg(A.detach().numpy(), parameter.grad.detach().cpu().flatten()[update_indices], maxiter=100)
                    parameter_idx += 1

                # A, b = torch.stack(A_list[client]), torch.stack(b_list)
                # x, LU = torch.solve(b, A)

                # ================= perform line search to find the optimal step ==============
                parameter_idx = 0
                # print('parameter size:', len(list(net.parameters())))
                param_group = list(optimizer.param_groups)[0]
                update_step_list = [[] for parameter in net.parameters()]

                for idx, parameter in enumerate(net.parameters()):
                    old_parameter = copy.deepcopy(parameter)
                    if parameter.nelement() < minimum_parameter_size:
                        continue
                    update_indices = update_indices_list[client][parameter_idx]

                    # line search
                    grad_improvement = parameter.grad.flatten().detach()[update_indices] @ x[parameter_idx][:,0] # precompute the gradient improvement
                    # print('line {} search improvement: {}'.format(idx, grad_improvement))
                    # print('old loss:', loss.item())
                    if grad_improvement > 0:
                        scale = 1 # param_group['lr']
                        parameter.data.flatten()[update_indices] -= 2 * scale * x[parameter_idx][:,0] # -2 grad
                        success = False
                        for linesearch_idx in range(10): # at most 10 iterations of line search
                            alpha_rate = 0.5 ** linesearch_idx * scale
                            ita = 1 
                            parameter.data.flatten()[update_indices] += x[parameter_idx][:,0] * alpha_rate # +1 + 0.5 + 0.25 ... grad, which results in -1 -0.5 -0.25 in total
                            tmp_output = net(inputs).detach()
                            tmp_loss = criterion(tmp_output, targets)
                            compared_value = loss.item() - ita * alpha_rate * grad_improvement
                            # print('linesearch idx: {}, new loss: {}, compared value: {}'.format(linesearch_idx, tmp_loss, compared_value) )
                            if tmp_loss <= compared_value: # if the improvement is good enough
                                success = True
                                break
                        parameter.data = old_parameter.data  # -2 grad
                        if success:
                            # pass 
                            parameter.grad.flatten()[update_indices] = x[parameter_idx][:,0] * alpha_rate / scale
                            # update_step_list[parameter_idx] = x[parameter_idx][:,0] * alpha_rate / (scale) # divide by lr since it will be added back later
                        else:
                            pass

                    # else:
                    #     # do nothing and debug, it is probably due to the non-positive definite issue
                    #     print('line search error with negative improvement:', grad_improvement)

                    parameter_idx += 1

            optimizer.step()

            # compute average loss
            train_loss += loss.item()
            train_loss_tracker.append(loss.item())
            average_loss = train_loss / (batch_idx + 1)
            # compute accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total
            # Print status
            tqdm_loader.set_postfix(loss=f'{average_loss:.3f}', accuracy=f'{acc:.3f}')
            # sys.stdout.write(f'\rEpoch {epoch}: Train Loss: {loss:.3f}' +
            #                  f'| Train Acc: {acc:.3f}')
            # sys.stdout.flush()
        train_acc_tracker.append(acc)
        sys.stdout.flush()


def implicit_block_newton_train(epoch, train_loss_tracker, train_acc_tracker):
    from scipy.sparse.linalg import LinearOperator
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    regularization_const = 0.1
    minimum_parameter_size = 20
    with tqdm.tqdm(trainloader) as tqdm_loader:
        for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            batch_size = len(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward(create_graph=True, retain_graph=True)

            x_list = []
            update_indices_list = []
            for parameter in net.parameters():
                if parameter.nelement() < minimum_parameter_size:
                    x_list.append(None)
                    update_indices_list.append(None)
                    continue

                update_indices = np.random.choice(parameter.nelement(), fixed_size)
                update_indices_list.append(update_indices)
                grad = parameter.grad.flatten()[update_indices]
                def mv(v):
                    z = grad @ torch.Tensor(v).to(device)
                    return torch.autograd.grad(z, parameter, retain_graph=True)[0].cpu().detach().flatten()[update_indices].numpy() + regularization_const * v
                A = LinearOperator((fixed_size, fixed_size), matvec=mv)
                x, info = scipy.sparse.linalg.cg(A, parameter.grad.cpu().detach().flatten()[update_indices]) 
                x = torch.Tensor(x).to(device)
                x_list.append(x)

                grad_improvement = parameter.grad.flatten()[update_indices] @ x # precompute the gradient improvement
                if grad_improvement > 0:
                    parameter.data.flatten()[update_indices] -= 2 * x # -2 grad
                    for linesearch_idx in range(0,5): # at most 10 iterations of line search
                        alpha_rate = 0.5 ** linesearch_idx
                        ita = 0.5
                        parameter.data.flatten()[update_indices] += x * (0.5 ** linesearch_idx) # +1 + 0.5 + 0.25 ... grad, which results in -1 -0.5 -0.25 in total
                        tmp_output = net(inputs).detach()
                        tmp_loss = criterion(outputs, targets)
                        if tmp_loss <= loss.item() - ita * alpha_rate * grad_improvement: # if the improvement is good enough
                            break
                    parameter.grad.flatten()[update_indices] = 0

            # update optimizer state
            optimizer.step()
            # compute average loss
            train_loss += loss.item()
            train_loss_tracker.append(loss.item())
            average_loss = train_loss / (batch_idx + 1)
            # compute accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total
            # Print status
            tqdm_loader.set_postfix(loss=f'{average_loss:.3f}', accuracy=f'{acc:.3f}')
            # sys.stdout.write(f'\rEpoch {epoch}: Train Loss: {loss:.3f}' +
            #                  f'| Train Acc: {acc:.3f}')
            # sys.stdout.flush()
        train_acc_tracker.append(acc)
        sys.stdout.flush()



method = 'explicit' # SGD, explicit, implicit, newton
device = 'cuda'
net = SmallConvNet()
net = net.to(device)
lr = 0.1 # 0.1, 1.0, 0.0001
milestones = [5,10,15,20]
epochs = 25 # 5 or 100
block_size, fixed_ratio = 32, 0.001
number_batches_recompute = 8

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                            weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=milestones,
                                                 gamma=0.1)

# Records the training loss and training accuracy during training
train_loss_tracker, train_acc_tracker = [], []

# Records the test loss and test accuracy during training
test_loss_tracker, test_acc_tracker = [], []

print('Training for {} epochs, with learning rate {} and milestones {}'.format(
      epochs, lr, milestones))

start_time = time.time()
# net.load_state_dict(torch.load('model.pt'))
SGD_warm_start = 0
for epoch in range(0, epochs):
    if method == 'SGD' or epoch < SGD_warm_start:
        train(epoch, train_loss_tracker, train_acc_tracker)
    elif method == 'newton':
        newton_train(epoch, train_loss_tracker, train_acc_tracker)
    elif method == 'explicit':
        explicit_block_newton_train(epoch, train_loss_tracker, train_acc_tracker)
    elif method == 'implicit':
        implicit_block_newton_train(epoch, train_loss_tracker, train_acc_tracker)
    test(epoch, test_loss_tracker, test_acc_tracker)
    scheduler.step()
    # torch.save(net.state_dict(), 'model.pt')

total_time = time.time() - start_time
print('Total training time: {} seconds'.format(total_time))


# ============ plotting the training loss and testing accuracy ============
import matplotlib.pyplot as plt

moving_average_train_loss = moving_average(train_loss_tracker)

plt.xlabel('batches')
plt.ylabel('training loss')
plt.plot(moving_average_train_loss)
plt.savefig('figures/{}_training_loss.png'.format(method))
# plt.show()
plt.clf()

plt.xlabel('epochs')
plt.ylabel('testing accuracy')
plt.plot(list(range(len(test_acc_tracker))), test_acc_tracker)
plt.savefig('figures/{}_testing_acc.png'.format(method))
# plt.show()
plt.clf()

f_result = open('results/{}.csv'.format(method), 'w')
f_result.write('training loss,' + ','.join([str(x.item()) for x in moving_average_train_loss]) + '\n')
f_result.write('testing accuracy,' + ','.join([str(x) for x in test_acc_tracker]) + '\n')
f_result.close()
