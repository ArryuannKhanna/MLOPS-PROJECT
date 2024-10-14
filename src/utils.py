import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loader(batch_size = 64):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root = './data',
        train = True ,
        download = False , 
        transform = transform
    )

    testset = torchvision.datasets.CIFAR10(
        root = './data',
        train = False ,
        download = False , 
        transform = transform
    )

    trainloader = torch.utils.data.DataLoader(trainset,batch_size,shuffle=True)
    testloader = torch.utils.data.DataLoader(testset,batch_size,shuffle=False)

    return trainloader,testloader