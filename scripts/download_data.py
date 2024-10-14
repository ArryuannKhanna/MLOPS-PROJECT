import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context



def download_data():
    transform = transforms.Compose([transforms.ToTensor()])
    datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)


if __name__=='__main__':
    download_data()
