from model import CNNModel
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_data_loader
import logging
import os 

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',handlers=[
    logging.FileHandler("logs/training.log"),
    logging.StreamHandler()
])


def train_model(epochs=10,batch_size=64,learning_rate=0.001):

    logging.info("Loading the dataset...")
    trainloader,testloader = get_data_loader()

    logging.info("Loading the model...")
    model = CNNModel()
    optimiser = optim.Adam(model.parameters(),lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


    logging.info(f"Starting the training for total of {epochs} epochs")

    for epoch in range(epochs):
        run_loss = 0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels = data
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimiser.step()
            run_loss += loss.item()

            if i%100==99:
                avg_loss = run_loss/100
                logging.info(f'Epoch [{epoch + 1}, Batch {i + 1}] - Loss: {avg_loss:.3f}')
                run_loss = 0
    logging.info('Training finished')

    model_save_path = './models/cnn_cifar10.pth'
    torch.save(model.state_dict(),model_save_path)
    logging.info('Model saved successfully')

if __name__ == '__main__':
    try:
        epochs = 10
        batch_size = 64
        learning_rate = 0.001
        train_model(epochs=epochs,batch_size=batch_size,learning_rate=learning_rate)

    except Exception as e:
        logging.info(f'Error training the model!!!-',e)
    



