import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def calculate_mean_image():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('dataset/',train = True,download = True,
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean = (0.5,), std = (0.5,))
                    ])),
        batch_size = 60000,
        shuffle = True)

    # Calculate mean images
    mean_images = np.zeros((10,28,28))
    for batch_idx, (data, label) in enumerate(train_loader):
        for i in range(10):
            mean_image = torch.mean(data[label == i], dim=0)[0]
            mean_images[i,:,:] = mean_image
            
    return mean_images

def one_layer(epoch, lr, reg, is_svm, batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('dataset/',train = True,download = True,
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean = (0.5,), std = (0.5,))
                    ])),
        batch_size = batch_size,
        shuffle = False,
        drop_last = True)
    
    W1 = 0.01*np.random.randn(784,10)
    b1 = np.zeros((1,10))
    
    for e in range(1,epoch+1):
        loss = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            image = data.reshape(batch_size,-1)
            scores = np.dot(image,W1)+b1
            
            if not is_svm:
                # Cross-entropy Loss
                exp_scores = np.exp(scores)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                correct_logprobs = -np.log(probs[range(batch_size),label])
                data_loss = np.sum(correct_logprobs)/batch_size
                
                # Cross-entropy backpropagation
                dscores = probs
                dscores[range(batch_size),label] -= 1
                dscores /= batch_size
            
            else:
                # SVM Loss
                margin = 1
                margins = np.maximum(0, scores - scores[np.arange(batch_size),label][:, np.newaxis] + margin)
                margins[np.arange(batch_size), label] = 0
                data_loss = np.sum(margins)/batch_size               
                
                # SVM backpropagation
                num_margin = np.sum(margins>0, axis=1)
                dscores = np.where(margins>0, 1/batch_size, 0)
                dscores[np.arange(batch_size), label] -= num_margin/batch_size
            
            reg_loss = 0.5*reg*np.sum(W1*W1)
            loss += (data_loss + reg_loss) 
            
            dW = np.dot(image.T, dscores)
            db = np.sum(dscores, axis=0, keepdims=True)
            
            dW += reg*W1

            W1 += -lr * dW
            b1 += -lr * db
        loss /= (batch_idx+1)
        print("epoch",e,"loss",loss)

    # Test
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('dataset',train=False,
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5,), std=(0.5,))
                    ])),
        batch_size = 10000,
        shuffle = False)

    for batch_idx, (data, label) in enumerate(test_loader):
        image = data.reshape(-1, 784)
        scores = np.dot(image,W1)+b1
        predictions = torch.tensor(np.argmax(scores, axis=1))
        correct = torch.sum(torch.eq(predictions, label)).item()
        accuracy = correct/100
    print("accuracy:",accuracy,"%")

    weight_matrix = (W1+b1).T.reshape((10,28,28))
    return loss, weight_matrix, accuracy

def two_layer(epoch, lr, reg, is_svm, batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('dataset/',train = True,download = True,
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean = (0.5,), std = (0.5,))
                    ])),
        batch_size = batch_size,
        shuffle = False,
        drop_last = True)
    
    W1 = 0.01*np.random.randn(784,128)
    b1 = np.zeros((1,128))

    W2 = 0.01*np.random.randn(128,10)
    b2 = np.zeros((1,10))

    for e in range(1,epoch+1):
        loss = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            image = data.reshape(batch_size,-1)
            hidden_layer = np.maximum(0,np.dot(image,W1)+b1)
            scores = np.dot(hidden_layer,W2)+b2
            
            if not is_svm:
                # Cross-entropy Loss
                exp_scores = np.exp(scores)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # 1000x10
                correct_logprobs = -np.log(probs[range(batch_size),label])
                data_loss = np.sum(correct_logprobs)/batch_size
                
                # Cross-entropy backpropagation
                dscores = probs
                dscores[range(batch_size),label] -= 1
                dscores /= batch_size
            
            else:
                # SVM Loss
                margin = 1
                margins = np.maximum(0, scores - scores[np.arange(batch_size),label][:, np.newaxis] + margin)
                margins[np.arange(batch_size), label] = 0
                data_loss = np.sum(margins)/batch_size                
                
                # SVM backpropagation
                num_margin = np.sum(margins>0, axis=1)
                dscores = np.where(margins>0, 1/batch_size, 0)
                dscores[np.arange(batch_size), label] -= num_margin/batch_size
            
            reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
            loss += (data_loss + reg_loss)
            
            dW2 = np.dot(hidden_layer.T, dscores)
            db2 = np.sum(dscores, axis=0, keepdims=True)
            dhidden = np.dot(dscores, W2.T)
            dhidden[hidden_layer<=0] = 0       
            dW1 = np.dot(image.T, dhidden)
            db1 = np.sum(dhidden, axis=0, keepdims=True)
            
            dW1 += reg*W1
            dW2 += reg*W2

            W1 += -lr * dW1
            b1 += -lr * db1
            W2 += -lr * dW2
            b2 += -lr * db2
            
        loss /= (batch_idx+1)
        print("epoch",e,"loss",loss)

    # Test
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('dataset',train=False,
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5,), std=(0.5,))
                    ])),
        batch_size = 10000,
        shuffle = False)

    for batch_idx, (data, label) in enumerate(test_loader):
        image = data.reshape(-1, 784)
        scores = np.dot(np.maximum(0,np.dot(image,W1)+b1),W2)+b2
        predictions = torch.tensor(np.argmax(scores, axis=1))
        correct = torch.sum(torch.eq(predictions, label)).item()
        accuracy = correct/100
    print("accuracy:",accuracy,"%")
    
    weight_matrix = np.dot(W1+b1,W2+b2).T.reshape((10,28,28))
    return loss, weight_matrix, accuracy

def plot_images(images, file_name):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'{i}')
        ax.axis('off')
    plt.savefig(file_name + '.jpg')
    plt.show()

def save_weight(weight_matrix, file_name):
    np.save(file_name + '.npy', weight_matrix)
    
def load_weight(file_name):
    return np.load(file_name + '.npy')

loss, weight_matrix, accuracy = two_layer(5, 1e-3, 1e-3, True, 1000)
#save_weight(weight_matrix, 'weight1')
plot_images(weight_matrix, 'weight_matrix')

mean_images = calculate_mean_image()
plot_images(mean_images, 'mean_images')
