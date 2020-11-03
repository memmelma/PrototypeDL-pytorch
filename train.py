import torch
import torchvision
from torchvision import transforms

from model import CAE
from network import *
from autoencoder_helpers import makedirs

from data_preprocessing import batch_elastic_transform

import time

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = 'cuda:5'

save_step = 100
test_display_step = 100

lambda_class = 20
lambda_ae = 1
lambda_r1 = 1
lambda_r2 = 1

learning_rate = 1e-4 # 2e-3
training_epochs = 1500
batch_size = 250
n_workers = 4

n_prototype_vectors = 15

sigma = 4
alpha =20

img_shape = (28,28)

# dataloader setup
mnist_data = torchvision.datasets.MNIST(root='dataset', train=True, download=True, transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True, num_workers=n_workers)
mnist_data_test = torchvision.datasets.MNIST(root='dataset', train=False, download=True, transform=transforms.ToTensor())
data_loader_test = torch.utils.data.DataLoader(mnist_data_test, batch_size=batch_size, shuffle=True, num_workers=n_workers)

# model setup
model = CAE(input_dim=(1,1,img_shape[0], img_shape[1]), n_prototype_vectors=n_prototype_vectors).to(device)

# optimizer setup
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for e in range(0, training_epochs):
    train_acc = 0
    max_iter = len(data_loader)
    start = time.time()

    for i, batch in enumerate(data_loader):
        imgs = batch[0]
        # to apply elastic transform, batch has to be flattened
        # store original size to put it back into orginial shape after transformation
        imgs_shape = imgs.shape
        imgs = batch_elastic_transform(imgs.view(batch_size, -1), sigma=sigma, alpha=alpha, height=img_shape[0], width=img_shape[1])
        imgs = torch.reshape(torch.tensor(imgs), imgs_shape)
        imgs = imgs.to(device)

        labels = batch[1].to(device)

        optimizer.zero_grad()        

        pred = model.forward(imgs)

        criterion = torch.nn.CrossEntropyLoss()
        class_loss = criterion(pred, labels)

        prototype_vectors = model.prototype_layer.prototype_vectors
        feature_vectors = model.feature_vectors

        r1_loss = torch.mean(torch.min(list_of_distances(prototype_vectors, feature_vectors.view(-1, model.input_dim_prototype)), dim=1)[0])
        r2_loss = torch.mean(torch.min(list_of_distances(feature_vectors.view(-1, model.input_dim_prototype ), prototype_vectors), dim=1)[0])
        
        rec = model.forward_dec(feature_vectors)
        ae_loss = torch.mean(list_of_norms(rec-imgs))

        loss = lambda_class * class_loss +\
                lambda_r1 * r1_loss +\
                lambda_r2 * r2_loss +\
                lambda_ae * ae_loss

        loss.backward()
        
        optimizer.step()

        # train accuracy
        max_vals, max_indices = torch.max(pred,1)
        n = max_indices.size(0)
        train_acc += (max_indices == labels).sum(dtype=torch.float32)/n
    
    train_acc /= max_iter

    print(f'epoch {e} - time {round(time.time()-start,2)} - loss {round(loss.item(),6)} - train accuracy {round(train_acc.item(),6)}')

    if (e+1) % save_step == 0 or e == training_epochs - 1:
        
        # save model states
        model_dir='results/states'
        makedirs(model_dir)

        state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ep': e
                }
        torch.save(state, os.path.join(model_dir, '%05d.pth' % (e)))

        # save results as images
        img_dir = 'results/imgs'
        makedirs(img_dir)

        # decode prototype vectors
        prototype_imgs = model.forward_dec(prototype_vectors.reshape((-1,10,2,2))).detach().cpu()

        # visualize the prototype images
        n_cols = 5
        n_rows = n_prototype_vectors // n_cols + 1 if n_prototype_vectors % n_cols != 0 else n_prototype_vectors // n_cols
        g, b = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
        for i in range(n_rows):
            for j in range(n_cols):
                if i*n_cols + j < n_prototype_vectors:
                    b[i][j].imshow(prototype_imgs[i*n_cols + j].reshape(img_shape[0], img_shape[1]),
                                    cmap='gray',
                                    interpolation='none')
                    b[i][j].axis('off')
                    
        plt.savefig(os.path.join(img_dir, 'prototype_result-' + str(e) + '.png'),
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        # apply encoding and decoding over a small subset of the training set
        imgs = []
        for batch in data_loader:
            imgs = batch[0].to(device)
            break

        examples_to_show = 10
        
        encoded = model.enc.forward(imgs[:examples_to_show])
        decoded = model.dec.forward(encoded)

        decoded = decoded.detach().cpu()
        imgs = imgs.detach().cpu()

        # compare original images to their reconstructions
        f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
        for i in range(examples_to_show):
            a[0][i].imshow(imgs[i].reshape(img_shape[0], img_shape[1]),
                            cmap='gray',
                            interpolation='none')
            a[0][i].axis('off')
            a[1][i].imshow(decoded[i].reshape(img_shape[0], img_shape[1]), 
                            cmap='gray',
                            interpolation='none')
            a[1][i].axis('off')
            
        plt.savefig(os.path.join(img_dir, 'decoding_result-' + str(e) + '.png'),
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        print(f'SAVED - epoch {e} - imgs @ {img_dir} - model @ {model_dir}')

    # test set accuracy evaluation
    if (e+1) % test_display_step == 0 or e == training_epochs - 1:
        max_iter = len(data_loader_test)
        test_acc = 0

        for i, batch in enumerate(data_loader_test):
            imgs = batch[0]
            imgs = imgs.to(device)
            labels = batch[1].to(device)

            pred = model.forward(imgs)

            # test accuracy
            max_vals, max_indices = torch.max(pred,1)
            n = max_indices.size(0)
            test_acc += (max_indices == labels).sum(dtype=torch.float32)/n

        test_acc /= max_iter

        print(f'TEST - epoch {e} - accuracy {round(test_acc.item(),6)}')