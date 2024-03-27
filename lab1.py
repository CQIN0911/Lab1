import os
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

import seaborn as sns
from matplotlib.ticker import MaxNLocator

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn

def plot_accuracy(model1_acc, model2_acc, name):
    # TODO plot training and testing accuracy curve
    epochs = range(1, len(model1_acc) + 1)
    plt.plot(epochs, model1_acc, marker='.', linestyle='-')
    plt.plot(epochs, model2_acc, marker='.', linestyle='-')
    plt.title(f'{name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.xticks(epochs)
    plt.legend(['ResNet18', 'ResNet50'])
    plt.savefig(f"visualization/{name}_acc.png")
    plt.clf()

def plot_f1_score(f1_score1, f1_score2, name):
    # TODO plot testing f1 score curve
    epochs = range(1, len(f1_score1) + 1)
    plt.plot(epochs, f1_score1, marker='.', linestyle='-')
    plt.plot(epochs, f1_score2, marker='.', linestyle='-')
    plt.title(f'{name} F1 score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    # plt.xticks(epochs)
    plt.legend(['ResNet18', 'ResNet50'])
    plt.savefig(f"visualization/{name}_F1_score.png")
    plt.clf()

def plot_confusion_matrix(confusion_matrix, name):
    # TODO plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Normal', 'Predicted Pneumonia'],
                yticklabels=['Actual Normal', 'Actual Pneumonia'])
    plt.title(f'Confusion Matrix {name}')
    plt.savefig(f"visualization/{name}_confusion.png")

def train(device, train_loader, test_loader, model, criterion, optimizer, val_loader, name):
    best_acc = 0.0
    best_model_wts = None
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    f1_train = []
    f1_val = []
    f1_test = []
    best_c_matrix = []
    epoch_num = 0
    
    for epoch in range(1, args.num_epochs+1):
        epoch_num += 1
        with torch.set_grad_enabled(True):
            avg_loss = 0.0
            train_acc = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0     
            for _, data in enumerate(tqdm(train_loader)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                outputs = torch.max(outputs, 1).indices
                sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
                tp += sub_tp
                tn += sub_tn
                fp += sub_fp
                fn += sub_fn          

            avg_loss /= len(train_loader.dataset)
            train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
            print(f'Epoch: {epoch}')
            print(f'↳ Loss: {avg_loss}')
            print(f'↳ Training Acc.(%): {train_acc:.2f}%')
            f1_score = (2*tp) / (2*tp+fp+fn)
            f1_train.append(f1_score)

        train_acc_list.append(train_acc)

        # write validation if you needed
        val_acc, f1_score, _ = test(val_loader, model, "val")        
        f1_val.append(f1_score)
        val_acc_list.append(val_acc)

        test_acc, f1_score, c_matrix = test(test_loader, model, "test")
        test_acc_list.append(test_acc)
        f1_test.append(f1_score)

        if test_acc > best_acc:
            best_acc = test_acc
            best_c_matrix = c_matrix
            best_model_wts = model.state_dict()

        if epoch_num % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/{name}/model_weights{epoch_num}.pt')

    torch.save(model.state_dict(), f'checkpoints/{name}/last_model_weights.pt')
    torch.save(best_model_wts, f'checkpoints/{name}/best_model_weights.pt')

    return train_acc_list, val_acc_list, test_acc_list, f1_train, f1_val, f1_test, best_c_matrix

def test(test_loader, model, name):
    val_acc = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        c_matrix = [[int(tp), int(fn)],
                    [int(fp), int(tn)]]
        
        val_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score = (2*tp) / (2*tp+fp+fn)
        print (f'↳ Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        print (f'↳ {name} Acc.(%): {val_acc:.2f}%')

    return val_acc, f1_score, c_matrix

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=2)

    # for training
    parser.add_argument('--num_epochs', type=int, required=False, default=30)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.9)

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='chest_xray')

    # for data augmentation
    parser.add_argument('--degree', type=int, default=90)
    parser.add_argument('--resize', type=int, default=224)

    args = parser.parse_args()

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    # set dataloader (Train and Test dataset, write your own validation dataloader if needed.)
    train_dataset = ImageFolder(root=os.path.join(args.dataset, 'train'),
                                transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                                transforms.RandomRotation(args.degree), #, resample=False
                                                                transforms.ToTensor()]))
    test_dataset = ImageFolder(root=os.path.join(args.dataset, 'test'),
                               transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                               transforms.ToTensor()]))
    val_dataset = ImageFolder(root=os.path.join(args.dataset, 'val'),
                               transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                               transforms.ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    os.makedirs("checkpoints/ResNet18", exist_ok=True)
    resnet18 = models.resnet18(pretrained=True)
    num_neurons = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_neurons, args.num_classes)
    model_18 = resnet18.to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
    criterion = criterion.to(device)
    optimizer = optim.Adam(model_18.parameters(), lr=args.lr, weight_decay=args.wd)

    # model_path = "checkpoints6/ResNet18/model_weights30.pt"
    # checkpoint = torch.load(model_path)
    # model_18.load_state_dict(checkpoint)
    # model_18.eval()
    
    train_acc_list_18, val_acc_list_18, test_acc_list_18, f1_train_18, f1_val_18, f1_test_18, best_c_matrix_18 = train(device, train_loader, test_loader, model_18, criterion, optimizer, val_loader, "ResNet18")

    os.makedirs("checkpoints/ResNet50", exist_ok=True)
    resnet50 = models.resnet50(pretrained=True)
    num_neurons = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_neurons, args.num_classes)
    model_50 = resnet50.to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
    criterion = criterion.to(device)
    optimizer = optim.Adam(model_50.parameters(), lr=args.lr, weight_decay=args.wd)

    # model_path = "checkpoints6/ResNet50/model_weights30.pt"
    # checkpoint = torch.load(model_path)
    # model_50.load_state_dict(checkpoint)
    # model_50.eval()

    train_acc_list_50, val_acc_list_50, test_acc_list_50, f1_train_50, f1_val_50, f1_test_50, best_c_matrix_50 = train(device, train_loader, test_loader, model_50, criterion, optimizer, val_loader, "ResNet50")

    # plot
    # os.makedirs("visualization", exist_ok=True)
    # plot_confusion_matrix(c_matrix_18, "ResNet18")
    # plot_confusion_matrix(c_matrix_50, "ResNet50")

    plot_accuracy(train_acc_list_18, train_acc_list_50, "Train")
    plot_accuracy(val_acc_list_18, val_acc_list_50, "Val")
    plot_accuracy(test_acc_list_18, test_acc_list_50, "Test")
    plot_f1_score(f1_train_18, f1_train_50, "Train")
    plot_f1_score(f1_val_18, f1_val_50, "Val")
    plot_f1_score(f1_test_18, f1_test_50, "Test")
    plot_confusion_matrix(best_c_matrix_18, "ResNet18")
    plot_confusion_matrix(best_c_matrix_50, "ResNet50")