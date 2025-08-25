import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
from scipy.io import loadmat
import numpy as np
import time
import os
from utils.dataprocess import load_data, normalize, traintwo_patch, split_train_test_labels, padpatch, loadtrandte_data
from model.DAHGM import DAHGM
from utils.evulate import calculate_metrics, evaluatetwo
from utils.output import save_metrics_and_accuracies

# -------------------------------------------------------------------------------
# Configuration Parameters
parser = argparse.ArgumentParser("CNNMamba")
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
parser.add_argument('--test_freq', type=int, default=10, help='Evaluation frequency in epochs')
parser.add_argument('--epoches', type=int, default=200, help='Total training epochs')
parser.add_argument('--learning_rate', type=float, default=5e-4, choices=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3], help='Learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='Learning rate decay factor')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer')
parser.add_argument('--dataset', default='Trento', choices=['Muufl', 'Trento', 'Houston'], help='Dataset selection')
parser.add_argument('--num_classes', type=int, default=6, choices=[11, 6, 15], help='Number of classes')
parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
parser.add_argument('--train_num', type=int, default=50, help='Number of training samples per class')
parser.add_argument('--patches1', type=int, default=13, choices=[9, 11, 13, 15, 17, 19], help='Patch size for HSI data')
parser.add_argument('--more0', action='store_true', default=False, help='Flag for additional data processing')
args = parser.parse_args()

# seed = 0
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# random.seed(seed)
# np.random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)

def create_dataloader():
    """Create data loaders for training and testing"""
    # Load dataset based on configuration
    Data1, Data2, gt, train_labels, test_labels = load_data(args.dataset)

    # Convert and normalize data
    Data1 = normalize(Data1.astype(np.float32))
    Data2 = normalize(Data2.astype(np.float32))

    # Split_data_into_training and testing_sets
    train_labels, _ = split_train_test_labels(
        train_labels, args.train_num, args.dataset
    )

    # train_labels, test_labels = loadtrandte_data(args.dataset)
    # _,_,_,train_labels, test_labels = load_data(args.dataset)
    # Prepare patches for model input
    pad_width = args.patches1 // 2
    TrainPatch1, TrainPatch2, TrainLabel = traintwo_patch(
        Data1, Data2, args.patches1, pad_width, train_labels, args.more0
    )
    TestPatch1, TestPatch2, TestLabel = traintwo_patch(
        Data1, Data2, args.patches1, pad_width, test_labels, args.more0
    )

    # Create PyTorch datasets
    train_dataset = Data.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel)
    test_dataset = Data.TensorDataset(TestPatch1, TestPatch2, TestLabel)


    # Create data loaders
    train_loader = Data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = Data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True
    )

    return train_loader, test_loader


def train(model, optimizer, criterion, scheduler, train_loader, test_loader, device):
    """Model training procedure"""
    best_accuracy = 0.0
    best_model_path = os.path.join(args.dataset, "best_model.pth")
    train_losses = []

    for epoch in range(args.epoches):
        model.train()
        training_loss = 0

        # Batch training
        for hsi, lidar, labels in train_loader:
            hsi, lidar, labels = hsi.to(device), lidar.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(hsi, lidar)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        # Periodic evaluation
        if (epoch + 1) % args.test_freq == 0:
            accuracy = evaluatetwo(model, test_loader, device)
            avg_loss = training_loss / len(train_loader)
            train_losses.append(avg_loss)

            print(f"Epoch [{epoch + 1}/{args.epoches}], "
                  f"Loss: {avg_loss:.4f}, "
                  f"Test Accuracy: {accuracy:.2f}%")

            # Save best model
            if accuracy > best_accuracy:
                os.makedirs(args.dataset, exist_ok=True)
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

        scheduler.step()

    return train_losses, best_model_path


def test(model, test_loader, best_model_path, num_classes, device):
    """Model evaluation procedure"""
    model.load_state_dict(torch.load(best_model_path,weights_only=True))
    model.eval()

    total, correct = 0, 0
    confusion_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for hsi, lidar, labels in test_loader:
            hsi, lidar, labels = hsi.to(device), lidar.to(device), labels.to(device)

            outputs = model(hsi, lidar)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update confusion matrix
            for t, p in zip(labels, predicted):
                confusion_matrix[t.item(), p.item()] += 1

    # Calculate metrics
    OA, AA, Kappa, class_accuracy = calculate_metrics(
        confusion_matrix, total, correct
    )

    print(f"\nOverall Accuracy (OA): {OA:.2f}%")
    print(f"Average Accuracy (AA): {AA:.2f}%")
    print(f"Kappa Coefficient: {Kappa:.4f}")

    for i, acc in enumerate([acc * 100 for acc in class_accuracy]):
        print(f"Class {i + 1} Accuracy: {acc:.2f}%")

    return confusion_matrix, class_accuracy, OA, AA, Kappa


if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model initialization based on dataset
    dataset_config = {
        'Houston': (144, 1),
        'Trento': (63, 1),
        'Muufl': (64, 2)
    }
    band1, band2 = dataset_config[args.dataset]
    model = DAHGM(band1, band2, args.num_classes).to(device)

    # Data preparation
    train_loader, test_loader = create_dataloader()

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.epoches // 10,
        gamma=args.gamma
    )

    # Training process
    torch.cuda.synchronize()
    start_time = time.time()
    train_losses, best_model_path = train(
        model, optimizer, criterion, scheduler,
        train_loader, test_loader, device
    )
    torch.cuda.synchronize()
    train_time = time.time() - start_time

    # Evaluation process
    torch.cuda.synchronize()
    start_time = time.time()
    results = test(model, test_loader, best_model_path, args.num_classes, device)
    torch.cuda.synchronize()
    test_time = time.time() - start_time

    # Save results
    save_metrics_and_accuracies(
        results[1], results[2], results[3], results[4],
        train_time, test_time, args.dataset

    )
