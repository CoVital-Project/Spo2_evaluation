# Code adapted from: https://github.com/bhpfelix/PyTorch-Time-Series-Classification-Benchmarks/
# Added mobilenet_v2
import shutil, os, csv, itertools, glob
import sys

sys.path.append("./data_loader")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from preprocessing.data_loader import Spo2Dataset, spo2_collate_fn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
import pandas as pd
import pickle as pk
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from VGG import vgg13bn, vgg16bn, vgg19bn
from mobilenet_v2 import mobilenet_v2
cuda = torch.cuda.is_available()
# Utils


def load_pickle(filename):
    try:
        p = open(filename, "r")
    except IOError:
        print("Pickle file cannot be opened.")
        return None
    try:
        picklelicious = pk.load(p)
    except ValueError:
        print("load_pickle failed once, trying again")
        p.close()
        p = open(filename, "r")
        picklelicious = pk.load(p)

    p.close()
    return picklelicious


def save_pickle(data_object, filename):
    pickle_file = open(filename, "w")
    pk.dump(data_object, pickle_file)
    pickle_file.close()


def read_data(filename):
    print("Loading Data...")
    df = pd.read_csv(filename, header=None)
    data = df.values
    return data


def read_line(csvfile, line):
    with open(csvfile, "r") as f:
        data = next(itertools.islice(csv.reader(f), line, None))
    return data


#####################################################################################################################
#  Trainer and Core Experiment Scripts
#####################################################################################################################


def run_trainer(
    experiment_path,
    model_path,
    model,
    train_loader,
    test_loader,
    get_acc,
    resume,
    batch_size,
    num_epoch,
):

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    def save_checkpoint(state, is_best, filename=model_path + "checkpoint.pth.tar"):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, model_path + "model_best.pth.tar")

    def get_last_checkpoint(model_path):
        fs = sorted(
            [f for f in os.listdir(model_path) if "Epoch" in f],
            key=lambda k: int(k.split()[1]),
        )
        return model_path + fs[-1] if len(fs) > 0 else None

    start_epoch = 0
    best_res = 100000
    lrcurve = []
    conf_mats = []
    resume_state = get_last_checkpoint(model_path) if resume else None
    if resume_state and os.path.isfile(resume_state):
        print("=> loading checkpoint '{}'".format(resume_state))
        checkpoint = torch.load(resume_state)
        start_epoch = checkpoint["epoch"] + 1
        best_res = checkpoint["val_acc"]
        lrcurve = checkpoint["lrcurve"]
        conf_mats = checkpoint["conf_mats"]
        model.load_state_dict(checkpoint["state_dict"])
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                resume_state, checkpoint["epoch"]
            )
        )
    else:
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-5)

    criterion = nn.L1Loss()
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5) # optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    def train(epoch):
        model.train()
        total, total_correct = 0., 0.
        loss_list = []
        for batch_idx, (data, target, _) in enumerate(train_loader):
            # data, target = Variable(data.float()), Variable(target.long())
            if cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data.reshape(data.shape[0], 3, -1, 2))
            loss = criterion(output[:, 0], target[:, 0])
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data.item())
            metric, correct, num_instance, conf_mat = get_acc(
                output[:, 0], target[:, 0]
            )
            """if batch_idx % 2 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Metric: {:.2f}%'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item(), metric))"""
        print(
            "Average Loss: {:.6f} Metric: {:.2f}%".format(
                np.average(loss_list), np.average(metric)
            )
        )
        return metric

    def test():
        model.eval()
        test_loss = 0.
        total, total_correct = [], []
        preds = []
        for data, target, _ in test_loader:
            # data, target = Variable(data.float(), volatile=True), Variable(target.long())
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data.reshape(data.shape[0], 3, -1, 2))
            test_loss += criterion(output, target).data.item()  # sum up batch loss

            total_correct.extend(target[:, 0])
            total.extend(output[:, 0])
            preds.extend(output[:, 0])
        metric, correct, num_instance, conf_mat = get_acc(
            torch.stack(total), torch.stack(total_correct)
        )
        test_loss /= len(test_loader.dataset)
        print(
            "\nTest set: Average loss: {:.4f}, Metric: ({:.2f}%)\n".format(
                test_loss, metric
            )
        )

        return torch.stack(preds), metric

    for epoch in range(start_epoch, num_epoch):
        is_best = False

        train_acc = train(epoch)
        preds, val_acc = test()

        # print("Training Confmat: ")
        # print(train_conf)
        # print("Testing Confmat: ")
        # print(val_conf)
        # print("Number of Predictions Made: ")
        # print(preds.shape)

        lrcurve.append((train_acc, val_acc))
        # scheduler.step(val_loss)

        if val_acc < best_res:
            best_res = val_acc
            is_best = True
            save_checkpoint(
                {
                    "epoch": epoch,
                    "arch": model.arch,
                    "state_dict": model.cpu().state_dict(),
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "optimizer": optimizer.state_dict(),
                    "lrcurve": lrcurve,
                    "test_predictions": preds,
                },
                is_best,
                model_path + "Epoch %d Acc %.4f.pt" % (epoch, val_acc),
            )

        if cuda:
            model.cuda()

    return lrcurve


def run_experiment(
    experiment_path,
    data_path,
    model_root,
    models,
    norm,
    get_acc,
    resume=False,
    num_epoch=10,
):

    exp_result = {}
    for batch_size, model in models:
        print("Running %s" % model.arch)

        print("Loading Data..")
        nemcova_train_data = Spo2Dataset("nemcova_data/train")
        nemcova_test_data = Spo2Dataset("nemcova_data/test")
        covital_train_data = Spo2Dataset("covital-data/covital-data-community")
        covital_test_data = Spo2Dataset("covital-data/covital-data-clinical")

        train_data = ConcatDataset([nemcova_train_data,covital_train_data ])
        train_data = ConcatDataset([nemcova_test_data,covital_test_data ])

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=spo2_collate_fn,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=spo2_collate_fn,
        )

        model_path = (
            os.path.join(
                experiment_path, model_root, "norm" if norm else "nonorm", model.arch
            )
            + "/"
        )
        lrcurve = run_trainer(
            experiment_path,
            model_path,
            model,
            train_loader,
            test_loader,
            get_acc,
            resume,
            batch_size,
            num_epoch,
        )
        exp_result[model.arch] = {"lrcurve": lrcurve}

    return exp_result


def get_models():
    return [
        # (4, resnet18(num_classes=2)), # Learning rate recommended 1e-5
        # (6, vgg13bn(num_classes=2)), # Learning rate recommended 1e-4, it overfits quick
        (4, mobilenet_v2(False, True))  # Learning rate recommended 1e-4
    ]


def get_acc(output, target):
    # takes in two tensors to compute accuracy
    pred = output.data.max(1, keepdim=True)[
        1
    ]  # get the index of the max log-probability
    correct = pred.eq(target.data.view_as(pred)).cpu().sum()
    conf_mat = confusion_matrix(
        pred.cpu().numpy(), target.data.cpu().numpy(), labels=range(15)
    )
    return pred.cpu().numpy(), correct, target.size(0), conf_mat


def get_mae(output, target):
    # takes in two tensors to compute Mean Absolute Error
    mae = mean_absolute_error(
        target.cpu().detach().numpy(), output.cpu().detach().numpy()
    )
    return mae, output.cpu().detach().numpy(), target.size(0), None


experiments = {
    "experiment_path": "roll",
    "data_path": "roll/roll",
    "model_root": "model",
    "models": get_models(),
    "norm": False,
    "get_acc": get_mae,
    "resume": False,
    "num_epoch": 1000,
}
run_experiment(**experiments)
