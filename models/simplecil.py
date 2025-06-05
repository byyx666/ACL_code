import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy


num_workers = 8
batch_size = 128

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.args = args
        
        if "warm" in self.args and self.args["warm"] is True:
            self.proto_list = None
            self.warm_lr = args["warm_lr"]
            self.warm_epochs = args["warm_epochs"]
            self.warm_temp = args["warm_temp"]
            self.weight_decay = args["weight_decay"]

    def after_task(self):
        self._known_classes = self._total_classes
    
    def replace_fc(self,trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.backbone(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
        return model

   
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        self._network.to(self._device)
        if "warm" in self.args and self.args["warm"] is True:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.warm_lr,
                weight_decay=self.weight_decay,
            )
            scheduler = None
            self._train_warm(train_loader, None, optimizer, scheduler)
        self.replace_fc(train_loader_for_protonet, self._network, None)

    def _train_warm(self, train_loader, test_loader, optimizer, scheduler):
        self._network.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = self._network.backbone(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        if self.proto_list is None:
            assert self._known_classes == 0
            self.proto_list = torch.zeros(self._total_classes, embedding_list.size(1)).to(self._device)
        else:
            assert self._known_classes != 0
            new_proto_list = torch.zeros(self._total_classes - self._known_classes, embedding_list.size(1)).to(self._device)
            self.proto_list = torch.cat([self.proto_list, new_proto_list], dim=0)
        class_list = torch.unique(label_list)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self.proto_list[class_index] = proto
        print(proto.size())

        for epoch in range(self.warm_epochs):
            self._network.train()
            losses = 0.0

            proto_selected = self.proto_list[self._known_classes: self._total_classes]
            proto_selected = F.normalize(proto_selected, dim=1)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                fake_targets = targets - self._known_classes

                embeddings = self._network.backbone(inputs)
                embeddings = F.normalize(embeddings, dim=1)
                logits_proto = torch.mm(embeddings, proto_selected.t())
                
                loss_clf = F.cross_entropy(logits_proto/self.warm_temp, fake_targets)
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            # scheduler.step()

            info = "Task {}, Epoch {}/{} => Loss {:.3f}".format(
                self._cur_task,
                epoch + 1,
                self.warm_epochs,
                losses / len(train_loader),
            )
            logging.info(info)

    
    def _train_ft(self, train_loader, test_loader, optimizer, scheduler):
        for epoch in range(self.warm_epochs):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            # scheduler.step()

            info = "Task {}, Epoch {}/{} => Loss {:.3f}".format(
                self._cur_task,
                epoch + 1,
                self.warm_epochs,
                losses / len(train_loader),
            )
            logging.info(info)
   