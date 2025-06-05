import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import VitNet_ssiat
from torch.distributions.multivariate_normal import MultivariateNormal
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from utils.loss import AngularPenaltySMLoss
import math
# tune the model at first session with adapter, and then conduct simplecil.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["backbone_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')
        self._network = VitNet_ssiat(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

        self._old_most_sentive = []
        self._update_grads = {}

        self.logit_norm = None
        self.tuned_epochs = None

        if "warm" in self.args and self.args["warm"] is True:
            self.proto_list = None
            self.warm_lr = args["warm_lr"]
            self.warm_epochs = args["warm_epochs"]
            self.warm_temp = args["warm_temp"]

    def after_task(self):
        self._known_classes = self._total_classes


    def extract_features(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(data_manager.get_task_size(self._cur_task))
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
    
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")

        self.train_dataset = train_dataset
        print("The number of training dataset:", len(self.train_dataset))

        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        train_dataset_for_protonet = data_manager.get_dataset(np.arange(0, self._total_classes), source="train",
                                                              mode="test")

        if self._known_classes > 0:
            test_dataset_old = data_manager.get_dataset(np.arange(0, self._known_classes), source="test", mode="test" )
            self.test_loader_old = DataLoader(test_dataset_old, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers)
        else:
            self.test_loader_old = None

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

      
        if self._cur_task >0:
            self._network.to(self._device)
            train_embeddings_old, _ = self.extract_features(self.train_loader, self._network, None)

        self._train(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

      
        if self._cur_task >0:
            train_embeddings_new, _ = self.extract_features(self.train_loader, self._network, None)
            old_class_mean = self._class_means[:self._known_classes]
            old_class_mean_copy=copy.deepcopy(old_class_mean)
            gap = self.displacement(train_embeddings_old, train_embeddings_new, old_class_mean, 4.0)
            if self.args['ssca'] is True:
                old_class_mean +=gap
                self._class_means[:self._known_classes] = old_class_mean

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        task_size = data_manager.get_task_size(self._cur_task)

        if self._cur_task>0 and self.args['ca_epochs']>0 and self.args['ca'] is True:
            self._stage2_compact_classifier(task_size, self.args['ca_epochs'])
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        
        if "warm" in self.args and self.args["warm"] is True:
            original_requires_grad = {p: p.requires_grad for p in self._network.parameters()}
            for p in self._network.parameters():
                p.requires_grad = True
            warm_optimizer = optim.SGD(
                    self._network.parameters(),
                    momentum=0.9,
                    lr=self.warm_lr,
                    weight_decay=self.weight_decay
                )
            scheduler = None
            self._train_warm(train_loader, self.test_loader_old, warm_optimizer, scheduler)
            
            for p in self._network.parameters():
                p.requires_grad = original_requires_grad[p]
        
        if self._cur_task == 0:
            self.tuned_epochs = self.args["init_epochs"]
            param_groups = [
                {'params': self._network.backbone.blocks[-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},

                {'params': self._network.backbone.blocks[:-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},

                {'params': self._network.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']}
            ]

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)

            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs,
                                                             eta_min=self.min_lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            
        else:
            self.tuned_epochs = self.args['inc_epochs']
            # show total parameters and trainable parameters
            param_groups = []

            param_groups.append(
                {'params': self._network.backbone.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']})
            param_groups.append(
                {'params': self._network.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']})


            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)

            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs,
                                                             eta_min=self.min_lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.tuned_epochs))
        loss_cos=AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss=loss_cos(logits[:, self._known_classes:], targets - self._known_classes)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )

            prog_bar.set_description(info)

        logging.info(info)


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
            # sims_dic = torch.einsum('ik,jk->ij', proto_selected, proto_selected)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                fake_targets = targets - self._known_classes

                embeddings = self._network.backbone(inputs)
                embeddings = F.normalize(embeddings, dim=1)
                logits_proto = torch.mm(embeddings, proto_selected.t())
                
                loss_clf = F.cross_entropy(logits_proto/self.warm_temp, fake_targets)

                # sims_proto = sims_dic[fake_targets]
                # loss_kdt = _KD_loss(
                #     logits_proto,
                #     sims_proto,
                #     self.warm_temp,
                # )
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            # scheduler.step()
            if test_loader:
                test_acc = self._compute_accuracy(self._network, test_loader)
            else:
                test_acc = 0.0
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Test_accy_old {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.warm_epochs,
                losses / len(train_loader),
                test_acc,
            )

            logging.info(info)

    def compute_irr_ratio(self):
        block_len = self._update_grads[self._cur_task]
        finetune_block = []
        ratio_list = []
        for block in self._update_grads[self._cur_task].keys():
            ratio = self._update_grads[self._cur_task][block] / self._update_grads[self._cur_task - 1][block]
            ratio_list.append(ratio)
            if ratio >= 0.9 and ratio <= 1.1:
                finetune_block.append(block)

        print("ratio", ratio_list)
        return block

    def cnt_match_block(self, old_blocks, new_blocks):
        finetune_block = []
        for nb in new_blocks:
            is_match = False
            for ob in old_blocks:
                if nb == ob:
                    is_match = True
                    break
            if is_match is False:
                finetune_block.append(nb)
        return finetune_block

    def compute_sentive(self):
        # eval module
        self._network.eval()
        sentive_network = copy.deepcopy(self._network)
        param_groups = [
            {'params': sentive_network.convnet.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']},
            {'params': sentive_network.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']}
        ]

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)

        update_magnitudes = {}
        for i, (_, inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = sentive_network(inputs)["logits"]
            loss = F.cross_entropy(logits[:, self._known_classes:], targets - self._known_classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for j, (name, param) in enumerate(sentive_network.named_parameters()):
                if "adapt" in name:
                    if name in update_magnitudes:
                        update_magnitudes[name] +=  (param.grad**2)# torch.norm(param.grad) / sum(param.shape)
                    else:
                        update_magnitudes[name] =  (param.grad**2)#torch.norm(param.grad) / sum(param.shape)
        grad_shapes = {}
        grad_shapes_int = {}
        for key in update_magnitudes.keys():
                grad_shapes[key] = update_magnitudes[key].shape
                grad_shapes_int[key] = np.cumprod(list(update_magnitudes[key].shape))[-1]
        # sort different block
        large_tensor = torch.cat([update_magnitudes[key].flatten() for key in grad_shapes.keys()])
        _, indexes = large_tensor.topk(math.ceil(0.0001* large_tensor.shape[0]))
        print(indexes)

        # Build up masks for unstructured tuning
        tmp_large_tensor = torch.zeros_like(large_tensor, device='cuda')
        tmp_large_tensor[indexes] = 1.

        tmp_large_tensor_list = tmp_large_tensor.split([shape for shape in grad_shapes_int.values()])

        structured_param_num = 0
        structured_names = []
        tuned_vectors = []

        unstructured_param_num = 0
        unstructured_name_shapes = {}
        unstructured_name_shapes_int = {}
        unstructured_grad_mask = {}
        grad_sum_dict = {}
        for i, key in enumerate(grad_shapes.keys()):
            grad_sum = tmp_large_tensor_list[i].view(grad_shapes[key]).sum()
            grad_sum_dict[key] = grad_sum
            cur_param_num = grad_sum.item()

            unstructured_param_num += grad_sum.item()
            unstructured_name_shapes[key] = tmp_large_tensor_list[i].view(grad_shapes[key]).shape
            unstructured_name_shapes_int[key] = np.cumprod(list(update_magnitudes[key].shape))[-1]
            unstructured_grad_mask[key] = tmp_large_tensor_list[i].view(grad_shapes[key])

        return unstructured_grad_mask #most_sentive
    
    def displacement(self, Y1, Y2, embedding_old, sigma):
        DY = Y2 - Y1
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1]) - np.tile(
            embedding_old[:, None, :], [1, Y1.shape[0], 1])) ** 2, axis=2)
        W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
        W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        displacement = np.sum(np.tile(W_norm[:, :, None], [
            1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement
    
    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        for p in self._network.fc.parameters():
            p.requires_grad = True

        run_epochs = ca_epochs
        crct_num = self._total_classes
        param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': self.init_lr,
                           'weight_decay': self.weight_decay}]

        optimizer = optim.SGD(network_params, lr=self.init_lr, momentum=0.9, weight_decay=self.weight_decay)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4], gamma=lrate_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self._network.to(self._device)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.eval()

        for epoch in range(run_epochs):
            losses = 0.
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256

            for c_id in range(crct_num):
                t_id = c_id // task_size
                decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64).to(self._device) * (
                            0.9 + decay)  # torch.from_numpy(self._class_means[c_id]).to(self._device)

                cls_cov = self._class_covs[c_id].to(self._device)
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)
                sampled_label.extend([c_id] * num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            inputs = sampled_data
            targets = sampled_label
            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in range(crct_num):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]

                # -stage two only use classifiers
                outputs = self._network.ca_forward(inp)
                logits = self.args['scale'] * outputs['logits']

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task + 1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]

                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)
                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, losses / self._total_classes, test_acc)
            logging.info(info)

    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self._known_classes
            new_class_means = np.zeros((self._total_classes, self.feature_dim))
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means
            # new_class_cov = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:
            self._class_means = np.zeros((self._total_classes, self.feature_dim))
            # self._class_covs = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

        radius = []
        for class_idx in range(self._known_classes, self._total_classes):

            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)

            # vectors = np.concatenate([vectors_aug, vectors])

            class_mean = np.mean(vectors, axis=0)
            if self._cur_task == 0:
                cov = np.cov(vectors.T)+ np.eye(class_mean.shape[-1]) * 1e-4
                radius.append(np.trace(cov) /768)
            # class_cov = np.cov(vectors.T)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-3

            self._class_means[class_idx, :] = class_mean
            self._class_covs[class_idx, ...] = class_cov

        if self._cur_task == 0:
                self.radius = np.sqrt(np.mean(radius))
                print(self.radius)
            # self._class_covs.append(class_cov)


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