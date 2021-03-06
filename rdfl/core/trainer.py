# Copyright (c) 2019 GalaxyLearning Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os
import copy
import time
import torch
import shutil
import logging
import requests
import importlib
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from inspect import isfunction
from rdfl.exceptions.fl_expection import GFLException
from rdfl.core.strategy import OptimizerStrategy, LossStrategy, SchedulerStrategy
from rdfl.utils.utils import LoggerFactory

JOB_PATH = os.path.join(os.path.abspath("."), "res", "jobs_client")
LOCAL_MODEL_BASE_PATH = os.path.join(os.path.abspath("."), "res", "models")
AGGREGATE_PATH = "tmp_aggregate_pars"
AGGREGATE_G_PATH = "tmp_aggregate_g_pars"
AGGREGATE_D_PATH = "tmp_aggregate_d_pars"
THRESHOLD = 0.5


class TrainStrategy(object):
    """
    TrainStrategy is the root class of all train strategy classes
    """

    def __init__(self, client_id):
        self.client_id = client_id
        self.fed_step = {}
        self.job_iter_dict = {}
        self.job_path = JOB_PATH

    def _parse_optimizer(self, optimizer, model, lr):
        if optimizer == OptimizerStrategy.OPTIM_SGD.value:
            return torch.optim.SGD(model.parameters(), lr, momentum=0.5)

    def _compute_loss(self, loss_function, output, label):
        """
        Return the loss according to the loss_function
        :param loss_function:
        :param output:
        :param label:
        :return:
        """

        if loss_function == LossStrategy.NLL_LOSS:
            loss = F.nll_loss(output, label)
        elif loss_function == LossStrategy.CE_LOSS:
            label = label.long()
            loss = F.cross_entropy(output, label)
        elif loss_function == LossStrategy.KLDIV_LOSS:
            loss = F.kl_div(output, label, reduction='batchmean')
        elif isfunction(loss_function):
            loss = loss_function(output, label)
        return loss

    def _compute_l2_dist(self, output, label):
        loss = F.mse_loss(output, label)
        return loss

    def _create_job_tmp_models_dir(self, client_id, job_id):
        """
        Create local temporary model directory according to client_id and job_id
        :param client_id:
        :param job_id:
        :return:
        """
        local_model_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id), "models_{}".format(client_id), "tmp_model_pars")
        if not os.path.exists(local_model_dir):
            os.makedirs(local_model_dir)
        return local_model_dir

    def _load_job_model(self, job_id, job_model_class_name):
        """
        Load model object according to job_id and model's class name
        :param job_id:
        :param job_model_class_name:
        :return:
        """
        module = importlib.import_module("res.models.models_{}.init_model_{}".format(job_id, job_id),
                                         "init_model_{}".format(job_id))
        model_class = getattr(module, job_model_class_name)
        return model_class()

    def _find_latest_aggregate_model_pars(self, job_id):
        """
        Return the latest aggregated model's parameters
        :param job_id:
        :return:
        """
        job_model_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id), "{}".format(AGGREGATE_PATH))
        if not os.path.exists(job_model_path):
            os.makedirs(job_model_path)
            init_model_pars_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                                "init_model_pars_{}".format(job_id))
            first_aggregate_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id), "tmp_aggregate_pars",
                                                "avg_pars_{}".format(0))
            if os.path.exists(init_model_pars_path):
                shutil.move(init_model_pars_path, first_aggregate_path)
        file_list = os.listdir(job_model_path)
        file_list = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(job_model_path, x)))
        
        if len(file_list) != 0:
            return os.path.join(job_model_path, file_list[-1]), len(file_list)
        return None, 0


class TrainNormalStrategy(TrainStrategy):
    """
    TrainNormalStrategy provides traditional training method and some necessary methods
    """

    def __init__(self, job, data, test_data, fed_step, client_id, local_epoch, model, curve, device):
        super(TrainNormalStrategy, self).__init__(client_id)
        self.job = job
        self.data = data
        self.test_data = test_data
        self.job_model_path = os.path.join(os.path.abspath("."), "models_{}".format(job.get_job_id()))
        self.fed_step = fed_step
        self.local_epoch = local_epoch
        self.accuracy_list = []
        self.loss_list = []
        self.model = model
        self.curve = curve
        self.device = device

    def train(self):
        pass

    def _test(self, global_model_pars):
        model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
        model.load_state_dict(global_model_pars)
        model.eval()
        device = self.device
        model = model.to(device)
        test_dataloader = torch.utils.data.DataLoader(self.test_data,
                                                           batch_size=self.model.get_train_strategy().get_batch_size(),
                                                           shuffle=True)
        with torch.no_grad():
            acc = 0
            for idx, (batch_data, batch_target) in enumerate(test_dataloader):
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                pred = model(batch_data)
                # log_pred = torch.log(F.softmax(pred, dim=1))
                loss = self._compute_loss(self.model.get_train_strategy().get_loss_function(), pred,
                                          batch_target)
                acc += torch.eq(pred.argmax(dim=1), batch_target).sum().float().item()
            self.logger.info(
                "test_loss: {}, test_accuracy:{}".format(loss.item(), float(acc) / float(len(test_dataloader.dataset))))
        model = model.to("cpu")

    def _train(self, train_model, job_models_path, fed_step, local_epoch):

        # TODO: transfer training code to c++ and invoked by python using pybind11

        step = 0
        accuracy = 0
        scheduler = None
        model = train_model.get_model()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = self.device
        # test_function = train_model.get_train_strategy().get_test_function()
        # if test_function is not None:
        #     test_accuracy, test_loss = test_function(model)
        #     self.logger.info("test_accuracy: {}, test_loss: {}".format(test_accuracy, test_loss))
        model = model.to(device)
        model.train()
        # if train_model.get_train_strategy().get_scheduler() is not None:
        #     scheduler = train_model.get_train_strategy().get_scheduler()
        while step < local_epoch:

            # if scheduler is not None:
            #     scheduler.step()
            acc = 0

            if train_model.get_train_strategy().get_optimizer() is not None:
                optimizer = self._generate_new_optimizer(model, train_model.get_train_strategy().get_optimizer())
            else:
                optimizer = self._generate_new_scheduler(model, train_model.get_train_strategy().get_scheduler())
            train_dataloader = torch.utils.data.DataLoader(self.data,
                                                                batch_size=self.model.get_train_strategy().get_batch_size(),
                                                                shuffle=True,
                                                                num_workers=0,
                                                                pin_memory=True)
            for idx, (batch_data, batch_target) in enumerate(train_dataloader):
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                pred = model(batch_data)
                # log_pred = torch.log(F.softmax(pred, dim=1))
                loss = self._compute_loss(train_model.get_train_strategy().get_loss_function(), pred, batch_target)
                batch_acc = torch.eq(pred.argmax(dim=1), batch_target).sum().float().item()
                acc += batch_acc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx % 200 == 0:
                    # accuracy_function = train_model.get_train_strategy().get_accuracy_function()
                    # if accuracy_function is not None and isfunction(accuracy_function):
                    #     accuracy = accuracy_function(pred, batch_target)
                    self.logger.info("train_loss: {}, train_acc: {}".format(loss.item(), float(batch_acc)/len(batch_target)))
            step += 1
            accuracy = acc / len(train_dataloader.dataset)


        torch.save(model.state_dict(),
                       os.path.join(job_models_path, "tmp_parameters_{}".format(fed_step)))

        return accuracy, loss.item()

    def _exec_finish_job(self, job_list):
        pass

    def _prepare_jobs_model(self, job_list):
        for job in job_list:
            self._prepare_job_model(job, None)

    def _prepare_job_model(self, job, server_url=None):
        job_model_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job.get_job_id()))
        job_init_model_path = os.path.join(job_model_path, "init_model_{}.py".format(job.get_job_id()))
        if server_url is None:
            # with open(job.get_train_model(), "r") as model_f:
            #     if not os.path.exists(job_init_model_path):
            #         f = open(job_init_model_path, "w")
            #         for line in model_f.readlines():
            #             f.write(line)
            #         f.close()
            if not os.path.exists(job_init_model_path):
                with open(job_init_model_path, "w") as model_f:
                    with open(job.get_train_model(), "r") as f:
                        for line in f.readlines():
                            model_f.write(line)
        else:
            if not os.path.exists(job_model_path):
                os.makedirs(job_model_path)
            if not os.path.exists(job_init_model_path):
                response = requests.get("/".join([server_url, "init_model", job.get_job_id()]))
                self._write_bfile_to_local(response, job_init_model_path)

    def _prepare_job_init_model_pars(self, job, server_url):
        job_init_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH,
                                               "models_{}".format(job.get_job_id()), "tmp_aggregate_pars")
        if not os.path.exists(job_init_model_pars_dir):
            os.makedirs(job_init_model_pars_dir)
        if len(os.listdir(job_init_model_pars_dir)) == 0:
            # print("/".join([server_url, "modelpars", job.get_job_id()]))
            response = requests.get("/".join([server_url, "modelpars", job.get_job_id()]))
            self._write_bfile_to_local(response, os.path.join(job_init_model_pars_dir, "avg_pars_0"))

    def _write_bfile_to_local(self, response, path):
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=512):
                if chunk:
                    f.write(chunk)

    def _prepare_upload_client_model_pars(self, job_id, client_id, fed_avg):
        job_init_model_pars_dir = os.path.join(os.path.abspath("."), LOCAL_MODEL_BASE_PATH,
                                               "models_{}".format(job_id), "models_{}".format(client_id))
        tmp_parameter_path = "tmp_parameters_{}".format(fed_avg)

        files = {
            'tmp_parameter_file': (
                'tmp_parameter_file', open(os.path.join(job_init_model_pars_dir, tmp_parameter_path), "rb"))
        }
        return files

    def _save_final_parameters(self, job_id, final_pars_path):
        file_path = os.path.join(os.path.abspath("."), "final_model_pars_{}".format(job_id))
        if os.path.exists(file_path):
            return
        with open(file_path, "wb") as w_f:
            with open(final_pars_path, "rb") as r_f:
                for line in r_f.readlines():
                    w_f.write(line)

    def _generate_new_optimizer(self, model, optimizer):
        state_dict = optimizer.state_dict()
        optimizer_class = optimizer.__class__
        params = state_dict['param_groups'][0]
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise GFLException("optimizer get wrong type value")

        if isinstance(optimizer, torch.optim.SGD):
            return optimizer_class(model.parameters(), lr=params['lr'], momentum=params['momentum'],
                                   dampening=params['dampening'], weight_decay=params['weight_decay'],
                                   nesterov=params['nesterov'])
        else:
            return optimizer_class(model.parameters(), lr=params['lr'], betas=params['betas'],
                                   eps=params['eps'], weight_decay=params['weight_decay'],
                                   amsgrad=params['amsgrad'])

    def _generate_new_scheduler(self, model, scheduler):
        scheduler_names = []
        for scheduler_item in SchedulerStrategy.__members__.items():
            scheduler_names.append(scheduler_item.value)
        if scheduler.__class__.__name__ not in scheduler_names:
            raise GFLException("optimizer get wrong type value")
        optimizer = scheduler.__getattribute__("optimizer")
        params = scheduler.state_dict()
        new_optimizer = self._generate_new_optimizer(model, optimizer)
        if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            return torch.optim.lr_scheduler.CyclicLR(new_optimizer, base_lr=params['base_lrs'],
                                                     max_lr=params['max_lrs'],
                                                     step_size_up=params['total_size'] * params['step_ratio'],
                                                     step_size_down=params['total_size'] - (params['total_size'] *
                                                                                            params['step_ratio']),
                                                     mode=params['mode'], gamma=params['gamma'],
                                                     scale_fn=params['scale_fn'], scale_mode=params['scale_mode'],
                                                     cycle_momentum=params['cycle_momentum'],
                                                     base_momentum=params['base_momentums'],
                                                     max_momentum=params['max_momentums'],
                                                     last_epoch=(-1 if params['last_epoch'] == 0 else params[
                                                         'last_epoch']))
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            return torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=params['T_max'],
                                                              eta_min=params['eta_min'],
                                                              last_epoch=(-1 if params['last_epoch'] == 0 else params[
                                                                  'last_epoch']))
        elif isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR):
            return torch.optim.lr_scheduler.ExponentialLR(new_optimizer, gamma=params['gamma'],
                                                          last_epoch=(-1 if params['last_epoch'] == 0 else params[
                                                              'last_epoch']))
        elif isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            return torch.optim.lr_scheduler.LambdaLR(new_optimizer, lr_lambda=params['lr_lamdas'],
                                                     last_epoch=(-1 if params['last_epoch'] == 0 else params[
                                                         'last_epoch']))
        elif isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR):
            return torch.optim.lr_scheduler.MultiStepLR(new_optimizer, milestones=params['milestones'],
                                                        gamma=params['gammas'],
                                                        last_epoch=(-1 if params['last_epoch'] == 0 else params[
                                                            'last_epoch']))
        elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(new_optimizer, mode=params['mode'],
                                                              factor=params['factor'], patience=params['patience'],
                                                              verbose=params['verbose'], threshold=params['threshold'],
                                                              threshold_mode=params['threshold_mode'],
                                                              cooldown=params['cooldown'], min_lr=params['min_lrs'],
                                                              eps=params['eps'])
        elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            return torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=params['step_size'], gamma=params['gamma'],
                                                   last_epoch=(-1 if params['last_epoch'] == 0 else params[
                                                       'last_epoch']))

    def _draw_curve(self):
        loss_x = range(0, self.job.get_epoch())
        accuracy_x = range(0, self.job.get_epoch())
        loss_y = self.loss_list
        accuracy_y = self.accuracy_list
        plt.subplot(2, 1, 1)
        plt.plot(loss_x, loss_y, ".-")
        plt.title("Train loss curve")
        plt.ylabel("Train loss")
        plt.xlabel("epoch")

        plt.subplot(2, 1, 2)
        plt.plot(accuracy_x, accuracy_y, "o-")
        plt.title("Train accuracy curve")
        plt.ylabel("Train accuracy")
        plt.xlabel("epoch")
        plt.show()


class TrainDistillationStrategy(TrainNormalStrategy):
    """
    TrainDistillationStrategy provides distillation training method and some necessary methods
    """

    def __init__(self, job, data, test_data, fed_step, client_id, local_epoch, models, curve, device):
        super(TrainDistillationStrategy, self).__init__(job, data, test_data, fed_step, client_id,local_epoch, models, curve, device)
        self.job_model_path = os.path.join(os.path.abspath("."), "res", "models", "models_{}".format(job.get_job_id()))
        # self.test_data = test_data

    def _load_other_models_pars(self, job_id, fed_step):
        """
        Load model's pars from other clients in fed_step round
        :param job_id:
        :param fed_step:
        :return:
        """
        job_model_base_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id))
        other_models_pars = []
        connected_clients_num = 0
        client_distillation_model_path = os.path.join(job_model_base_path, "models_{}".format(self.client_id), "distillation_model_pars")
        if len(os.listdir(client_distillation_model_path)) >= fed_step:
            return other_models_pars, 0
        for f in os.listdir(job_model_base_path):
            if f.find("models_") != -1 and int(f.split("_")[-1]) != int(self.client_id):
                connected_clients_num += 1
                files = os.listdir(os.path.join(job_model_base_path, f, "tmp_model_pars"))
                # files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(job_model_base_path, f, "tmp_model_pars", x)))
                # if len(files) == 0 or int(files[-1].split("_")[-1]) < fed_step:
                if len(files) == 0 or len(files) < fed_step:
                    return other_models_pars, 0
                else:
                    other_models_pars.append(os.path.join(job_model_base_path, f, "tmp_model_pars", files[-1])) #TODO: fix files[-1]
        time.sleep(0.5)
        for i in range(len(other_models_pars)):
            other_models_pars[i] = torch.load(other_models_pars[i])
        return other_models_pars, connected_clients_num+1

    def _calc_rate(self, received, total):
        """
        Calculate response rate of clients
        :param received:
        :param total:
        :return:
        """
        if total == 0:
            return 0
        return received / total

    def _train_with_distillation(self, train_model, other_models_pars, local_epoch, distillation_model_path, job_l2_dist):
        """
        Distillation training method
        :param train_model:
        :param other_models_pars:
        :param job_models_path:
        :return:
        """
        # TODO: transfer training code to c++ and invoked by python using pybind11
        step = 0
        scheduler = None
        device = self.device
        model = train_model.get_model()
        model, other_model = model.to(device), copy.deepcopy(model).to(device)
        model.train()
        if train_model.get_train_strategy().get_scheduler() is not None:
            scheduler = train_model.get_train_strategy().get_scheduler()
        while step < local_epoch:

            train_dataloader = torch.utils.data.DataLoader(self.data,
                                                           batch_size=self.model.get_train_strategy().get_batch_size(),
                                                           shuffle=True,
                                                           num_workers=0,
                                                           pin_memory=True)

            if scheduler is not None:
                scheduler.step()

            optimizer = self._generate_new_optimizer(model, train_model.get_train_strategy().get_optimizer())
            acc = 0
            for idx, (batch_data, batch_target) in enumerate(train_dataloader):
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)
                kl_pred = model(batch_data)
                pred = F.log_softmax(kl_pred, dim=1)
                acc += torch.eq(kl_pred.argmax(dim=1), batch_target).sum().float().item()
                loss_distillation = 0
                for other_model_pars in other_models_pars:
                    other_model.load_state_dict(other_model_pars)
                    other_model_kl_pred = other_model(batch_data).detach()

                    loss_distillation += self._compute_loss(LossStrategy.KLDIV_LOSS, F.log_softmax(kl_pred, dim=1),
                                                                F.softmax(other_model_kl_pred, dim=1))

                loss_s = self._compute_loss(train_model.get_train_strategy().get_loss_function(), kl_pred, batch_target)
                loss = loss_s + self.job.get_distillation_alpha() * loss_distillation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 200 == 0:
                    self.logger.info("distillation_loss: {}".format(loss.item()))
                #     self.logger.info("distillation_loss: {}".format(loss.item()))
            step += 1
            accuracy = acc / len(train_dataloader.dataset)

        torch.save(model.state_dict(),
                       os.path.join(distillation_model_path, "tmp_parameters_{}".format(self.fed_step[self.job.get_job_id()] + 1)))
        return accuracy, loss.item()


class TrainStandloneNormalStrategy(TrainNormalStrategy):
    """
    TrainStandloneNormalStrategy is responsible for controlling the process of traditional training in standalone mode
    """

    def __init__(self, job, data, test_data, fed_step, client_id, local_epoch, model, curve, device):
        super(TrainStandloneNormalStrategy, self).__init__(job, data, test_data,  fed_step, client_id, local_epoch, model, curve, device)
        self.logger = LoggerFactory.getLogger("TrainStandloneNormalStrategy", client_id, logging.INFO)

    def _create_job_models_dir(self, client_id, job_id):
        model_client_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                               "models_{}".format(client_id))
        if not os.path.exists(model_client_path):
            os.makedirs(model_client_path)
        return model_client_path

    def train(self):
        model_client_path = self._create_job_models_dir(self.client_id, self.job.get_job_id())
        while True:
            self.fed_step[self.job.get_job_id()] = 0 if self.fed_step.get(self.job.get_job_id()) is None else \
                self.fed_step.get(self.job.get_job_id())
            if self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) == self.job.get_epoch():
                self.logger.info("job_{} completed".format(self.job.get_job_id()))
                if self.curve is True:
                    self._draw_curve()
                break
            elif self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) > self.job.get_epoch():
                self.logger.warning("job_{} has completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                break
            aggregate_file, fed_step = self._find_latest_aggregate_model_pars(self.job.get_job_id())
            if aggregate_file is not None and self.fed_step.get(self.job.get_job_id()) != fed_step:
            #     if self.job.get_job_id() in runtime_config.EXEC_JOB_LIST:
            #         runtime_config.EXEC_JOB_LIST.remove(self.job.get_job_id())
            #     self.fed_step[self.job.get_job_id()] = fed_step
            # if self.job.get_job_id() not in runtime_config.EXEC_JOB_LIST:
                # job_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
                # if aggregate_file is not None:
                self.logger.info("load {} parameters".format(aggregate_file))
                new_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
                time.sleep(0.5)
                model_pars = torch.load(aggregate_file)
                new_model.load_state_dict(model_pars)
                self._test(model_pars)
                self.model.set_model(new_model)
                self.logger.info("job_{} is training, Aggregator strategy: {}".format(self.job.get_job_id(),
                                                                                      self.job.get_aggregate_strategy()))
                # runtime_config.EXEC_JOB_LIST.append(self.job.get_job_id())
                self.acc, loss = self._train(self.model, model_client_path, self.fed_step.get(self.job.get_job_id()), self.local_epoch)

                self.fed_step[self.job.get_job_id()] = fed_step
                # self.logger.info("job_{} {}th train accuracy: {}".format(self.job.get_job_id(),
                #                                                          self.fed_step.get(self.job.get_job_id()),
                #                                                          self.acc))


class TrainStandloneGANFedAvgStrategy(TrainStandloneNormalStrategy):
    """
    TrainStandloneNormalStrategy is responsible for controlling the process of traditional training in standalone mode
    """

    def __init__(self, job, data, test_data, fed_step, client_id, local_epoch, g_model, d_model, curve, device):
        super(TrainStandloneGANFedAvgStrategy, self).__init__(job, data, test_data,  fed_step, client_id, local_epoch, None, curve, device)
        self.g_model = g_model
        self.d_model = d_model
        self.logger = LoggerFactory.getLogger("TrainStandloneGANFedAvgStrategy", client_id, logging.INFO)

    def _load_job_gan_model(self, job_id, flag, job_model_class_name):

        if flag == "G":
            module = importlib.import_module("res.models.models_{}.init_g_model_{}".format(job_id, job_id),
                                            "init_g_model_{}".format(job_id))
        else:
            module = importlib.import_module("res.models.models_{}.init_d_model_{}".format(job_id, job_id),
                                             "init_d_model_{}".format(job_id))
        model_class = getattr(module, job_model_class_name)
        return model_class()

    def _find_latest_gan_aggregate_model_pars(self, aggregate_g_path, aggregate_d_path, job_id):

        # job_model_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id), "{}".format(AGGREGATE_PATH))
        if not os.path.exists(aggregate_g_path) and not os.path.exists(aggregate_d_path):
            os.makedirs(aggregate_g_path)
            os.makedirs(aggregate_d_path)
            init_g_model_pars_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                                "init_g_model_pars_{}".format(job_id))
            init_d_model_pars_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                                  "init_d_model_pars_{}".format(job_id))
            first_g_aggregate_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id), "tmp_aggregate_g_pars",
                                                "avg_pars_{}".format(0))
            first_d_aggregate_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                                  "tmp_aggregate_d_pars",
                                                  "avg_pars_{}".format(0))

            if os.path.exists(init_g_model_pars_path) and os.path.exists(init_d_model_pars_path):
                shutil.move(init_g_model_pars_path, first_g_aggregate_path)
                shutil.move(init_d_model_pars_path, first_d_aggregate_path)
        g_file_list, d_file_list = os.listdir(aggregate_g_path), os.listdir(aggregate_d_path)
        # file_list = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(job_model_path, x)))
        aggregate_g_file, aggregate_d_file = "", ""
        for file in g_file_list:
            if file.split("_")[-1] == str(len(g_file_list)-1):
                aggregate_g_file = file
                break
        for file in d_file_list:
            if file.split("_")[-1] == str(len(d_file_list)-1):
                aggregate_d_file = file
                break

        if len(g_file_list) != 0 and len(d_file_list) != 0 and aggregate_g_file != "" and aggregate_d_file != "":
            return os.path.join(aggregate_g_path, aggregate_g_file), os.path.join(aggregate_d_path, aggregate_d_file),  len(g_file_list), len(d_file_list)
        return None, None, 0, 0

    def _get_gan_optimizer(self, model, train_model):
        if train_model.get_train_strategy().get_optimizer() is not None:
            optimizer = self._generate_new_optimizer(model, train_model.get_train_strategy().get_optimizer())
        else:
            optimizer = self._generate_new_scheduler(model, train_model.get_train_strategy().get_scheduler())
        return optimizer


    def _create_local_gan_model_dir(self, client_id, job_id):
        model_client_path = self._create_job_models_dir(client_id, job_id)
        job_models_G_path, job_models_D_path = os.path.join(model_client_path, "tmp_G_models"), os.path.join(model_client_path, "tmp_D_models")
        if not os.path.exists(job_models_G_path):
            os.mkdir(job_models_G_path)
        if not os.path.exists(job_models_D_path):
            os.mkdir(job_models_D_path)
        return job_models_G_path, job_models_D_path

    def _save_generated_img(self, client_id, job_id, fed_step, fake_imgs):
        generated_imgs_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                               "models_{}".format(client_id), "generated_imgs")
        if not os.path.exists(generated_imgs_path):
            os.mkdir(generated_imgs_path)
        save_image(fake_imgs, os.path.join(generated_imgs_path, "img_{}.jpg".format(fed_step)))


    def _calc_gradient_penalty(self, netD, real_data, fake_data):
        # print "real_data: ", real_data.size(), fake_data.size()
        # alpha = torch.rand(real_data.shape[0], 1)
        # alpha = alpha.expand(real_data.shape[0], int(real_data.nelement()/real_data.shape[0])).contiguous().view(real_data.shape[0], 3, 32, 32)
        device = self.device
        alpha = torch.FloatTensor(np.random.random((real_data.size(0), 1, 1, 1)))
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def _train_gan(self, train_g_model, train_d_model, job_models_G_path, job_models_D_path, fed_step, local_epoch):

        step = 0
        g_model, d_model = train_g_model.get_model(), train_d_model.get_model()
        device = self.device
        g_model, d_model = g_model.to(device), d_model.to(device)
        g_model.train()
        d_model.train()
        # if train_model.get_train_strategy().get_scheduler() is not None:
        #     scheduler = train_model.get_train_strategy().get_scheduler()
        while step < local_epoch:
            # if scheduler is not None:
            #     scheduler.step()
            acc = 0
            optimizer_G = self._get_gan_optimizer(g_model, train_g_model)
            optimizer_D = self._get_gan_optimizer(d_model, train_d_model)

            train_dataloader = torch.utils.data.DataLoader(self.data,
                                                           batch_size=self.g_model.get_train_strategy().get_batch_size(),
                                                           shuffle=True,
                                                           num_workers=0,
                                                           pin_memory=True)
            for idx, (batch_data, batch_target) in enumerate(train_dataloader):
                batch_imgs, batch_target = batch_data.to(device), batch_target.to(device)
                z = torch.randn(batch_imgs.shape[0], 100)
                z = z.view(-1, 100, 1, 1)
                z = z.to(device)
                fake_imgs = g_model(z)
                real_validity = d_model(batch_imgs)
                fake_validity = d_model(fake_imgs)
                gradient_penalty = self._calc_gradient_penalty(d_model, batch_imgs, fake_imgs)
                d_loss = torch.mean(fake_validity) - torch.mean(real_validity) + gradient_penalty
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                if idx % 5 == 0:
                    fake_imgs = g_model(z)
                    fake_validity = d_model(fake_imgs)
                    g_loss = -torch.mean(fake_validity)
                    optimizer_G.zero_grad()
                    g_loss.backward()
                    optimizer_G.step()
                    self.logger.info(
                        "train_D_loss: {}, train_G_loss: {}".format(d_loss, g_loss))
            step += 1
        self._save_generated_img(self.client_id, self.job.get_job_id(), fed_step, fake_imgs)
        torch.save(g_model.state_dict(),
                   os.path.join(job_models_G_path, "tmp_G_parameters_{}".format(fed_step)))
        torch.save(d_model.state_dict(),
                   os.path.join(job_models_D_path, "tmp_D_parameters_{}".format(fed_step)))

        return d_loss, g_loss

    def train(self):

        job_models_G_path, job_models_D_path = self._create_local_gan_model_dir(self.client_id, self.job.get_job_id())

        while True:
            self.fed_step[self.job.get_job_id()] = 0 if self.fed_step.get(self.job.get_job_id()) is None else \
                self.fed_step.get(self.job.get_job_id())
            if self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) == self.job.get_epoch():
                self.logger.info("job_{} completed".format(self.job.get_job_id()))
                break
            elif self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) > self.job.get_epoch():
                self.logger.warning("job_{} has completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                break
            aggregate_g_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(self.job.get_job_id()),
                                          "{}".format(AGGREGATE_G_PATH))
            aggregate_d_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(self.job.get_job_id()),
                                            "{}".format(AGGREGATE_D_PATH))
            aggregate_g_file, aggregate_d_file, g_fed_step, d_fed_step = self._find_latest_gan_aggregate_model_pars(aggregate_g_path, aggregate_d_path, self.job.get_job_id())

            # aggregate_d_file, fed_step = self._find_latest_gan_aggregate_model_pars(aggregate_d_path, self.job.get_job_id())
            if aggregate_g_file is not None and aggregate_d_file is not None and self.fed_step.get(self.job.get_job_id()) != g_fed_step and g_fed_step == d_fed_step:
                self.logger.info("load g {} parameters, d {} parameters".format(aggregate_g_file, aggregate_d_file))
                new_g_model = self._load_job_gan_model(self.job.get_job_id(), "G", self.job.get_train_g_model_class_name())
                new_d_model = self._load_job_gan_model(self.job.get_job_id(), "D", self.job.get_train_d_model_class_name())
                time.sleep(0.5)
                model_g_pars = torch.load(aggregate_g_file)
                model_d_pars = torch.load(aggregate_d_file)
                new_g_model.load_state_dict(model_g_pars)
                new_d_model.load_state_dict(model_d_pars)
                # self._test(model_g_pars)
                self.g_model.set_model(new_g_model)
                self.d_model.set_model(new_d_model)
                self.logger.info("job_{} is training, Aggregator strategy: {}".format(self.job.get_job_id(),
                                                                                      self.job.get_aggregate_strategy()))
                d_loss, g_loss = self._train_gan(self.g_model, self.d_model, job_models_G_path, job_models_D_path, self.fed_step.get(self.job.get_job_id()), self.local_epoch)

                self.fed_step[self.job.get_job_id()] = g_fed_step




class TrainStandloneDistillationStrategy(TrainDistillationStrategy):
    """
    TrainStandloneDistillationStrategy is responsible for controlling the process of distillation training in standalone mode
    """

    def __init__(self, job, data, test_data, fed_step, client_id, local_epoch, model, curve, device):
        super(TrainStandloneDistillationStrategy, self).__init__(job, data, test_data, fed_step, client_id, local_epoch, model, curve, device)
        # self.train_model = self._load_job_model(job.get_job_id(), job.get_train_model_class_name())
        self.train_model = model
        self.logger = LoggerFactory.getLogger("TrainStandloneDistillationStrategy", -1, logging.INFO)

    def _create_dislillation_model_pars_path(self, client_id, job_id):
        distillation_model_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id), "models_{}".format(client_id),
                                       "distillation_model_pars")
        if not os.path.exists(distillation_model_path):
            os.makedirs(distillation_model_path)
        return distillation_model_path


    def _init_global_model(self, job_id, fed_step):
        init_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                           "global_models")
        if not os.path.exists(init_model_pars_dir):
            os.makedirs(init_model_pars_dir)
        init_global_model_pars_path = os.path.join(init_model_pars_dir, "global_parameters_{}".format(fed_step))
        if not os.path.exists(init_global_model_pars_path):
            new_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
            torch.save(new_model.state_dict(), init_global_model_pars_path)



    def _load_local_fed_step(self, job_tmp_models_path):
        return len(os.listdir(job_tmp_models_path))

    def _load_global_model(self, job_id, fed_step):
        global_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                             "global_models")
        global_model_path = os.path.join(global_model_pars_dir, "global_parameters_{}".format(fed_step))
        self.logger.info("load {} parameters".format(global_model_path))
        if not os.path.exists(global_model_path):
            return None
        new_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
        time.sleep(0.5)
        model_pars = torch.load(global_model_path)
        new_model.load_state_dict(model_pars)
        return new_model

    def _save_global_model(self, job_id, fed_step, global_model_pars):
        global_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                             "global_models")
        global_model_pars_path = os.path.join(global_model_pars_dir, "global_parameters_{}".format(fed_step))
        torch.save(global_model_pars, global_model_pars_path)


    def _load_distillation_model(self, distillation_path):
        new_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
        model_pars = torch.load(distillation_path)
        new_model.load_state_dict(model_pars)
        return new_model


    def _could_fed_avg(self, job_id, fed_step):
        distillation_model_pars = []
        job_model_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id))
        for model_dir in os.listdir(job_model_dir):
            if model_dir.find("models_") != -1:
                distillation_dir = os.path.join(job_model_dir, model_dir, "distillation_model_pars")
                file_list = os.listdir(distillation_dir)
                file_list = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(distillation_dir, x)))
                if len(file_list) == 0 or int(file_list[-1].split("_")[-1]) != fed_step:
                    return False, []
                else:
                    distillation_model_pars.append(os.path.join(distillation_dir, file_list[-1]))
        time.sleep(1)
        return True, distillation_model_pars

    def _calc_kl_loss(self, last_global_model, distillation_model_list):

        device = self.device

        num_batch = 0
        last_global_model = last_global_model.to(device)
        train_dataloader = torch.utils.data.DataLoader(self.data,
                                                       batch_size=self.model.get_train_strategy().get_batch_size(),
                                                       shuffle=True,
                                                       num_workers=0,
                                                       pin_memory=True)
        with torch.no_grad():
            for i in range(len(distillation_model_list)):
                distillation_model_list[i] = distillation_model_list[i].to(device)
            loss_kl_list = [0 for _ in range(len(distillation_model_list))]
            for idx, (batch_data, batch_target) in enumerate(train_dataloader):
                batch_data = batch_data.to(device)
                kl_pred = last_global_model(batch_data)
                for i in range(len(distillation_model_list)):
                    other_model_kl_pred = distillation_model_list[i](batch_data)
                    loss_kl_list[i] += self._compute_loss(LossStrategy.KLDIV_LOSS, F.log_softmax(kl_pred, dim=1),
                                                                F.softmax(other_model_kl_pred, dim=1)).item()

                num_batch += 1
        sum_kl_loss = 0
        print("kl_list:  ", loss_kl_list)
        print("num_batch: ", num_batch)
        for i in range(len(loss_kl_list)):
            loss_kl_list[i] /= num_batch
            sum_kl_loss += loss_kl_list[i]
        return loss_kl_list, sum_kl_loss


    def _calc_model_pars_weight(self, kl_list, sum_kl_loss):
        weight_list = []
        n = len(kl_list)

        for kl_loss in kl_list:
            weight_list.append(float(kl_loss)/float(sum_kl_loss))
        return weight_list


    def _fed_avg_aggregate(self, disillation_model_pars_list, weight_list, job_id, fed_step):
        avg_model_par = disillation_model_pars_list[0]
        for key in avg_model_par.keys():
            for i in range(1, len(disillation_model_pars_list)):
                # avg_model_par[key] += weight_list[i]*disillation_model_pars_list[i][key]
                avg_model_par[key] += disillation_model_pars_list[i][key]
            avg_model_par[key] = torch.div(avg_model_par[key], len(disillation_model_pars_list))
        self._test(avg_model_par)
        self._save_global_model(job_id, fed_step, avg_model_par)

    def _execute_fed_avg(self, client_id, job_id, fed_step, distillation_model_pars_file_list):
        self.logger.info("client {} execute FedAvg".format(client_id))
        last_global_model = self._load_global_model(job_id, fed_step-1)
        distillation_model_list = []
        for distillation_model_pars_file in distillation_model_pars_file_list:
            distillation_model = self._load_distillation_model(distillation_model_pars_file)
            distillation_model_list.append(distillation_model)
        # kl_list, sum_kl_loss = self._calc_kl_loss(last_global_model, distillation_model_list)
        distillation_model_pars_list = [distillation_model.state_dict() for distillation_model in distillation_model_list]
        # weight_list = self._calc_model_pars_weight(kl_list, sum_kl_loss)
        # print("weight_list: ", weight_list)
        self._fed_avg_aggregate(distillation_model_pars_list, [], job_id, fed_step)

    def _get_fed_step(self, job_id):
        global_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                             "global_models")
        if not os.path.exists(global_model_pars_dir):
            return 0
        return len(os.listdir(global_model_pars_dir))-1


    def train(self):
        distillation_model_path = self._create_dislillation_model_pars_path(self.client_id, self.job.get_job_id())
        job_tmp_models_path = self._create_job_tmp_models_dir(self.client_id, self.job.get_job_id())
        self._init_global_model(self.job.get_job_id(), 0)
        while True:
            self.fed_step[self.job.get_job_id()] = self._get_fed_step(self.job.get_job_id())
            # self.fed_step[self.job.get_job_id()] = 0 if self.fed_step.get(
            #     self.job.get_job_id()) is None else self.fed_step.get(self.job.get_job_id())
            # print("test_iter_num: ", self.job_iter_dict[self.job.get_job_id()])
            if self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) >= self.job.get_epoch():
                final_pars_path = os.path.join(self.job_model_path, "models_{}".format(self.client_id),
                                               "tmp_parameters_{}".format(self.fed_step.get(self.job.get_job_id())))
                if os.path.exists(final_pars_path):
                    self._save_final_parameters(self.job.get_job_id(), final_pars_path)
                    self.logger.info("job_{} completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                if self.curve is True:
                    self._draw_curve()
                break

            # aggregate_file, _ = self._find_latest_aggregate_model_pars(self.job.get_job_id())

            local_fed_step = self._load_local_fed_step(job_tmp_models_path)
            if(local_fed_step < self.fed_step[self.job.get_job_id()]+1):
                # aggregate_file = self._find_latest_global_model_pars(self.job.get_job_id())
                global_model = self._load_global_model(self.job.get_job_id(), self.fed_step[self.job.get_job_id()])
                if global_model is not None:
                    self.model.set_model(global_model)
                    self._train(self.model, job_tmp_models_path, self.fed_step[self.job.get_job_id()]+1, self.local_epoch)
            other_model_pars, connected_clients_num = self._load_other_models_pars(self.job.get_job_id(),
                                                                                   self.fed_step[self.job.get_job_id()]+1)
            # job_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
            # self.logger.info("job_{} is training, Aggregator strategy: {}, L2_dist: {}".format(self.job.get_job_id(),
            #                                                                                    self.job.get_aggregate_strategy(),
            #                                                                                    self.job.get_l2_dist()))
            if other_model_pars is not None and connected_clients_num:

                self.logger.info("model distillating....")


                self.acc, loss = self._train_with_distillation(self.model, other_model_pars, self.local_epoch, distillation_model_path,
                                                               self.job.get_l2_dist())
                # self.accuracy_list.append(self.acc)
                # self.loss_list.append(loss)
                self.logger.info("model distillation success")

                if(int(self.client_id) == (self.fed_step[self.job.get_job_id()] % connected_clients_num)):
                    # print(self.client_id, self.fed_step[self.job.get_job_id()] % connected_clients_num)
                    while True:
                        is_fed_avg, distillation_model_pars = self._could_fed_avg(self.job.get_job_id(), self.fed_step[self.job.get_job_id()]+1)
                        print("could fed_avg: {}".format(is_fed_avg))
                        if is_fed_avg:
                            self._execute_fed_avg(self.client_id, self.job.get_job_id(), self.fed_step[self.job.get_job_id()]+1, distillation_model_pars)
                            break
                        time.sleep(2)

                self.fed_step[self.job.get_job_id()] = self.fed_step.get(self.job.get_job_id()) + 1


class TrainStandloneGANDistillationStrategy(TrainStandloneDistillationStrategy):


    def __init__(self, job, data, test_data, fed_step, client_id, local_epoch, g_model, d_model, curve, device):
        super(TrainStandloneGANDistillationStrategy, self).__init__(job, data, test_data, fed_step, client_id, local_epoch, None, curve, device)
        # self.train_model = self._load_job_model(job.get_job_id(), job.get_train_model_class_name())
        self.train_g_model = g_model
        self.train_d_model = d_model
        self.logger = LoggerFactory.getLogger("TrainStandloneGANDistillationStrategy", -1, logging.INFO)

    def _get_gan_optimizer(self, model, train_model):
        if train_model.get_train_strategy().get_optimizer() is not None:
            optimizer = self._generate_new_optimizer(model, train_model.get_train_strategy().get_optimizer())
        else:
            optimizer = self._generate_new_scheduler(model, train_model.get_train_strategy().get_scheduler())
        return optimizer

    def _save_global_generated_img(self, client_id, job_id, fed_step):
        g_global_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                               "g_global_models")
        g_global_model_path = os.path.join(g_global_model_pars_dir, "global_parameters_{}".format(fed_step))
        new_g_model = self._load_job_gan_model(self.job.get_job_id(), "G", self.job.get_train_g_model_class_name())
        time.sleep(0.5)
        g_model_pars = torch.load(g_global_model_path)
        new_g_model.load_state_dict(g_model_pars)
        new_g_model = new_g_model.to(self.device)
        generated_imgs_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                           "models_{}".format(client_id), "global_generated_imgs")
        if not os.path.exists(generated_imgs_path):
            os.mkdir(generated_imgs_path)

        z = torch.randn(32, 100)
        z = z.view(-1, 100, 1, 1)
        z = z.to(self.device)
        global_fake_imgs = new_g_model(z)

        save_image(global_fake_imgs, os.path.join(generated_imgs_path, "img_{}.jpg".format(fed_step)))

    def _save_generated_img(self, client_id, job_id, fed_step, fake_imgs):
        generated_imgs_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                           "models_{}".format(client_id), "generated_imgs")
        if not os.path.exists(generated_imgs_path):
            os.mkdir(generated_imgs_path)
        save_image(fake_imgs, os.path.join(generated_imgs_path, "img_{}.jpg".format(fed_step)))

    def _calc_gradient_penalty(self, netD, real_data, fake_data):
        device = self.device
        alpha = torch.FloatTensor(np.random.random((real_data.size(0), 1, 1, 1)))
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)


        disc_interpolates = netD(interpolates)


        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def _create_dislillation_gan_model_pars_path(self, client_id, job_id):
        distillation_g_model_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id), "models_{}".format(client_id),
                                       "distillation_g_model_pars")
        distillation_d_model_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                                 "models_{}".format(client_id),
                                                 "distillation_d_model_pars")
        if not os.path.exists(distillation_g_model_path):
            os.makedirs(distillation_g_model_path)
        if not os.path.exists(distillation_d_model_path):
            os.makedirs(distillation_d_model_path)
        return distillation_g_model_path, distillation_d_model_path

    def _create_job_gan_tmp_models_dir(self, client_id, job_id):

        local_g_model_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id), "models_{}".format(client_id), "tmp_g_model_pars")
        local_d_model_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id), "models_{}".format(client_id),
                                       "tmp_d_model_pars")
        if not os.path.exists(local_g_model_dir):
            os.makedirs(local_g_model_dir)
        if not os.path.exists(local_d_model_dir):
            os.makedirs(local_d_model_dir)
        return local_g_model_dir, local_d_model_dir

    def _load_job_gan_model(self, job_id, flag, job_model_class_name):

        if flag == "G":
            module = importlib.import_module("res.models.models_{}.init_g_model_{}".format(job_id, job_id),
                                             "init_g_model_{}".format(job_id))
        else:
            module = importlib.import_module("res.models.models_{}.init_d_model_{}".format(job_id, job_id),
                                             "init_d_model_{}".format(job_id))
        model_class = getattr(module, job_model_class_name)
        return model_class()

    def _init_gan_global_model(self, job_id, fed_step):
        init_g_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                           "g_global_models")
        init_d_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                             "d_global_models")
        if not os.path.exists(init_g_model_pars_dir):
            os.makedirs(init_g_model_pars_dir)
        if not os.path.exists(init_d_model_pars_dir):
            os.makedirs(init_d_model_pars_dir)
        init_g_global_model_pars_path = os.path.join(init_g_model_pars_dir, "global_parameters_{}".format(fed_step))
        init_d_global_model_pars_path = os.path.join(init_d_model_pars_dir, "global_parameters_{}".format(fed_step))
        if not os.path.exists(init_g_global_model_pars_path):
            new_model = self._load_job_gan_model(self.job.get_job_id(), "G", self.job.get_train_g_model_class_name())
            torch.save(new_model.state_dict(), init_g_global_model_pars_path)
        if not os.path.exists(init_d_global_model_pars_path):
            new_model = self._load_job_gan_model(self.job.get_job_id(), "D", self.job.get_train_d_model_class_name())
            torch.save(new_model.state_dict(), init_d_global_model_pars_path)


    def _load_gan_global_model(self, job_id, fed_step):
        g_global_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                             "g_global_models")
        g_global_model_path = os.path.join(g_global_model_pars_dir, "global_parameters_{}".format(fed_step))

        d_global_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                               "d_global_models")
        d_global_model_path = os.path.join(d_global_model_pars_dir, "global_parameters_{}".format(fed_step))

        # self.logger.info("load g parameters {}, load d parameters {}".format(g_global_model_path, d_global_model_path))
        if not os.path.exists(g_global_model_path) or not os.path.exists(d_global_model_path):
            return None, None
        self.logger.info("load g {} parameters, load f {} parameters".format(g_global_model_path, d_global_model_path))
        new_g_model = self._load_job_gan_model(self.job.get_job_id(), "G", self.job.get_train_g_model_class_name())
        new_d_model = self._load_job_gan_model(self.job.get_job_id(), "D", self.job.get_train_d_model_class_name())
        time.sleep(0.5)
        g_model_pars = torch.load(g_global_model_path)
        new_g_model.load_state_dict(g_model_pars)
        d_model_pars = torch.load(d_global_model_path)
        new_d_model.load_state_dict(d_model_pars)
        return new_g_model, new_d_model


    def _train_gan(self, train_g_model, train_d_model, job_models_G_path, job_models_D_path, fed_step, local_epoch):

        step = 0
        g_model, d_model = train_g_model.get_model(), train_d_model.get_model()
        device = self.device
        g_model, d_model = g_model.to(device), d_model.to(device)
        g_model.train()
        d_model.train()
        while step < local_epoch:

            optimizer_G = self._get_gan_optimizer(g_model, train_g_model)
            optimizer_D = self._get_gan_optimizer(d_model, train_d_model)

            train_dataloader = torch.utils.data.DataLoader(self.data,
                                                           batch_size=self.train_g_model.get_train_strategy().get_batch_size(),
                                                           shuffle=True,
                                                           num_workers=0,
                                                           pin_memory=True)
            for idx, (batch_data, batch_target) in enumerate(train_dataloader):
                batch_imgs, batch_target = batch_data.to(device), batch_target.to(device)
                z = torch.randn(batch_imgs.shape[0], 100)
                z = z.view(-1, 100, 1, 1)
                z = z.to(device)
                fake_imgs = g_model(z)
                real_validity = d_model(batch_imgs)
                fake_validity = d_model(fake_imgs)
                gradient_penalty = self._calc_gradient_penalty(d_model, batch_imgs, fake_imgs)
                d_loss = torch.mean(fake_validity) - torch.mean(real_validity) + gradient_penalty
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                if idx % 5 == 0:
                    fake_imgs = g_model(z)
                    fake_validity = d_model(fake_imgs)
                    g_loss = -torch.mean(fake_validity)
                    optimizer_G.zero_grad()
                    g_loss.backward()
                    optimizer_G.step()
                    self.logger.info(
                        "train_D_loss: {}, train_G_loss: {}".format(d_loss, g_loss))
            step += 1
        self._save_generated_img(self.client_id, self.job.get_job_id(), fed_step, fake_imgs)
        torch.save(g_model.state_dict(),
                   os.path.join(job_models_G_path, "tmp_G_parameters_{}".format(fed_step)))

        # global_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(self.job.get_job_id()),
        #                                      "g_global_models")
        # global_model_pars_path = os.path.join(global_model_pars_dir, "global_parameters_{}".format(fed_step))
        # torch.save(g_model.state_dict(), global_model_pars_path)
        torch.save(d_model.state_dict(),
                   os.path.join(job_models_D_path, "tmp_D_parameters_{}".format(fed_step)))

        return d_loss, g_loss

    def _load_other_gan_models_pars(self, job_id, fed_step):

        job_model_base_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id))
        other_g_models_pars, other_d_models_pars = [], []
        connected_clients_num = 0
        client_distillation_g_model_path = os.path.join(job_model_base_path, "models_{}".format(self.client_id),
                                                      "distillation_g_model_pars")
        client_distillation_d_model_path = os.path.join(job_model_base_path, "models_{}".format(self.client_id),
                                                        "distillation_d_model_pars")
        if len(os.listdir(client_distillation_g_model_path)) >= fed_step and len(os.listdir(client_distillation_d_model_path)) >= fed_step:
            return None, None, 0
        for f in os.listdir(job_model_base_path):
            if f.find("models_") != -1 and int(f.split("_")[-1]) != int(self.client_id):
                connected_clients_num += 1
                d_files = os.listdir(os.path.join(job_model_base_path, f, "tmp_d_model_pars"))
                if len(d_files) == 0 or len(d_files) < fed_step:
                    return None, None, 0
                else:
                    other_g_models_pars.append(os.path.join(job_model_base_path, f, "tmp_g_model_pars", "tmp_G_parameters_{}".format(fed_step)))
                    other_d_models_pars.append(os.path.join(job_model_base_path, f, "tmp_d_model_pars",
                                                            "tmp_D_parameters_{}".format(fed_step)))
        time.sleep(0.5)
        for i in range(len(other_g_models_pars)):
            other_g_models_pars[i] = torch.load(other_g_models_pars[i])
        for i in range(len(other_d_models_pars)):
            other_d_models_pars[i] = torch.load(other_d_models_pars[i])
        return other_g_models_pars, other_d_models_pars, connected_clients_num + 1


    def _train_with_gan_distillation(self, train_g_model, train_d_model, other_g_models_pars, other_d_models_pars, local_epoch, distillation_g_model_path, distillation_d_model_path, job_l2_dist):

        step = 0
        scheduler = None
        device = self.device
        g_model = train_g_model.get_model()
        d_model = train_d_model.get_model()
        g_model, other_g_model = g_model.to(device), copy.deepcopy(g_model).to(device)
        d_model, other_d_model = d_model.to(device), copy.deepcopy(d_model).to(device)
        d_model.train()
        g_model.train()
        if train_g_model.get_train_strategy().get_scheduler() is not None:
            g_scheduler = train_g_model.get_train_strategy().get_scheduler()
        if train_d_model.get_train_strategy().get_scheduler() is not None:
            d_scheduler = train_d_model.get_train_strategy().get_scheduler()
        while step < local_epoch:

            train_dataloader = torch.utils.data.DataLoader(self.data,
                                                           batch_size=train_g_model.get_train_strategy().get_batch_size(),
                                                           shuffle=True,
                                                           num_workers=0,
                                                           pin_memory=True)

            if scheduler is not None:
                scheduler.step()

            g_optimizer = self._generate_new_optimizer(g_model, train_g_model.get_train_strategy().get_optimizer())
            d_optimizer = self._generate_new_optimizer(d_model, train_d_model.get_train_strategy().get_optimizer())
            acc = 0
            for idx, (batch_data, batch_target) in enumerate(train_dataloader):
                batch_data = batch_data.to(device)
                # batch_target = batch_target.to(device)
                z = torch.randn(batch_data.shape[0], 100)
                z = z.view(-1, 100, 1, 1)
                z = z.to(device)
                fake_imgs = g_model(z)
                real_validity = d_model(batch_data)
                fake_validity = d_model(fake_imgs)
                # g_pred = F.log_softmax(g_kl_pred, dim=1)
                # d_pred = F.log_softmax(d_kl_pred, dim=1)
                # acc += torch.eq(kl_pred.argmax(dim=1), batch_target).sum().float().item()
                loss_g_distillation, loss_d_distillation = 0, 0
                    # loss_g_distillation += self._compute_loss(LossStrategy.KLDIV_LOSS, F.log_softmax(fake_imgs, dim=1),
                    #                                             F.softmax(other_model_g_kl_pred, dim=1))

                for other_d_model_par in other_d_models_pars:
                    other_d_model.load_state_dict(other_d_model_par)
                    other_real_validity = other_d_model(batch_data).detach()
                    other_fake_validity = other_d_model(fake_imgs).detach()
                    # other_gradient_penalty = self._calc_gradient_penalty(other_d_model, batch_data, fake_imgs)
                    # loss_d_distillation += (torch.mean(other_fake_validity) - torch.mean(other_real_validity) + other_gradient_penalty)
                    loss_d_distillation += (F.mse_loss(real_validity, other_real_validity) + F.mse_loss(fake_validity, other_fake_validity))
                    # loss_d_distillation += self._compute_loss(LossStrategy.KLDIV_LOSS, F.log_softmax(real_validity, dim=1),
                    #                                             F.softmax(other_model_d_kl_pred, dim=1))


                # gradient_penalty = self._calc_gradient_penalty(d_model, batch_data, fake_imgs)
                # d_loss_s = torch.mean(fake_validity) - torch.mean(real_validity) + gradient_penalty
                # d_loss = self.job.get_distillation_alpha() * loss_d_distillation
                d_loss = torch.log(loss_d_distillation)
                # d_loss = loss_d_distillation
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                fake_imgs = g_model(z)
                fake_validity = d_model(fake_imgs)
                for other_g_model_par in other_g_models_pars:
                    other_g_model.load_state_dict(other_g_model_par)
                    other_fake_imgs = other_g_model(z).detach()
                    # other_fake_validity = d_model(other_fake_imgs)
                    loss_g_distillation += F.mse_loss(other_fake_imgs, fake_imgs)
                # g_loss_l = -torch.mean(fake_validity)
                # g_loss = self.job.get_distillation_alpha() * loss_g_distillation
                g_loss = torch.log(loss_g_distillation)
                # g_loss = g_loss_l + g_loss_d
                # g_loss = loss_g_distillation
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                if idx % 100 == 0:
                    self.logger.info("distillation_g_loss: {}, distillation_d_loss: {}".format(g_loss.item(), d_loss.item()))
                #     self.logger.info("distillation_loss: {}".format(loss.item()))
            step += 1
            # accuracy = acc / len(train_dataloader.dataset)

        torch.save(g_model.state_dict(),
                       os.path.join(distillation_g_model_path, "tmp_G_parameters_{}".format(self.fed_step[self.job.get_job_id()] + 1)))
        torch.save(d_model.state_dict(),
                   os.path.join(distillation_d_model_path,
                                "tmp_D_parameters_{}".format(self.fed_step[self.job.get_job_id()] + 1)))
        return None, d_loss.item()


    def _could_gan_fed_avg(self, job_id, fed_step):
        distillation_g_model_pars, distillation_d_model_pars = [], []
        job_model_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id))
        for model_dir in os.listdir(job_model_dir):
            if model_dir.find("models_") != -1:
                g_distillation_dir = os.path.join(job_model_dir, model_dir, "distillation_g_model_pars")
                d_distillation_dir = os.path.join(job_model_dir, model_dir, "distillation_d_model_pars")
                g_file_list = os.listdir(g_distillation_dir)
                d_file_list = os.listdir(d_distillation_dir)
                # file_list = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(distillation_dir, x)))
                if len(g_file_list) == 0 or len(d_file_list) == 0 or len(g_file_list) != fed_step or len(d_file_list) != fed_step:
                    return False, [], []
                else:
                    distillation_g_model_pars.append(os.path.join(g_distillation_dir, "tmp_G_parameters_{}".format(fed_step)))
                    distillation_d_model_pars.append(
                        os.path.join(d_distillation_dir, "tmp_D_parameters_{}".format(fed_step)))
        time.sleep(0.5)
        return True, distillation_g_model_pars, distillation_d_model_pars


    def _save_gan_global_model(self, job_id, fed_step, flag, global_model_pars):
        if flag == "G":
            global_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                                 "g_global_models")
            global_model_pars_path = os.path.join(global_model_pars_dir, "global_parameters_{}".format(fed_step))
        else:
            global_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                                 "d_global_models")
            global_model_pars_path = os.path.join(global_model_pars_dir, "global_parameters_{}".format(fed_step))

        torch.save(global_model_pars, global_model_pars_path)

    def _load_gan_d_distillation_model(self, distillation_d_path):
        new_d_model = self._load_job_gan_model(self.job.get_job_id(), "D", self.job.get_train_d_model_class_name())
        d_model_pars = torch.load(distillation_d_path)
        new_d_model.load_state_dict(d_model_pars)
        return new_d_model

    def _load_gan_g_distillation_model(self, distillation_g_path):
        new_g_model = self._load_job_gan_model(self.job.get_job_id(), "G", self.job.get_train_g_model_class_name())
        g_model_pars = torch.load(distillation_g_path)
        new_g_model.load_state_dict(g_model_pars)
        return new_g_model

    def _gan_fed_avg_aggregate(self, disillation_g_model_pars_list, disillation_d_model_pars_list, job_id, fed_step):
        avg_g_model_par = disillation_g_model_pars_list[0]
        avg_d_model_par = disillation_d_model_pars_list[0]
        for key in avg_g_model_par.keys():
            for i in range(1, len(disillation_g_model_pars_list)):
                # avg_model_par[key] += weight_list[i]*disillation_model_pars_list[i][key]
                avg_g_model_par[key] += disillation_g_model_pars_list[i][key]
            avg_g_model_par[key] = torch.div(avg_g_model_par[key], len(disillation_g_model_pars_list))
        for key in avg_d_model_par.keys():
            for i in range(1, len(disillation_d_model_pars_list)):
                # avg_model_par[key] += weight_list[i]*disillation_model_pars_list[i][key]
                avg_d_model_par[key] += disillation_d_model_pars_list[i][key]
            avg_d_model_par[key] = torch.div(avg_d_model_par[key], len(disillation_d_model_pars_list))
        # self._test(avg_model_par)
        self._save_gan_global_model(job_id, fed_step, "G", avg_g_model_par)
        self._save_gan_global_model(job_id, fed_step, "D", avg_d_model_par)

    def _execute_gan_fed_avg(self, client_id, job_id, fed_step, distillation_g_model_pars_file_list, distillation_d_model_pars_file_list):
        self.logger.info("client {} execute FedAvg".format(client_id))
        # last_global_model = self._load_global_model(job_id, fed_step - 1)
        distillation_g_model_list, distillation_d_model_list = [], []
        for distillation_g_model_pars_file in distillation_g_model_pars_file_list:
            distillation_g_model = self._load_gan_g_distillation_model(distillation_g_model_pars_file)
            distillation_g_model_list.append(distillation_g_model)

        for distillation_d_model_pars_file in distillation_d_model_pars_file_list:
            distillation_d_model = self._load_gan_d_distillation_model(distillation_d_model_pars_file)
            distillation_d_model_list.append(distillation_d_model)

        # kl_list, sum_kl_loss = self._calc_kl_loss(last_global_model, distillation_model_list)
        distillation_g_model_pars_list = [distillation_model.state_dict() for distillation_model in
                                        distillation_g_model_list]
        distillation_d_model_pars_list = [distillation_model.state_dict() for distillation_model in
                                          distillation_d_model_list]
        self._gan_fed_avg_aggregate(distillation_g_model_pars_list, distillation_d_model_pars_list, job_id, fed_step)


    def train(self):
        distillation_g_model_path, distillation_d_model_path = self._create_dislillation_gan_model_pars_path(self.client_id, self.job.get_job_id())
        local_g_models_path, local_d_models_path = self._create_job_gan_tmp_models_dir(self.client_id, self.job.get_job_id())
        self._init_gan_global_model(self.job.get_job_id(), 0)
        self.fed_step[self.job.get_job_id()] = 0
        while True:
            # self.fed_step[self.job.get_job_id()] = self._get_fed_step(self.job.get_job_id())

            if self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) >= self.job.get_epoch():
                break

            # local_g_fed_step = self._load_local_fed_step(local_g_models_path)
            local_g_fed_step, local_d_fed_step = self._load_local_fed_step(local_g_models_path), self._load_local_fed_step(local_d_models_path)
            if (local_g_fed_step == local_d_fed_step and local_d_fed_step < self.fed_step[self.job.get_job_id()] + 1):
                # aggregate_file = self._find_latest_global_model_pars(self.job.get_job_id())
                g_global_model, d_global_model = self._load_gan_global_model(self.job.get_job_id(), self.fed_step[self.job.get_job_id()])
                if g_global_model is not None and d_global_model is not None:
                    self.train_g_model.set_model(g_global_model)
                    self.train_d_model.set_model(d_global_model)
                    self._train_gan(self.train_g_model, self.train_d_model, local_g_models_path, local_d_models_path, self.fed_step[self.job.get_job_id()] + 1,
                                self.local_epoch)
            other_g_model_pars, other_d_model_pars, connected_clients_num = self._load_other_gan_models_pars(self.job.get_job_id(),
                                                                                   self.fed_step[self.job.get_job_id()]+1)
            # job_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
            # self.logger.info("job_{} is training, Aggregator strategy: {}, L2_dist: {}".format(self.job.get_job_id(),
            #                                                                                    self.job.get_aggregate_strategy(),
            #                                                                                    self.job.get_l2_dist()))
            if other_d_model_pars is not None and connected_clients_num != 0:

                self.logger.info("model distillating....")

                self.acc, loss = self._train_with_gan_distillation(self.train_g_model, self.train_d_model, other_g_model_pars, other_d_model_pars, self.local_epoch,
                                                            distillation_g_model_path, distillation_d_model_path,
                                                               self.job.get_l2_dist())
                # self.accuracy_list.append(self.acc)
                # self.loss_list.append(loss)
                self.logger.info("model distillation success")

                if (int(self.client_id) == (self.fed_step[self.job.get_job_id()] % connected_clients_num)):
                    # print(self.client_id, self.fed_step[self.job.get_job_id()] % connected_clients_num)
                    while True:
                        is_fed_avg, distillation_g_model_pars, distillation_d_model_pars = self._could_gan_fed_avg(self.job.get_job_id(), self.fed_step[
                            self.job.get_job_id()] + 1)
                        print("could fed_avg: {}".format(is_fed_avg))
                        if is_fed_avg:
                            self._execute_gan_fed_avg(self.client_id, self.job.get_job_id(),
                                                  self.fed_step[self.job.get_job_id()] + 1, distillation_g_model_pars, distillation_d_model_pars)
                            self.logger.info("execute_gan_fed_avg success")
                            self._save_global_generated_img(0, self.job.get_job_id(), self.fed_step[self.job.get_job_id()] + 1)
                            break
                        time.sleep(1)

                self.fed_step[self.job.get_job_id()] = self.fed_step.get(self.job.get_job_id()) + 1





class TrainMPCNormalStrategy(TrainNormalStrategy):
    """
    TrainMPCNormalStrategy is responsible for controlling the process of traditional training in cluster mode
    """

    def __init__(self, job, data, fed_step, client_ip, client_port, server_url, client_id, local_epoch, model, curve):
        super(TrainMPCNormalStrategy, self).__init__(job, data, fed_step, client_id, local_epoch, model, curve)
        self.server_url = server_url
        self.client_ip = client_ip
        self.client_port = client_port
        self.logger = LoggerFactory.getLogger("TrainMPCNormalStrategy", client_id, logging.INFO)

    def train(self):
        while True:
            self.fed_step[self.job.get_job_id()] = 0 if self.fed_step.get(self.job.get_job_id()) is None else \
                self.fed_step.get(self.job.get_job_id())
            # print("test_iter_num: ", self.job_iter_dict[self.job.get_job_id()])
            if self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) == self.job.get_epoch():
                self.logger.info("job_{} completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                if self.curve is True:
                    self._draw_curve()
                break
            elif self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) > self.job.get_epoch():
                self.logger.warning("job_{} has completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                if self.curve is True:
                    self._draw_curve()
                break
            self._prepare_job_model(self.job, self.server_url)
            self._prepare_job_init_model_pars(self.job, self.server_url)
            aggregate_file, fed_step = self._find_latest_aggregate_model_pars(self.job.get_job_id())
            if aggregate_file is not None and self.fed_step.get(self.job.get_job_id()) != fed_step:
                job_models_path = self._create_job_models_dir(self.client_id, self.job.get_job_id())
                # job_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
                self.logger.info("load {} parameters".format(aggregate_file))
                new_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
                model_pars = torch.load(aggregate_file)
                new_model.load_state_dict(model_pars)
                self.model.set_model(new_model)
                self.fed_step[self.job.get_job_id()] = fed_step
                self.logger.info("job_{} is training, Aggregator strategy: {}".format(self.job.get_job_id(),
                                                                                      self.job.get_aggregate_strategy()))
                self.acc, loss = self._train(self.model, job_models_path, self.fed_step.get(self.job.get_job_id()), self.local_epoch)
                self.loss_list.append(loss)
                self.accuracy_list.append(self.acc)
                files = self._prepare_upload_client_model_pars(self.job.get_job_id(), self.client_id,
                                                               self.fed_step.get(self.job.get_job_id()))
                response = requests.post("/".join(
                    [self.server_url, "modelpars", "%s" % self.client_id, "%s" % self.job.get_job_id(),
                     "%s" % self.fed_step[self.job.get_job_id()]]),
                    data=None, files=files)
                # print(response)


class TrainMPCDistillationStrategy(TrainDistillationStrategy):
    """
    TrainMPCDistillationStrategy is responsible for controlling the process of distillation training in cluster mode
    """

    def __init__(self, job, data, fed_step, client_ip, client_port, server_url, client_id, local_epoch, model, curve):
        super(TrainMPCDistillationStrategy, self).__init__(job, data, fed_step, client_id, local_epoch, model, curve)
        self.client_ip = client_ip
        self.client_port = client_port
        self.server_url = server_url
        self.logger = LoggerFactory.getLogger("TrainMPCDistillationStrategy", client_id, logging.INFO)

    def train(self):
        while True:
            if self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) >= self.job.get_epoch():
                final_pars_path = os.path.join(self.job_model_path, "models_{}".format(self.client_id),
                                               "tmp_parameters_{}".format(self.fed_step.get(self.job.get_job_id()) + 1))
                if os.path.exists(final_pars_path):
                    self._save_final_parameters(self.job.get_job_id(), final_pars_path)
                    self.logger.info("job_{} completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                if self.curve is True:
                    self._draw_curve()
                break
            self._prepare_job_model(self.job)
            self._prepare_job_init_model_pars(self.job, self.server_url)
            job_models_path = self._create_job_models_dir(self.client_id, self.job.get_job_id())
            # job_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
            response = requests.get("/".join([self.server_url, "otherclients", self.job.get_job_id()]))
            connected_clients_id = response.json()['data']
            for client_id in connected_clients_id:
                self.fed_step[self.job.get_job_id()] = 0 if self.fed_step.get(
                    self.job.get_job_id()) is None else self.fed_step.get(self.job.get_job_id())
                response = requests.get("/".join(
                    [self.server_url, "otherparameters", '%s' % self.job.get_job_id(), '%s' % client_id,
                     '%s' % (self.fed_step.get(self.job.get_job_id()) + 1)]))
                parameter_path = os.path.join(job_models_path,
                                              "tmp_parameters_{}".format(self.fed_step.get(self.job.get_job_id()) + 1))
                if response.status_code == 202:
                    self._write_bfile_to_local(response, parameter_path)
            other_model_pars, _ = self._load_other_models_pars(self.job.get_job_id(),
                                                               self.fed_step.get(self.job.get_job_id()))

            self.logger.info("job_{} is training, Aggregator strategy: {}, L2_dist: {}".format(self.job.get_job_id(),
                                                                                               self.job.get_aggregate_strategy(),
                                                                                               self.job.get_l2_dist()))
            if other_model_pars is not None and self._calc_rate(len(other_model_pars),
                                                                len(connected_clients_id)) >= THRESHOLD:

                self.logger.info("model distillating....")
                self.fed_step[self.job.get_job_id()] = self.fed_step.get(self.job.get_job_id()) + 1
                self.acc, loss = self._train_with_distillation(self.model, other_model_pars, self.local_epoch,
                                                               os.path.join(LOCAL_MODEL_BASE_PATH,
                                                                            "models_{}".format(self.job.get_job_id()),
                                                                            "models_{}".format(self.client_id)),
                                                               self.job.get_l2_dist())
                self.loss_list.append(loss)
                self.accuracy_list.append(self.acc)
                self.logger.info("model distillation success")
                files = self._prepare_upload_client_model_pars(self.job.get_job_id(), self.client_id,
                                                               self.fed_step.get(self.job.get_job_id()) + 1)
                response = requests.post("/".join(
                    [self.server_url, "modelpars", "%s" % self.client_id, self.job.get_job_id(),
                     "%s" % (self.fed_step.get(self.job.get_job_id()) + 1)]), data=None, files=files)
            else:
                job_model_client_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(self.job.get_job_id()),
                                                     "models_{}".format(self.client_id))
                if not os.path.exists(os.path.join(job_model_client_path, "tmp_parameters_{}".format(
                        self.fed_step.get(self.job.get_job_id()) + 1))):
                    self._train(self.model, job_model_client_path, self.fed_step.get(self.job.get_job_id()) + 1, self.local_epoch)
                    files = self._prepare_upload_client_model_pars(self.job.get_job_id(), self.client_id,
                                                                   self.fed_step.get(self.job.get_job_id()) + 1)
                    response = requests.post("/".join(
                        [self.server_url, "modelpars", "%s" % self.client_id, self.job.get_job_id(),
                         "%s" % (self.fed_step.get(self.job.get_job_id()) + 1)]), data=None, files=files)
