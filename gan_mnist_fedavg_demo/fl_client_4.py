import os
import torch
from torchvision import datasets, transforms
from rdfl.core.client import FLClient
from rdfl.core.strategy import WorkModeStrategy, TrainStrategy, LossStrategy
from rdfl.core.trainer_controller import TrainerController
import sys
sys.path.append(os.path.join(os.path.abspath("../gan_mnist_fedavg_demo"), "utils"))
CLIENT_ID = 4

if __name__ == "__main__":
    # CLIENT_ID = int(sys.argv[1])

    dataset_path = os.path.join(os.path.abspath("../gan_mnist_fedavg_demo"), "data",
                                "train_dataset_{}".format(CLIENT_ID))
    test_dataset_path = os.path.join(os.path.abspath("../gan_mnist_fedavg_demo"), "data",
                                     "test_dataset")
    test_dataset = torch.load(test_dataset_path)
    dataset = torch.load(dataset_path)
    client = FLClient()

    gfl_g_model, gfl_d_model = client.get_remote_gan_gfl_models()

    g_optimizer = torch.optim.Adam(gfl_g_model.get_model().parameters(), lr=1e-4, betas=(0.5, 0.9))
    train_g_strategy = TrainStrategy(optimizer=g_optimizer, batch_size=64, loss_function=LossStrategy.CE_LOSS)
    gfl_g_model.set_train_strategy(train_g_strategy)

    d_optimizer = torch.optim.Adam(gfl_d_model.get_model().parameters(), lr=1e-4, betas=(0.5, 0.9))
    train_d_strategy = TrainStrategy(optimizer=d_optimizer, batch_size=64, loss_function=LossStrategy.CE_LOSS)
    gfl_d_model.set_train_strategy(train_d_strategy)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TrainerController(work_mode=WorkModeStrategy.WORKMODE_STANDALONE, g_model=gfl_g_model, d_model=gfl_d_model, data=dataset,
                      test_data=test_dataset, client_id=CLIENT_ID,
                      curve=False, local_epoch=5, concurrent_num=3, device=device).start()
