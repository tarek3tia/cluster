import os

from pytorch_lightning import Trainer, LightningModule
from torch import optim, nn
from torch.distributed import init_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "2"
os.environ["NODE_RANK"] = "0"

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitAutoEncoder(LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


trainer = Trainer(limit_train_batches=100, max_epochs=1, num_nodes=2, accelerator='gpu', strategy='ddp',
                  logger=False)
init_process_group(backend='nccl', init_method='tcp://192.168.126.228:12345', world_size=2, rank=0)
# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)
# setup data
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = DataLoader(dataset)
print('....waiting for other nodes to join.... ')
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
