from src.VAE import VAE
from src.Encoder import Encoder
from src.Decoder import Decoder
from src.CustomDecoder import CustomDecoder
import scanpy as sc
import torch
from tqdm import tqdm
import pandas as pd
import os

torch.manual_seed(42)


def read_adata():
    train_path = 'data/SAD2022Z_Project1_GEX_train.h5ad'
    test_path = 'data/SAD2022Z_Project1_GEX_test.h5ad'
    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)
    return train_adata, test_adata


def train(data, model, optimizer, batch_size, epoch, beta):
    model.train()
    permutation = torch.randperm(data.size()[0])
    train_loss = 0
    train_kl_loss = 0
    train_recon_loss = 0
    for i in tqdm(range(0, data.size()[0], batch_size)):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        mini_batch = data[indices]
        loss, kl_loss, recon_loss = model.loss_function(mini_batch, beta)
        loss.backward()
        train_loss += loss.item()
        train_kl_loss += kl_loss.item()
        train_recon_loss += recon_loss.item()
        optimizer.step()
    train_loss /= data.size()[0]
    train_kl_loss /= data.size()[0]
    train_recon_loss /= data.size()[0]
    print(f'====> Epoch: {epoch} Average train loss: {train_loss}')
    return train_loss, train_kl_loss, train_recon_loss


def test(data, model, epoch, beta):
    model.eval()
    with torch.no_grad():
        test_loss, test_kl_loss, test_recon_loss = model.loss_function(data, beta)
    test_loss /= data.size()[0]
    test_kl_loss /= data.size()[0]
    test_recon_loss /= data.size()[0]
    print(f'====> Epoch: {epoch} Average test loss: {test_loss}')
    return test_loss.item(), test_kl_loss.item(), test_recon_loss.item()


def train_VAE(model, train_data, test_data, name):
    n_epochs = 50  # or whatever
    batch_size = 512  # or whatever
    beta = 1

    optimizer = torch.optim.Adam(params=model.parameters())
    train_loss_list = []
    train_kl_loss_list = []
    train_recon_loss_list = []
    test_loss_list = []
    test_kl_loss_list = []
    test_recon_loss_list = []

    for epoch in range(n_epochs):
        train_loss, train_kl_loss, train_recon_loss = train(train_data, model, optimizer, batch_size, epoch, beta)
        train_loss_list.append(train_loss)
        train_kl_loss_list.append(train_kl_loss)
        train_recon_loss_list.append(train_recon_loss)
        test_loss, test_kl_loss, test_recon_loss = test(test_data, model, epoch, beta)
        test_loss_list.append(test_loss)
        test_kl_loss_list.append(test_kl_loss)
        test_recon_loss_list.append(test_recon_loss)

    model = model.cpu()
    torch.save(model, f'{name}_model.pt')
    df = pd.DataFrame()
    df['test_loss'] = test_loss_list
    df['test_kl_loss'] = test_kl_loss_list
    df['test_recon_loss'] = test_recon_loss_list
    df['train_loss'] = train_loss_list
    df['train_kl_loss'] = train_kl_loss_list
    df['train_recon_loss'] = train_recon_loss_list
    df.to_csv(f'{name}_losses.csv')


def main():
    if not os.path.exists('Models'):
        os.mkdir('Models')
    train_adata, test_adata = read_adata()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    test_data = torch.Tensor(test_adata.layers['counts'].toarray()).to(device)
    train_data = torch.Tensor(train_adata.layers['counts'].toarray()).to(device)

    print("Training the first model: ")

    encoder = Encoder([5000, 2000, 1500, 1000, 50]).to(device)
    decoder = Decoder([50, 1000, 1500, 2000, 5000]).to(device)
    vae = VAE(encoder, decoder).to(device)
    train_VAE(vae, train_data, test_data, 'Models/VanillaVAE50')

    print("Training the second model: ")

    encoder = Encoder([5000, 2000, 1500, 1000, 10]).to(device)
    decoder = Decoder([10, 1000, 1500, 2000, 5000]).to(device)
    vae = VAE(encoder, decoder).to(device)
    train_VAE(vae, train_data, test_data, 'Models/VanillaVAE10')

    print("Training the third model: ")

    encoder = Encoder([5000, 2000, 1500, 1000, 5]).to(device)
    decoder = Decoder([5, 1000, 1500, 2000, 5000]).to(device)
    vae = VAE(encoder, decoder).to(device)
    train_VAE(vae, train_data, test_data, 'Models/VanillaVAE5')

    print("Training the fourth model: ")

    device = torch.device("cpu")
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    encoder = Encoder([5000, 2000, 1500, 1000, 10]).to(device)
    decoder = CustomDecoder([10, 1000, 1500, 2000, 5000]).to(device)
    vae = VAE(encoder, decoder).to(device)
    train_VAE(vae, train_data, test_data, 'Models/CustomVAE10')


if __name__ == '__main__':
    main()
