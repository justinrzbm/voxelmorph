'''
Author: Justin Rozeboom
wrozeboo@ualberta.ca

Pytorch training script for Voxelmorph and different losses.
'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import voxelmorph.torch as vxm
import neurite as ne
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
import matplotlib.pyplot as plt
import csv

import loss
import myutil

# # Load MNIST data
# transform = Compose([
#     ToTensor(),
#     lambda x: torch.nn.functional.pad(x, (2, 2, 2, 2), mode='constant', value=0),
#     Normalize((0.5,), (0.5,))
# ])
# mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
# mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)
# digit_sel = 3
# # Extract only instances of the digit 5
# x_train = torch.stack([img for img, label in mnist_train if label == digit_sel])
# y_train = torch.tensor([label for _, label in mnist_train if label == digit_sel])
# x_test = torch.stack([img for img, label in mnist_test if label == digit_sel])
# y_test = torch.tensor([label for _, label in mnist_test if label == digit_sel])

# # Load 2d MRI
# npz = np.load('data/brains2d/tutorial_data.npz')
# X = np.concatenate((npz['train'], npz['validate']))
# split = int(X.shape[0] * 0.8)
# x_train = torch.from_numpy(X[:split]).float()
# x_test = torch.from_numpy(X[split:]).float()

# Load OASIS slices
oasis_path = 'C:/Users/justi/Documents/GitHub/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks/Data/OASIS'
print('Loading OASIS data...')
X = myutil.get_oasis_data(oasis_path)
split = int(X.shape[0] * 0.8)
x_train = X[:split]
x_test = X[split:]
# idx = np.random.randint(0, x_train.shape[0], [5,])
# example_digits = [f for f in x_train[idx, ...]]
# ne.plot.slices(example_digits, cmaps=['gray'], do_colorbars=True)


# Split training data into train/validate
print('shape of x_train: {}, max value: {}, min value: {}'.format(x_train.shape, x_train.max(), x_train.min()))

# default UNet features
# nb_features = [
#     [32, 32, 32, 32],         # Encoder features
#     [32, 32, 32, 32, 32, 16]  # Decoder features
# ]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device available:", device)

# voxelmorph model
inshape = x_train.shape[-2:]
print("model input shape:", inshape)

# Pick the model, Non-diffeomorphic (MICCAI 2018 version) or Diffeomorphic (latest)
# vxm_model = vxm.networks.VxmDense(inshape, int_steps=0).to(device)            # Non-diffeomorphic MICCAI 2018
vxm_model = vxm.networks.VxmDense(inshape, int_steps=7, bidir=True).to(device)


# Define losses
mse_loss = vxm.losses.MSE().loss
# grad_loss = vxm.losses.Grad(penalty='l2').loss
grad_loss = loss.grad_loss # custom
jdet_loss = loss.jdet_loss
jdet_percent = loss.jdet_percent
bending_loss = loss.bending_loss
sym_similarity_loss = loss.sym_similarity_loss
loss_name = None

epochs = 100

def compute_loss(x_source, y_source, x_target, y_target, v, phi, loss_name):
    '''
    Choose loss based on `loss_name`.
    Parameters:
    - x_source: The source image.
    - y_source: The warped source -> target image.
    - x_target: The target image.
    - y_target: The warped target -> source image.
    - v: The velocity field (post-integration).
    - phi: The deformation field.
    - loss_name (str): The name of the loss function to compute. 
      Supported values are:
        - 'GradL2': Gradient loss with L2 penalty.
        - 'GradL1': Gradient loss with L1 penalty.
        - 'Jdet': Jacobian determinant loss.
        - 'GradL2+Jdet'
        - 'Bending': Bending energy loss.
        - 'Bending+Jdet'
        - ...etc
    Returns:
    - loss: The total loss.
    - lsim: The similarity loss term.
    - lreg: The regularization loss term.
    Notes:
    - `*_target` is not None implies that bidirectional loss computation is possible.
    '''
    lambda_smooth = 0.1
    lambda_jdet = 100

    if v is None and loss_name in ['VL2', 'SymSim+VL2', 'SymSim+VL2+Jdet']:
        raise ValueError("Need diffeomorphic model with velocity flow to compute some losses.")

    if loss_name == 'GradL2':
        lsim = mse_loss(x_target, y_source)
        lreg = lambda_smooth * grad_loss(None, phi, penalty='l2')
        loss = lsim + lreg

    elif loss_name == 'GradL1':
        lsim = mse_loss(x_target, y_source)
        lreg = lambda_smooth * grad_loss(None, phi, penalty='l1')
        loss = lsim + lreg

    elif loss_name == 'VL2':
        lsim = mse_loss(x_target, y_source)
        lreg = lambda_smooth * grad_loss(None, v, penalty='l2')
        loss = lsim + lreg
    
    elif loss_name == 'Jdet':
        lsim = mse_loss(x_target, y_source)
        lreg = lambda_jdet * jdet_loss(None, phi)
        loss = lsim + lreg

    elif loss_name == 'GradL2+Jdet':
        lsim = mse_loss(x_target, y_source)
        lreg = lambda_smooth * grad_loss(None, phi) + lambda_jdet * jdet_loss(None, phi)
        loss = lsim + lreg

    elif loss_name == 'Bending':
        lsim = mse_loss(x_target, y_source)
        lreg = lambda_smooth * bending_loss(None, phi)
        loss = lsim + lreg

    elif loss_name == 'Bending+Jdet':
        lsim = mse_loss(x_target, y_source)
        lreg = lambda_smooth * bending_loss(None, phi) + lambda_jdet * jdet_loss(None, phi)
        loss = lsim + lreg

    elif loss_name == 'SymSim+GradL2':
        lsim = sym_similarity_loss(I=x_source, J=x_target, I_warped=y_source, J_warped=y_target)
        lreg = lambda_smooth * grad_loss(None, phi, penalty='l2')
        loss = lsim + lreg

    elif loss_name == 'SymSim+VL2':
        lsim = sym_similarity_loss(I=x_source, J=x_target, I_warped=y_source, J_warped=y_target)
        lreg = lambda_smooth * grad_loss(None, v, penalty='l2')
        loss = lsim + lreg
    
    elif loss_name == 'SymSim+VL2+Jdet':
        lsim = sym_similarity_loss(I=x_source, J=x_target, I_warped=y_source, J_warped=y_target)
        lreg = lambda_smooth * grad_loss(None, v, penalty='l2') + lambda_jdet * jdet_loss(None, phi)
        loss = lsim + lreg

    else:
        raise ValueError

    return loss, lsim, lreg


# Optimizer
optimizer = optim.Adam(vxm_model.parameters(), lr=1e-3)#1e-4 might be better

# --- Data Generator ---
def vxm_data_generator(x_data, batch_size=32, seed=42):
    """
    Generator that takes in x_data of size [N, C, H, W], and prepares data for
    the pytorch VoxelMorph model.

    inputs:  moving [bs, C, H, W], fixed image [bs, C, H, W]
    target: moved image [bs, C, H, W]
    """
    torch.manual_seed(seed)  # reproducibility
    while True:
        # Prepare inputs
        idx1 = torch.randint(0, x_data.shape[0], (batch_size,))
        moving_images = x_data[idx1].to(device)
        idx2 = torch.randint(0, x_data.shape[0], (batch_size,))
        fixed_images = x_data[idx2].to(device)
        inputs = [moving_images, fixed_images]

        # Prepare outputs
        target = fixed_images

        yield (inputs, target)

'''
The variable names are a bit confusing right now, this is what they mean:
    - x_source: The source image.
    - y_source: The warped source -> target image.
    - x_target: The target image.
    - y_target: The warped target -> source image.
'''
def train(loss_name, nb_epochs=100):
    train_generator = vxm_data_generator(x_train)
    steps_per_epoch = 100
    training_losses = []

    for epoch in range(nb_epochs):
        vxm_model.train()
        epoch_loss = 0
        for step in range(steps_per_epoch):
            inputs, target = next(train_generator)
            x_source, x_target = inputs

            optimizer.zero_grad()
            y_source, y_target, v, phi = vxm_model(x_source, x_target)
            loss, lsim, lreg = compute_loss(x_source, y_source, x_target, y_target, v, phi, loss_name)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / steps_per_epoch
        training_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{nb_epochs}, Loss: {avg_epoch_loss:.4f}, MSE Loss: {lsim.item():.4f}, Reg Loss: {lreg.item():.4f}")

    # Save the trained model
    model_save_path = f'savedmodel/vxm_{loss_name}_e{nb_epochs}.pth'
    torch.save(vxm_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot training history
    plt.figure(figsize=(10, 5))
    if len(training_losses) > 1 and training_losses[0] > 7 * training_losses[1]:
        plt.plot(range(2, nb_epochs + 1), training_losses[1:], '.-', label='Training Loss') # exclude epoch 1 for plot
    else:
        plt.plot(range(1, nb_epochs + 1), training_losses, '.-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid()
    training_history_path = f'savedmodel/training_loss_{loss_name}.png'
    plt.savefig(training_history_path)
    print(f"Training history plot saved to {training_history_path}")

def eval(loss_name):
    print(f"Evaluating on test set (len={len(x_test)})")

    # Load the trained model
    model_load_path = f'savedmodel/vxm_{loss_name}_e{epochs}.pth'
    if os.path.exists(model_load_path):
        vxm_model.load_state_dict(torch.load(model_load_path, map_location=device))
        print(f"Model loaded from {model_load_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_load_path}")

    # --- Full Evaluation on Test Set ---
    test_generator = vxm_data_generator(x_test, batch_size=1)
    test_results = []

    vxm_model.eval()
    with torch.no_grad():
        sim_losses = []
        reg_losses = []
        total_losses = []
        jdpcs = []
        for _ in range(len(x_test)):
            test_input, test_target = next(test_generator)
            x_source, x_target = test_input
            y_source, y_target, v, phi = vxm_model(x_source, x_target)

            total_loss, lsim, lreg = compute_loss(x_source, y_source, x_target, y_target, v, phi, loss_name)
            jdpc = jdet_percent(phi)

            sim_losses.append(lsim.item())
            reg_losses.append(lreg.item())
            total_losses.append(total_loss.item())
            jdpcs.append(jdpc)

        avg_sim_loss = np.mean(sim_losses)
        std_sim_loss = np.std(sim_losses)
        avg_reg_loss = np.mean(reg_losses)
        std_reg_loss = np.std(reg_losses)
        avg_total_loss = np.mean(total_losses)
        std_total_loss = np.std(total_losses)
        avg_jdpc = np.mean(jdpcs)
        std_jdpc = np.std(jdpcs)

        print(f"Average Similarity Loss: {avg_sim_loss:.4f} ({std_sim_loss:.4f})")
        print(f"Average Regularization Loss: {avg_reg_loss:.4f} ({std_reg_loss:.4f})")
        print(f"Average Total Loss: {avg_total_loss:.4f} ({std_total_loss:.4f})")
        print(f"Average Jdet Percent: {avg_jdpc:.4f} ({std_jdpc:.4f})")

        # Prepare test results for CSV
        csv_columns = ['Loss Name', 'Sim Loss (Avg (Std))', 'Reg Loss (Avg (Std))', 'Total Loss (Avg (Std))', '%|J|<0 (Avg (Std))']
        test_results = [
            {
            'Loss Name': loss_name,
            'Sim Loss (Avg (Std))': f"{avg_sim_loss:.4f} ({std_sim_loss:.4f})",
            'Reg Loss (Avg (Std))': f"{avg_reg_loss:.4f} ({std_reg_loss:.4f})",
            'Total Loss (Avg (Std))': f"{avg_total_loss:.4f} ({std_total_loss:.4f})",
            '%|J|<0 (Avg (Std))': f"{avg_jdpc:.4f} ({std_jdpc:.4f})"
            }
        ]
        csv_file = 'test_results.csv'
        if not os.path.exists(csv_file):
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=csv_columns)
                writer.writeheader()
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)
            writer.writerows(test_results)
        print(f"Test results for loss '{loss_name}' appended to {csv_file}")

        # visualize some results for the same example
        example_slices_path = f'savedmodel/example_slices_{loss_name}.png'
        example_flow_path = f'savedmodel/example_flow_{loss_name}.png'
        vxm_model.eval()
        with torch.no_grad():
            y_source, _, v, phi = vxm_model(*example_input)
        if v==None:
            example_images = [img[0, 0].cpu().numpy() for img in example_input + [y_source, phi]]
            example_titles = ['moving', 'fixed', 'moved', 'phi']
            example_flow = phi[0].permute(1, 2, 0).cpu().numpy()  # flow shape (H, W, 2) for visualization
        else:
            example_images = [img[0, 0].cpu().numpy() for img in example_input + [y_source, v]]
            example_titles = ['moving', 'fixed', 'moved', 'velocity']
            example_flow = v[0].permute(1, 2, 0).cpu().numpy()  # flow shape (H, W, 2) for visualization

        myutil.slices(example_images, titles=example_titles, cmaps=['gray'], do_colorbars=True, savepath=example_slices_path)
        myutil.flow([example_flow], width=5, savepath=example_flow_path)


test_generator = vxm_data_generator(x_test, batch_size=1)
example_input, example_target = next(test_generator)

# for loss_name in ['Jdet', 'Bending', 'Bending+Jdet', 'VL2', 'SymSim+VL2', 'SymSim+VL2+Jdet']:
for loss_name in ['GradL2', 'GradL1', 'GradL2+Jdet', 'SymSim+GradL2']:
# for loss_name in ['GradL2', 'GradL1', 'VL2', 'Jdet', 'GradL2+Jdet', 'Bending', 'Bending+Jdet', 'SymSim+GradL2', 'SymSim+VL2', 'SymSim+VL2+Jdet']:
    print(f"\nTraining with loss: {loss_name}")
    train(loss_name, nb_epochs=epochs)
    eval(loss_name)


# TODO need to retest bending loss, Jdet, VL2 - done
# TODO need to retest on MNIST entirely





# # --- Registration ---
# val_generator = vxm_data_generator(x_val, batch_size=1)
# val_input = next(val_generator)[0]  # Extract only the inputs from the generator
# vxm_model.eval()
# with torch.no_grad():
#     val_pred = vxm_model(*val_input)

# # Visualize results
# images = [img[0, 0].cpu().numpy() for img in val_input + [val_pred[0], val_pred[1]]]
# titles = ['moving', 'fixed', 'moved', 'flow']
# ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

# # Visualize flow
# flow = val_pred[1][0].permute(1, 2, 0).cpu().numpy()  # flow shape (H, W, 2)
# ne.plot.flow([flow], width=5)


