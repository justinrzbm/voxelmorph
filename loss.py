'''
Author: Justin Rozeboom
'''

# don't use any tensorflow
import torch
from voxelmorph.torch.losses import MSE as vxm_mse_loss

def grad_loss(y_true, phi, penalty='l2'):
    '''
    Regularization loss: L2 penalty on the gradient of the deformation field.
    `penalty` also optionally supports L1 penalty.
    Can be used to encourage smoothness on phi or v.

    From Voxelmorph
    Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019). VoxelMorph: A Learning Framework for Deformable Medical Image Registration. IEEE Transactions on Medical Imaging, 38(8), 1788–1800. https://doi.org/10.1109/TMI.2019.2897538
    '''
    # phi: (B, 2, H, W)
    dy = torch.abs(phi[:, :, 1:, :] - phi[:, :, :-1, :]) / 2
    dx = torch.abs(phi[:, :, :, 1:] - phi[:, :, :, :-1]) / 2

    if penalty.lower() == 'l2':
        return torch.mean(dx ** 2) + torch.mean(dy ** 2)
    elif penalty.lower() == 'l1':
        return torch.mean(dx) + torch.mean(dy)
    else:
        raise ValueError("Invalid penalty. Must be 'l2' or 'l1'.")
    

def jdet_loss(y_true, phi):
    '''
    Jacobian determinant loss. Penalizes voxels/pixels with negative Jacobian determinants.

    From: Mok, T. C. W., & Chung, A. C. S. (2021). Fast Symmetric Diffeomorphic Image Registration with Convolutional Neural Networks (No. arXiv:2003.09514). arXiv. https://doi.org/10.48550/arXiv.2003.09514
    '''
    # phi: (B, 2, H, W)
    # central difference
    dy = phi[:, :, 1:, :-1] - phi[:, :, :-1, :-1]
    dx = phi[:, :, :-1, 1:] - phi[:, :, :-1, :-1]

    # 2d jacobian determinant is = (∂phi_x/∂x) * (∂phi_y/∂y) - (∂phi_x/∂y) * (∂phi_y/∂x)
    Jdet = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    
    # penalize only negative det
    # positive det is squashed to 0 with relu
    neg_Jdet = torch.relu(-Jdet)
    return torch.sum(neg_Jdet) / Jdet.numel()

def jdet_percent(phi):
    '''
    Calculate the percentage of negative Jacobian determinant voxels / pixels in the image. 
    Can input a batch or single image.
    '''
    if phi.dim() == 3:
        phi = phi.unsqueeze(0)
    # phi: (B, 2, H, W)
    dy = phi[:, :, 1:, :-1] - phi[:, :, :-1, :-1]
    dx = phi[:, :, :-1, 1:] - phi[:, :, :-1, :-1]

    Jdet = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]

    neg_count = torch.sum(Jdet < 0) # number of negative Jdet pixels
    total_count = Jdet.numel()

    percentage = (neg_count / total_count) * 100
    return percentage.item() # convert to scalar

def bending_loss(y_true, phi):
    '''
    Penalizes second derivatives (bending energy) of the deformation field phi. 

    As in FFD: https://ieeexplore.ieee.org/document/796284 - This is the 2D version
    '''
    # phi: (B, 2, H, W)
    # 1D central difference for second derivative: f''(x) ≈ f(x+1) - 2*f(x) + f(x-1)
    ddx = phi[:, :, :, 2:] - 2*phi[:, :, :, 1:-1] + phi[:, :, :, :-2]
    ddy = phi[:, :, 2:, :] - 2*phi[:, :, 1:-1, :] + phi[:, :, :-2, :]
    ddxy = phi[:, :, 2:, 2:] - phi[:, :, 2:, :-2] - phi[:, :, :-2, 2:] + phi[:, :, :-2, :-2]  

    bending_energy = torch.mean(ddy ** 2) + torch.mean(ddx ** 2) + 2*torch.mean(ddxy ** 2)
    return bending_energy



# The following functions require the diffeomorphic flow calculations (SVF) and sometimes bidirectional flow.
# the v, phi, and warped image can be passed to these function directly (no warping inside the function)

def v_magnitude_loss(v):
    '''
    L2 penalty on velocity field v.
    
    Different from L_mag in SYMNet: https://arxiv.org/abs/2003.09514 that tries to make Vxy and Vyx the same magnitude.
    '''
    return torch.mean(v ** 2)

def sym_similarity_loss(I, J, I_warped, J_warped):
    '''
    Image similarity loss in both forward and backwards direction.
    as in SYMNet: https://arxiv.org/abs/2003.09514
    as in CycleMorph: https://arxiv.org/abs/2008.05772
    '''
    # Compute the mean squared error for both terms
    mse_I = torch.mean((I_warped - J) ** 2)
    mse_J = torch.mean((J_warped - I) ** 2)
    loss = mse_I + mse_J
    return loss


def inv_consistency_loss(phixy, phiyx):
    '''
    Encourages phi in either direction to be inverses of one another.

    CycleMorph: https://arxiv.org/abs/2008.05772 equation (7)
    '''
    raise NotImplementedError

def inv_magnitude_loss(phixy, phiyx):
    '''
    Encourages forward and backwards deformation fields to have the same magnitude.

    CycleMorph: https://arxiv.org/abs/2008.05772
    '''
    raise NotImplementedError



# Testing
if __name__ == '__main__':
    # Test grad_loss
    phi = torch.randn(1, 2, 5, 5)
    print("grad_loss:", grad_loss(None, phi, reg_type='l2'))
    print("grad_loss:", grad_loss(None, phi, reg_type='l1'))

    # Test jdet_loss
    print("jdet_loss:", jdet_loss(None, phi))

    # Test jdet_percent
    print("jdet_percent:", jdet_percent(phi))

    # Test bending_loss
    # Zero deformation field
    zeros = torch.zeros(1, 2, 5, 5)
    assert bending_loss(None, zeros).item() == 0.0, "Bending loss for zero deformation field should be 0."

    # Random deformation field
    loss = bending_loss(None, phi)
    print("Random bending loss:", loss.item())
    assert loss.item() > 0, "Bending loss for random deformation field should be greater than 0."

    # Constant deformation field
    ones = torch.ones(1, 2, 5, 5)
    assert bending_loss(None, ones).item() == 0.0, "Bending loss for constant deformation field should be 0."

    # Test inv_consistency_loss

    # Create synthetic X and Y
    X = torch.linspace(0, 1, steps=25).view(1, 1, 5, 5)
    Y = torch.zeros_like(X)  # Initialize Y with zeros
    Y[:, :, 1:, 1:] = X[:, :, :-1, :-1]  # Shift X right and down, pad with zeros

    # Create warp fields that perfectly map X to Y and Y to X
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, 5),
        torch.linspace(-1, 1, 5),
        indexing='ij'
    ), dim=0).unsqueeze(0)  # Shape: (1, 2, 5, 5)

    # Define the warp fields
    phixy = grid.clone()
    phiyx = grid.clone()

    # Apply horizontal and vertical displacement for X -> Y and Y -> X
    phixy[..., 0] += 0.5 * (2 / 4)  # Move right (normalized grid adjustment)
    phixy[..., 1] += 0.5 * (2 / 4)  # Move down (normalized grid adjustment)

    phiyx[..., 0] -= 0.5 * (2 / 4)  # Move left (inverse of right)
    phiyx[..., 1] -= 0.5 * (2 / 4)  # Move up (inverse of down)

    # Test inv_consistency_loss with perfect warp fields
    loss = inv_consistency_loss(X, Y, phixy, phiyx)
    print("inv_consistency_loss with perfect warp fields:", loss.item())
    assert loss.item() == 0.0, "Loss should be 0 for perfect warp fields."
