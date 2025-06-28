import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt, PTVs):
        pred_A = pred
        gt_dose = gt[0]
        possible_dose_mask = gt[1]


        pred_A = pred_A[possible_dose_mask > 0]

        gt_dose = gt_dose[possible_dose_mask > 0]

        L1_loss = self.L1_loss_func(pred_A, gt_dose)
        return L1_loss
    
import torch
import torch.nn as nn
from pytorch_msssim import ssim


    
class Loss_DC(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

        # Define learnable weights for max and min dose penalties
        self.max_dose_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.min_dose_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)


    def forward(self, pred, gt, PTVs):

        device = pred.device
        pred_A = pred
        gt_dose = gt[0]
        possible_dose_mask = gt[1]

        
       
        print("PredA", pred_A.shape)
        print("gtdose", gt_dose.shape)
        print("possible_dose_mask", possible_dose_mask.shape)
        print("PTV:", PTVs.shape)
        # Mask the predicted and ground truth values
        pred_A = pred_A[possible_dose_mask > 0]
        gt_dose = gt_dose[possible_dose_mask > 0]

        print("Masked PredA:", pred_A.shape)
        print("Masked gt_dose:", gt_dose.shape)

        # L1 loss
        L1_loss = self.L1_loss_func(pred_A, gt_dose)

        # Dynamic Dose Constraints (penalizing extreme doses)
        max_dose_limit = gt_dose.max()
        min_dose_limit = gt_dose.min()

        # Squared penalties for extreme doses
        max_dose_penalty = torch.clamp(pred_A.max() - max_dose_limit, min=0) ** 2
        min_dose_penalty = torch.clamp(min_dose_limit - pred_A.min(), min=0) ** 2

        # Apply learnable weights for dose constraint penalties
        dose_constraint_loss = self.max_dose_weight * max_dose_penalty + self.min_dose_weight * min_dose_penalty
        
        # Total loss
        total_loss = L1_loss + dose_constraint_loss 

        return total_loss
    

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedLoss(nn.Module):
    def __init__(self):
        super(AdvancedLoss, self).__init__()

        # Initialize learnable parameters for each loss weight
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Weight for L1 loss
        self.beta = nn.Parameter(torch.tensor(1.0))   # Weight for smoothness loss

        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt, PTV, OAR, mask=None):
        # Extract ground truth components
        gt_dose = gt[0]  # GT dose values
        possible_dose_mask = gt[1]  # Mask for valid dose regions
    
        # Mask the predictions and ground truth based on valid regions
        pred_masked = pred[possible_dose_mask > 0]
        gt_dose_masked = gt_dose[possible_dose_mask > 0]

        print(OAR.shape)
        # L1 Loss
        L1_loss = self.L1_loss_func(pred_masked, gt_dose_masked)

        # Edge-Aware Smoothness Regularization
        smoothness_loss = self.compute_smoothness_loss(pred, gt_dose, mask)



        # Weighted sum of all losses using learnable weights
        total_loss = (self.alpha * L1_loss +
                      self.beta * smoothness_loss)

        return total_loss

    def compute_smoothness_loss(self, pred, gt_dose, mask):
        # Compute gradients along the x, y, z directions
        gradient_x = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
        gradient_y = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
        gradient_z = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])

        # Edge mask based on ground truth dose gradients
        edge_mask_x = torch.abs(gt_dose[:, :, 1:, :, :] - gt_dose[:, :, :-1, :, :])
        edge_mask_y = torch.abs(gt_dose[:, :, :, 1:, :] - gt_dose[:, :, :, :-1, :])
        edge_mask_z = torch.abs(gt_dose[:, :, :, :, 1:] - gt_dose[:, :, :, :, :-1])

        # Penalize gradients where there are no edges
        smoothness_loss = (gradient_x * (1 - edge_mask_x)).mean() + \
                           (gradient_y * (1 - edge_mask_y)).mean() + \
                           (gradient_z * (1 - edge_mask_z)).mean()

        return smoothness_loss


class SharpDoseLoss(nn.Module):
    def __init__(self):
        super(SharpDoseLoss, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))    
        self.beta = nn.Parameter(torch.tensor(2.0))     
        self.gamma = nn.Parameter(torch.tensor(1.5))    
        
        self.L1_loss_func = nn.L1Loss(reduction='mean')
        
    def forward(self, pred, gt, PTV, OAR, mask=None):
        gt_dose = gt[0]
        possible_dose_mask = gt[1]
        
        # Basic L1 Loss
        pred_masked = pred[possible_dose_mask > 0]
        gt_masked = gt_dose[possible_dose_mask > 0]
        L1_loss = self.L1_loss_func(pred_masked, gt_masked)
        
        # Enhanced gradient matching with focus on sharp transitions
        gradient_loss = self.compute_sharp_gradient_loss(pred, gt_dose)
        
        # Special focus on high gradient regions
        high_gradient_loss = self.compute_high_gradient_region_loss(pred, gt_dose)
        
        total_loss = (self.alpha * L1_loss + 
                     self.beta * gradient_loss +
                     self.gamma * high_gradient_loss)
        
        return total_loss
    
    def compute_sharp_gradient_loss(self, pred, gt_dose):
        # Compute gradients in all directions
        dx_pred = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        dy_pred = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        dz_pred = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        
        dx_gt = gt_dose[:, :, 1:, :, :] - gt_dose[:, :, :-1, :, :]
        dy_gt = gt_dose[:, :, :, 1:, :] - gt_dose[:, :, :, :-1, :]
        dz_gt = gt_dose[:, :, :, :, 1:] - gt_dose[:, :, :, :, :-1]
        
        gradient_loss = (torch.pow(torch.abs(dx_pred - dx_gt), 2.0).mean() +
                        torch.pow(torch.abs(dy_pred - dy_gt), 2.0).mean() +
                        torch.pow(torch.abs(dz_pred - dz_gt), 2.0).mean())
        
        return gradient_loss
    
    def compute_high_gradient_region_loss(self, pred, gt_dose):
        # Compute padded gradients to maintain size
        dx_gt = torch.zeros_like(gt_dose)
        dy_gt = torch.zeros_like(gt_dose)
        dz_gt = torch.zeros_like(gt_dose)
        
        dx_gt[:, :, 1:, :, :] = gt_dose[:, :, 1:, :, :] - gt_dose[:, :, :-1, :, :]
        dy_gt[:, :, :, 1:, :] = gt_dose[:, :, :, 1:, :] - gt_dose[:, :, :, :-1, :]
        dz_gt[:, :, :, :, 1:] = gt_dose[:, :, :, :, 1:] - gt_dose[:, :, :, :, :-1]
        
        # Calculate gradient magnitude
        gradient_magnitude = torch.sqrt(dx_gt**2 + dy_gt**2 + dz_gt**2)
        
        # Create mask for high gradient regions
        threshold = gradient_magnitude.mean() + gradient_magnitude.std()
        high_gradient_mask = (gradient_magnitude > threshold)
        
        # Calculate predicted gradients with padding
        dx_pred = torch.zeros_like(pred)
        dy_pred = torch.zeros_like(pred)
        dz_pred = torch.zeros_like(pred)
        
        dx_pred[:, :, 1:, :, :] = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        dy_pred[:, :, :, 1:, :] = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        dz_pred[:, :, :, :, 1:] = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        
        # Focus loss on high gradient regions
        high_gradient_loss = F.mse_loss(
            dx_pred[high_gradient_mask],
            dx_gt[high_gradient_mask]
        ) + F.mse_loss(
            dy_pred[high_gradient_mask],
            dy_gt[high_gradient_mask]
        ) + F.mse_loss(
            dz_pred[high_gradient_mask],
            dz_gt[high_gradient_mask]
        )
        
        return high_gradient_loss
    

class Loss_DC_PTV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

        # Define learnable weights for max and min dose penalties
        self.max_dose_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.min_dose_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # Weight for penalizing incorrect dose in PTV region
        self.PTV_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # Weight for penalizing incorrect dose in OAR region
        self.OAR_weights = nn.Parameter(torch.ones(7), requires_grad=True)  # One weight for each OAR

    def forward(self, pred, gt, PTVs, OAR):
        device = pred.device
        pred_A = pred
        gt_dose = gt[0]
        possible_dose_mask = gt[1]
        
        
        # Masked values using possible_dose_mask
        pred_A_masked = pred_A[possible_dose_mask > 0]
        gt_dose_masked = gt_dose[possible_dose_mask > 0]

        # ---- Standard L1 loss ----
        L1_loss = self.L1_loss_func(pred_A_masked, gt_dose_masked)

        # ---- PTV Weighted Loss ----
        # Extract values inside the PTV region
        pred_PTV = pred_A[PTVs > 0]  # Predicted doses within PTV
        gt_PTV = gt_dose[PTVs > 0]  # Ground truth doses within PTV

        PTV_Loss = self.L1_loss_func(pred_PTV, gt_PTV)  # L1 loss in PTV
        PTV_Loss = self.PTV_weight * PTV_Loss  # Scale by learnable weight

        # ---- OAR Combined Binary Mask ----
        # Create a combined binary mask for all OARs (any voxel in any OAR is set to 1)
        combined_OAR_mask = torch.sum(OAR, dim=1) > 0  # This creates a binary mask for all OARs combined
        
        # Make sure the combined_OAR_mask has the same shape as pred_A by adding an extra channel dimension (1)
        combined_OAR_mask = combined_OAR_mask.unsqueeze(1)  # Shape [batch_size, 1, height, width, depth]
        
        # Predicted doses inside the combined OAR region (mask applied to OAR region)
        pred_OAR_combined = pred_A[combined_OAR_mask > 0]  
        gt_OAR_combined = gt_dose[combined_OAR_mask > 0]  
        
        # L1 loss for the combined OAR region
        OAR_Loss = self.L1_loss_func(pred_OAR_combined, gt_OAR_combined)

        # ---- Dose Constraints (Penalizing Extreme Doses) ----
        max_dose_limit = gt_dose_masked.max()
        min_dose_limit = gt_dose_masked.min()

        max_dose_penalty = torch.clamp(pred_A_masked.max() - max_dose_limit, min=0) ** 2
        min_dose_penalty = torch.clamp(min_dose_limit - pred_A_masked.min(), min=0) ** 2

        dose_constraint_loss = self.max_dose_weight * max_dose_penalty + self.min_dose_weight * min_dose_penalty

        # ---- Total Loss ----
        total_loss = L1_loss + PTV_Loss + OAR_Loss + dose_constraint_loss

        return total_loss



class Loss_DC_PTV(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

        # Define learnable weights for max and min dose penalties
        self.max_dose_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.min_dose_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # Weight for penalizing incorrect dose in PTV region
        self.PTV_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # Weight for penalizing incorrect dose in OAR region
        self.OAR_weights = nn.Parameter(torch.ones(7), requires_grad=True)  # One weight for each OAR

    def forward(self, pred, gt, PTVs, OAR):
        device = pred.device
        pred_A = pred
        gt_dose = gt[0]
        possible_dose_mask = gt[1]
        
        
        # Masked values using possible_dose_mask
        pred_A_masked = pred_A[possible_dose_mask > 0]
        gt_dose_masked = gt_dose[possible_dose_mask > 0]

        # ---- Standard L1 loss ----
        L1_loss = self.L1_loss_func(pred_A_masked, gt_dose_masked)

        # ---- PTV Weighted Loss ----
        # Extract values inside the PTV region
        pred_PTV = pred_A[PTVs > 0]  # Predicted doses within PTV
        gt_PTV = gt_dose[PTVs > 0]  # Ground truth doses within PTV

        PTV_Loss = self.L1_loss_func(pred_PTV, gt_PTV)  # L1 loss in PTV
        PTV_Loss = self.PTV_weight * PTV_Loss  # Scale by learnable weight

        # ---- OAR Combined Binary Mask ----
        # Create a combined binary mask for all OARs (any voxel in any OAR is set to 1)
        combined_OAR_mask = torch.sum(OAR, dim=1) > 0  # This creates a binary mask for all OARs combined
        
        # Make sure the combined_OAR_mask has the same shape as pred_A by adding an extra channel dimension (1)
        combined_OAR_mask = combined_OAR_mask.unsqueeze(1)  # Shape [batch_size, 1, height, width, depth]
        
        # Predicted doses inside the combined OAR region (mask applied to OAR region)
        pred_OAR_combined = pred_A[combined_OAR_mask > 0]  
        gt_OAR_combined = gt_dose[combined_OAR_mask > 0]  
        
        # L1 loss for the combined OAR region
        OAR_Loss = self.L1_loss_func(pred_OAR_combined, gt_OAR_combined)

        # ---- Total Loss ----
        total_loss = L1_loss + PTV_Loss + OAR_Loss


        # ---- Check if PTV and Combined OAR masks are identical ----
        if torch.equal(combined_OAR_mask, PTVs):
            print("PTV and OAR masks are identical.")
        else:
            print("PTV and OAR masks are NOT identical.")
        
        # ---- Check if PTV and OAR regions contain non-zero values (i.e., if they exist) ----
        if torch.any(PTVs > 0):
            print("PTV region exists (non-zero).")
        else:
            print("PTV region is all zeros.")
        
        if torch.any(combined_OAR_mask > 0):
            print("OAR region exists (non-zero).")
        else:
            print("OAR region is all zeros.")
      
     
        return total_loss