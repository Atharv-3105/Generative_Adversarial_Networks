import torch
import torch.nn as nn

"""Gradient Penalty was introduced to stabilize the training of WGANs
It is used to enforce the Lipschitz constraint on the critic.
In this we add interpolated samples between real and fake samples and compute the gradient penalty.
"""
def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, c, h,w  = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1,1).repeat(1,c,h,w)).to(device)
    interpolated_images = real*epsilon + fake*(1-epsilon)  #for eg:- Epsilon=0.1; Then Interpolated_Image will be generated with 10% of real image and 90% of fake image.
    
    #Calculate the critic score of the interpolated images
    interpolated_critic_score = critic(interpolated_images)
    
    #We are computing gradients of interpolated_critic_score w.r.t interpolated_images
    gradients = torch.autograd.grad(
        inputs = interpolated_images,
        outputs= interpolated_critic_score,
        grad_outputs=torch.ones_like(interpolated_critic_score),
        create_graph=True,
        retain_graph=True
        )[0]
    
    gradients = gradients.view(gradients.shape[0], -1) #We are flattening the gradients
    gradient_norm = gradients.norm(2, dim=1) #Calculating the L2 norm of the gradients
    gradient_penalty = torch.mean((gradient_norm -1)**2)
    return gradient_penalty