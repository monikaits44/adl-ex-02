import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm
import math


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # betas from cumulative product ratios
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)


def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x
    x = torch.linspace(-6, 6, timesteps)
    sig = torch.sigmoid(x)
    betas = sig * (beta_end - beta_start) + beta_start
    return betas


class Diffusion:
    """
    Diffusion model supporting both unconditional and conditional (classifier-free guidance) generation.
    Use classes=None for fully unconditional training/sampling.
    """

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps

        self.img_size = img_size
        self.device = device

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps)

        # (2.2): Compute the central values for the equation in the forward pass already here so you can quickly use them in the forward pass.
        # Note that the function torch.cumprod may be of help
        if not isinstance(self.betas, torch.Tensor):
            self.betas = torch.tensor(self.betas, dtype=torch.float32)
        self.betas = self.betas.to(self.device)


        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        # These are already computed above:
        # - sqrt_alphas_cumprod and sqrt_one_minus_alphas_cumprod for forward process

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # Coefficients for computing the mean of q(x_{t-1} | x_t, x_0)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, classes=None, guidance_scale=0.0):
        """
        Single step reverse diffusion with optional classifier-free guidance.
        
        Args:
            model: The denoising model
            x: Current noisy image [B, C, H, W]
            t: Current timestep [B]
            t_index: Index of current timestep (for checking if t=0)
            classes: Class labels [B] or None for unconditional
            guidance_scale: Strength of classifier-free guidance (0 = no guidance)
        """
        # extract scalars/vectors for the batch shape
        betas_t = extract(self.betas, t, x.shape)  # beta_t
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = torch.sqrt(extract(self.sqrt_recip_alphas, t, x.shape))

        # Classifier-free guidance: interpolate between conditional and unconditional predictions
        if classes is not None and guidance_scale > 0:
            # Get conditional prediction
            predicted_noise_cond = model(x, t, classes)
            # Get unconditional prediction (using null token)
            null_classes = torch.full_like(classes, model.null_class if hasattr(model, 'null_class') else 0)
            predicted_noise_uncond = model(x, t, null_classes)
            # Apply guidance: eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            predicted_noise = predicted_noise_uncond + guidance_scale * (predicted_noise_cond - predicted_noise_uncond)
        else:
            # Standard prediction without guidance
            predicted_noise = model(x, t, classes) if classes is not None else model(x, t)

        # Equation for predicted mean (see DDPM eq 11/12)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, classes=None, guidance_scale=0.0):
        """
        Algorithm 2: Sampling from DDPM paper - reverse diffusion process.
        Generates images by iteratively denoising from pure noise x_T to clean image x_0.
        
        Args:
            model: The denoising model
            image_size: Size of images to generate
            batch_size: Number of images to generate
            channels: Number of color channels
            classes: Class labels [B] for conditional generation, or None for unconditional
            guidance_scale: Strength of classifier-free guidance (0 = no guidance, typical: 3-7)
        
        Returns:
            Generated images [B, C, H, W] in range [-1, 1]
        """
        model.eval()
        x = torch.randn((batch_size, channels, image_size, image_size), device=self.device)

        for i in tqdm(reversed(range(0, self.timesteps)), desc="sampling"):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, i, classes=classes, guidance_scale=guidance_scale)

        model.train()
        return x

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None, classes=None):
        """
        Forward diffusion process (adds noise to images).
        Note: classes parameter is not used in forward process but included for API consistency.
        """
        if noise is None:
            noise = torch.randn_like(x_zero)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_zero.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_zero.shape)

        return sqrt_alphas_cumprod_t * x_zero + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1", classes=None):
        """
        Compute training loss for the denoising model.
        
        Args:
            denoise_model: The model to train
            x_zero: Original clean images [B, C, H, W]
            t: Timesteps [B]
            noise: Optional pre-generated noise
            loss_type: "l1" or "l2"
            classes: Optional class labels [B] for conditional training
        """
        if noise is None:
            noise = torch.randn_like(x_zero)

        x_noisy = self.q_sample(x_zero=x_zero, t=t, noise=noise)
        
        # Predict noise with optional class conditioning
        if classes is not None:
            predicted_noise = denoise_model(x_noisy, t, classes)
        else:
            predicted_noise = denoise_model(x_noisy, t)

        if loss_type == "l1":
            loss = F.l1_loss(predicted_noise, noise)
        elif loss_type == "l2":
            loss = F.mse_loss(predicted_noise, noise)
        else:
            raise NotImplementedError(f"Loss {loss_type} not implemented")

        return loss