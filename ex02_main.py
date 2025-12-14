import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os

from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule, cosine_beta_schedule
from torchvision.utils import save_image

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--save_dir', type=str, default="outputs", help='where to store generated images')
    
    # Inference arguments
    parser.add_argument('--inference', action='store_true', default=False, help='Run inference only (no training)')
    parser.add_argument('--model_path', type=str, default='models/DDPM_30epochs_cosine/ckpt.pt', help='Path to saved model checkpoint')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of samples to generate during inference')
    
    # Conditional generation arguments
    parser.add_argument('--conditional', action='store_true', default=False, help='Enable conditional generation with classifier-free guidance')
    parser.add_argument('--p_uncond', type=float, default=0.1, help='Probability of unconditional training (default: 0.1)')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='Classifier-free guidance scale for inference (default: 3.0)')
    parser.add_argument('--sample_classes', type=str, default=None, help='Comma-separated class indices to sample (e.g., "0,1,2" for specific classes)')

    return parser.parse_args()


def sample_and_save_images(n_images, diffusor, model, device, store_path, image_size=32, channels=3, 
                           classes=None, guidance_scale=0.0):
    """
    Generate n_images using diffusor.sample and save them to store_path directory.
    Each image is saved as PNG. Images are rescaled from [-1,1] -> [0,1] before saving.
    
    Args:
        n_images: Number of images to generate
        diffusor: Diffusion model instance
        model: U-Net model
        device: Device to run on
        store_path: Directory to save images
        image_size: Size of images
        channels: Number of channels
        classes: Optional class labels for conditional generation
        guidance_scale: Classifier-free guidance scale (0 = no guidance)
    """
    os.makedirs(store_path, exist_ok=True)
    model.to(device)
    model.eval()
    batch_size = min(n_images, 16)  # sample in one batch up to 16
    results = []
    remaining = n_images
    idx = 0

    while remaining > 0:
        this_batch = min(remaining, batch_size)
        
        # Prepare class labels for this batch if conditional
        batch_classes = None
        if classes is not None:
            if len(classes) >= this_batch:
                batch_classes = classes[:this_batch].to(device)
                classes = classes[this_batch:]  # Remove used classes
            else:
                # Not enough classes provided, repeat or use None
                batch_classes = None
        
        samples = diffusor.sample(model, image_size, batch_size=this_batch, channels=channels,
                                 classes=batch_classes, guidance_scale=guidance_scale)
        # samples are in [-1,1], convert to [0,1]
        samples = (samples.clamp(-1, 1) + 1) / 2.0
        for i in range(this_batch):
            class_label = f"_class{batch_classes[i].item()}" if batch_classes is not None else ""
            out_path = os.path.join(store_path, f"sample_{idx:04d}{class_label}.png")
            save_image(samples[i], out_path)
            idx += 1
        remaining -= this_batch

    model.train()
    return store_path


def test(model, testloader, diffusor, device, args):
    """
    Validation / test loop:
    - compute average loss on given dataloader using diffusor.p_losses
    - also generate 8 samples and store them under args.save_dir/epoch-X if dataloader is validation
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="validation"):
            images = images.to(device)
            labels = labels.to(device) if args.conditional else None
            batch_size = images.shape[0]
            t = torch.randint(0, diffusor.timesteps, (batch_size,), device=device).long()
            loss = diffusor.p_losses(model, images, t, loss_type="l2", classes=labels)
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    print(f"Validation Loss: {avg_loss:.6f}")

    # also generate a small grid of images for monitoring
    out_dir = os.path.join(args.save_dir, "samples")
    # For conditional models, sample from random classes
    sample_classes = None
    guidance = 0.0
    if args.conditional and hasattr(model, 'num_classes'):
        sample_classes = torch.randint(0, model.num_classes, (8,), device=device)
        guidance = args.guidance_scale
    sample_and_save_images(8, diffusor, model, device, out_dir, image_size=diffusor.img_size,
                          classes=sample_classes, guidance_scale=guidance)
    model.train()
    return avg_loss


def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(trainloader, desc=f"train epoch {epoch}")
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        labels = labels.to(device) if args.conditional else None
        optimizer.zero_grad()

        # Algorithm 1: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, loss_type="l2", classes=labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': loss.item()})

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break
    
    avg_train_loss = total_loss / max(1, n_batches)
    return avg_train_loss


def run_inference_test(args):
    """
    Load a saved model checkpoint and generate images for inference/testing.
    Usage: python ex02_main.py --inference --model_path models/DDPM_30epochs_cosine/ckpt.pt --num_samples 64
    For conditional: python ex02_main.py --inference --conditional --guidance_scale 3.0 --sample_classes "0,1,2,3"
    """
    timesteps = args.timesteps
    image_size = 32
    channels = 3
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {args.model_path}...")
    # Initialize model with conditional support if requested
    if args.conditional:
        model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,),
                    class_free_guidance=True, num_classes=10, p_uncond=args.p_uncond).to(device)
        print(f"Using conditional model with classifier-free guidance (scale={args.guidance_scale})")
    else:
        model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    my_scheduler = lambda x: cosine_beta_schedule(x)
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)
    
    # Prepare class labels if conditional
    sample_classes = None
    guidance_scale = 0.0
    if args.conditional:
        guidance_scale = args.guidance_scale
        if args.sample_classes:
            # Parse comma-separated class indices
            class_list = [int(c.strip()) for c in args.sample_classes.split(',')]
            # Repeat to match num_samples
            class_list = (class_list * (args.num_samples // len(class_list) + 1))[:args.num_samples]
            sample_classes = torch.tensor(class_list, device=device)
            print(f"Sampling from classes: {class_list[:min(10, len(class_list))]}...")
        else:
            # Sample random classes
            sample_classes = torch.randint(0, 10, (args.num_samples,), device=device)
            print(f"Sampling from random CIFAR-10 classes")
    
    # Generate samples
    num_samples = getattr(args, 'num_samples', 64)
    inference_dir = os.path.join(args.save_dir, "inference_samples")
    print(f"Generating {num_samples} samples...")
    sample_and_save_images(num_samples, diffusor, model, device, inference_dir, 
                          image_size=image_size, channels=channels,
                          classes=sample_classes, guidance_scale=guidance_scale)
    print(f"✓ Samples saved to {inference_dir}")


def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    # Initialize model with conditional support if requested
    if args.conditional:
        model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,),
                    class_free_guidance=True, num_classes=10, p_uncond=args.p_uncond).to(device)
        print(f"Training conditional model with classifier-free guidance (p_uncond={args.p_uncond})")
    else:
        model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)
        print("Training unconditional model")
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Initialize loss tracking
    loss_history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'epochs': []
    }

    my_scheduler = lambda x: cosine_beta_schedule(x)  # Using cosine schedule for better quality
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to aid diffusion process
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)

    # create save dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_loss = train(model, trainloader, optimizer, diffusor, epoch, device, args)
        val_loss = test(model, valloader, diffusor, device, args)
        test_loss = test(model, testloader, diffusor, device, args)
        
        # Track losses
        loss_history['train_loss'].append(train_loss)
        loss_history['val_loss'].append(val_loss)
        loss_history['test_loss'].append(test_loss)
        loss_history['epochs'].append(epoch)
        
        print(f"\nEpoch {epoch} Summary - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Test Loss: {test_loss:.6f}\n")
    
    # Save loss history
    import json
    loss_file = os.path.join(args.save_dir, f"loss_history_{args.run_name}.json")
    with open(loss_file, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"\n✓ Loss history saved to {loss_file}")

    # save final model and generate samples
    if args.save_model:
        out_model_dir = os.path.join("./models", args.run_name)
        os.makedirs(out_model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_model_dir, "ckpt.pt"))

    # Generate final samples
    final_classes = None
    final_guidance = 0.0
    if args.conditional:
        # Generate 2 samples per class for CIFAR-10
        final_classes = torch.arange(10, device=device).repeat_interleave(2)[:16]
        final_guidance = args.guidance_scale
    sample_and_save_images(16, diffusor, model, device, args.save_dir, image_size=image_size, channels=channels,
                          classes=final_classes, guidance_scale=final_guidance)


if __name__ == '__main__':
    args = parse_args()

    # --- (2.2) Visualization Capabilities ---
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    if args.inference:
        print("\n➡️ Running inference mode...")
        run_inference_test(args)
    else:
        print("\n➡️ Visualization enabled: generated samples will be saved after each epoch.")
        print(f"➡️ Images will be stored in: {args.save_dir}/samples\n")
        run(args)

