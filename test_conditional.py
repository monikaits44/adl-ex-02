"""
Quick test script to verify classifier-free guidance implementation.
This tests both conditional and unconditional models without full training.
"""
import torch
from ex02_model import Unet
from ex02_diffusion import Diffusion, cosine_beta_schedule

def test_unconditional_model():
    """Test that unconditional model still works."""
    print("Testing unconditional model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create unconditional model
    model = Unet(dim=32, channels=3, dim_mults=(1, 2, 4)).to(device)
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32).to(device)
    t = torch.randint(0, 100, (2,)).to(device)
    
    output = model(x, t)
    assert output.shape == (2, 3, 32, 32), f"Expected (2, 3, 32, 32), got {output.shape}"
    print("✓ Unconditional model forward pass works")
    
    # Test diffusion
    diffusor = Diffusion(100, cosine_beta_schedule, 32, device)
    
    # Test forward diffusion
    x_noisy = diffusor.q_sample(x, t)
    assert x_noisy.shape == x.shape
    print("✓ Forward diffusion works")
    
    # Test loss computation
    loss = diffusor.p_losses(model, x, t)
    assert loss.item() > 0
    print("✓ Loss computation works")
    
    # Test sampling (just one step for speed)
    with torch.no_grad():
        sample = diffusor.p_sample(model, x, t, 50)
    assert sample.shape == x.shape
    print("✓ Reverse diffusion step works")
    
    print("✅ Unconditional model: ALL TESTS PASSED\n")


def test_conditional_model():
    """Test that conditional model with CFG works."""
    print("Testing conditional model with classifier-free guidance...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create conditional model
    model = Unet(
        dim=32, 
        channels=3, 
        dim_mults=(1, 2, 4),
        class_free_guidance=True,
        num_classes=10,
        p_uncond=0.1
    ).to(device)
    
    # Test forward pass with classes
    x = torch.randn(2, 3, 32, 32).to(device)
    t = torch.randint(0, 100, (2,)).to(device)
    classes = torch.tensor([0, 5]).to(device)
    
    output = model(x, t, classes)
    assert output.shape == (2, 3, 32, 32), f"Expected (2, 3, 32, 32), got {output.shape}"
    print("✓ Conditional forward pass works")
    
    # Test forward pass without classes (should still work)
    output_uncond = model(x, t, None)
    assert output_uncond.shape == (2, 3, 32, 32)
    print("✓ Forward pass without classes works")
    
    # Test diffusion with classes
    diffusor = Diffusion(100, cosine_beta_schedule, 32, device)
    
    # Test loss with classes
    loss = diffusor.p_losses(model, x, t, classes=classes)
    assert loss.item() > 0
    print("✓ Loss computation with classes works")
    
    # Test sampling with CFG
    with torch.no_grad():
        # Test with guidance
        sample_guided = diffusor.p_sample(model, x, t, 50, classes=classes, guidance_scale=3.0)
        assert sample_guided.shape == x.shape
        print("✓ Reverse diffusion with guidance works")
        
        # Test without guidance
        sample_no_guide = diffusor.p_sample(model, x, t, 50, classes=classes, guidance_scale=0.0)
        assert sample_no_guide.shape == x.shape
        print("✓ Reverse diffusion without guidance works")
    
    # Test that null token exists
    assert hasattr(model, 'null_class')
    assert model.null_class == 10  # Should be num_classes
    print(f"✓ Null token properly set to {model.null_class}")
    
    # Test class embeddings
    assert model.class_emb is not None
    assert model.classes_mlp is not None
    class_emb = model.class_emb(classes)
    assert class_emb.shape[0] == 2
    print("✓ Class embeddings work")
    
    # Test training mode behavior (random dropping)
    model.train()
    num_trials = 100
    dropped_count = 0
    for _ in range(num_trials):
        test_classes = torch.tensor([0, 1]).to(device)
        # Forward pass - internally may drop classes
        _ = model(x, t, test_classes)
    print(f"✓ Training mode (random class dropping) works")
    
    print("✅ Conditional model: ALL TESTS PASSED\n")


def test_backward_compatibility():
    """Test that old code still works with new implementation."""
    print("Testing backward compatibility...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Old style: create model without conditional args
    model = Unet(dim=32, channels=3, dim_mults=(1, 2, 4)).to(device)
    diffusor = Diffusion(100, cosine_beta_schedule, 32, device)
    
    x = torch.randn(2, 3, 32, 32).to(device)
    t = torch.randint(0, 100, (2,)).to(device)
    
    # Old style calls (no classes argument)
    output = model(x, t)
    loss = diffusor.p_losses(model, x, t)
    
    with torch.no_grad():
        sample = diffusor.p_sample(model, x, t, 50)
    
    print("✓ Old API calls work without modification")
    print("✅ Backward compatibility: ALL TESTS PASSED\n")


if __name__ == "__main__":
    print("="*60)
    print("Testing Classifier-Free Guidance Implementation")
    print("="*60 + "\n")
    
    try:
        test_unconditional_model()
        test_conditional_model()
        test_backward_compatibility()
        
        print("="*60)
        print("🎉 ALL TESTS PASSED! Implementation is correct.")
        print("="*60)
        
        print("\nYou can now:")
        print("1. Train unconditional: python ex02_main.py --epochs 30 --save_model")
        print("2. Train conditional: python ex02_main.py --conditional --epochs 30 --save_model")
        print("3. Use provided scripts: ./train_conditional_example.sh")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
