import streamlit as st
import torch
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.src.gan_trainer import ArtGANTrainer

st.set_page_config(page_title="DCGAN Image Generator", layout="wide")

st.title("🎨 DCGAN Art Generator")

# Charger le modèle (avec cache)
@st.cache_resource
def load_model():
    checkpoint = torch.load("checkpoint_epoch_200.pth", map_location='cpu')
    
    trainer = ArtGANTrainer(
        mode="dcgan",
        image_size=checkpoint['image_size'],
        channels=checkpoint['channels'],
        latent_dim=checkpoint['latent_dim'],
        device="cpu"
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer.generator.load_state_dict(checkpoint['generator_state'])
    trainer.generator = trainer.generator.to(device)
    trainer.generator.eval()
    
    return trainer.generator, checkpoint['latent_dim'], device

generator, latent_dim, device = load_model()

# Sidebar avec slider et bouton
num_images = st.sidebar.slider("Number of images to generate", 1, 64, 16)

if st.sidebar.button("🎲 Generate Images", type="primary", use_container_width=True):
    with st.spinner("Generating images..."):
        with torch.no_grad():
            noise = torch.randn(num_images, latent_dim, 1, 1).to(device)
            fake_images = generator(noise)
            
            # Normaliser de [-1, 1] à [0, 1]
            fake_images = (fake_images + 1) / 2
            fake_images = fake_images.clamp(0, 1)
            
            # Afficher en grille
            cols = 4
            rows = (num_images + cols - 1) // cols
            
            for row in range(rows):
                cols_streamlit = st.columns(cols)
                for col_idx in range(cols):
                    img_idx = row * cols + col_idx
                    if img_idx < num_images:
                        img_tensor = fake_images[img_idx]
                        img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
                        img_pil = Image.fromarray((img_np * 255).astype('uint8'))
                        cols_streamlit[col_idx].image(img_pil, use_container_width=True)
            
            st.success(f"✨ Generated {num_images} images!")