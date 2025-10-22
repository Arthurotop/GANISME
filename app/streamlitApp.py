import streamlit as st
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from src.gan_trainer import ArtGANTrainer
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# === CONFIGURATION DE L'APPLICATION ===
st.set_page_config(page_title="ArtGAN Generator", layout="centered")
st.title("Générateur d'images avec GAN")

# === CHARGER LE MODÈLE ===
@st.cache_resource
def load_trainer():
    trainer = ArtGANTrainer.from_checkpoint("model/checkpoints/dcgan_64.pt",
device="cpu")
    return trainer

trainer = load_trainer()

# === PARAMÈTRES UTILISATEUR ===
num_samples = st.slider("Nombre d'images à générer", 1, 16, 9)
display_mode = st.radio("Mode d'affichage :", ["Grille", "Individuelles"])

# === GÉNÉRATION ===
if st.button("🖼️ Générer"):
    with st.spinner("Génération des images en cours..."):
        images = trainer.generate(num_samples=num_samples, denormalize=True)

        if display_mode == "Grille":
            nrow = int(num_samples ** 0.5) or 1
            grid = make_grid(images, nrow=nrow)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(grid.permute(1, 2, 0).numpy())
            ax.axis("off")
            st.pyplot(fig)

        else:
            for img in images:
                fig, ax = plt.subplots()
                ax.imshow(img.permute(1, 2, 0).numpy())
                ax.axis("off")
                st.pyplot(fig)