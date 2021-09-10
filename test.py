from models.wrapper import CLIPFinetuningWrapper
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule

print("Loading CLIP...")
model = CLIPFinetuningWrapper("ViT-B/32")
print("Loading Data...")
dm = TextImageDataModule(folder='./data_dir',batch_size=1)

trainer = Trainer(
    max_epochs=1
)

trainer.fit(model,dm)