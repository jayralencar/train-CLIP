from data.text_image_dm import TextImageDataModule

dm = TextImageDataModule(folder="./data_dir",batch_size= 1)

dm.setup()

print(len(dm.train_dataloader()))