from data.text_image_dm import TextImageDataModule

dm = TextImageDataModule(folder="./data_dir",batch_size= 1)

dm.setup()

print('Train')
print(len(dm.train_dataloader()))
print('Validation')
print(len(dm.val_dataloader()))