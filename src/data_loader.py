import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    return train_transform, val_test_transform


def get_dataloaders(data_dir, batch_size=32, img_size=224):
    train_transform, val_test_transform = get_transforms(img_size)

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=val_test_transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=val_test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names