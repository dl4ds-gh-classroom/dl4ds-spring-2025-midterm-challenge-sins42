import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json

################################################################################
# Model Definition (Simple Example - You need to complete)
# For Part 1, you need to manually define a network.
# For Part 2 you have the option of using a predefined network and
# for Part 3 you have the option of using a predefined, pretrained network to
# finetune.
################################################################################

def modify_resnet18_for_cifar(model, dropout_rate=0.3):
    # Modify the first convolutional layer
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove the max pooling layer
    model.maxpool = nn.Identity()

    # Disable inplace operations for all ReLU layers
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    # Add dropout after each block with spatial dropout for more effective regularization
    original_forward = model.forward
    
    def create_forward_with_dropout(dropout_rate):
        def new_forward(x):
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            
            x = model.layer1(x)
            x = F.dropout2d(x, p=dropout_rate/2, training=model.training)  # Spatial dropout
            
            x = model.layer2(x)
            x = F.dropout2d(x, p=dropout_rate/2, training=model.training)
            
            x = model.layer3(x)
            x = F.dropout2d(x, p=dropout_rate, training=model.training)
            
            x = model.layer4(x)
            x = F.dropout2d(x, p=dropout_rate, training=model.training)
            
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            x = model.fc(x)
            
            return x
        return new_forward
        
    model.forward = create_forward_with_dropout(dropout_rate)

    return model

def load_pretrained_resnet18(CONFIG):
    model = models.resnet18(pretrained=True)

    # Disable inplace operations for all ReLU layers
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    dropout_rate = CONFIG.get("dropout_rate", 0.3)
    model = modify_resnet18_for_cifar(model)  # Modify the first layers

    # Improved classifier head with multiple layers
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, 100)
    )

    return model.to(CONFIG["device"])

################################################################################
# Mixup Augmentation Functions
################################################################################
def mixup_data(x, y, alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch with mixup augmentation."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # Use mixup if configured
    use_mixup = CONFIG.get("use_mixup", True)
    mixup_alpha = CONFIG.get("mixup_alpha", 0.4)

    # put the trainloader iterator in a tqdm so it can print progress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):

        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        # Apply mixup if enabled
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha)
        
        optimizer.zero_grad()  # zero the parameter gradients
        outputs = model(inputs)
        
        # Calculate loss (with or without mixup)
        if use_mixup:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, labels)
            
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Step the learning rate scheduler if using OneCycleLR
        if isinstance(CONFIG.get("scheduler"), optim.lr_scheduler.OneCycleLR):
            CONFIG["scheduler"].step()

        running_loss += loss.item()
        
        # For accuracy calculation (handle mixup case)
        if use_mixup:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            # For mixup, calculate weighted accuracy based on the mixup weight
            correct += (lam * predicted.eq(targets_a).float() + 
                        (1 - lam) * predicted.eq(targets_b).float()).sum().item()
        else:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device,  use_tta=False, num_augments=5):
    """Validate the model with optional Test-Time Augmentation."""
    model.eval()  # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients
        
        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate through the validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            
            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)

            # Standard forward pass
            outputs = model(inputs)
            
            # If using TTA
            if use_tta:
                # Add horizontal flip augmentation
                inputs_flip = torch.flip(inputs, dims=[3])  # Horizontal flip
                outputs_flip = model(inputs_flip)
                
                # Average predictions
                outputs = (outputs + outputs_flip) / 2
            
            loss = criterion(outputs, labels)  # calculate loss 

            running_loss += loss.item() 
            _, predicted = outputs.max(1)  # get predicted class

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():

    ############################################################################
    #    Configuration Dictionary (Modify as needed)
    ############################################################################
    # It's convenient to put all the configuration in a dictionary so that we have
    # one place to change the configuration.
    # It's also convenient to pass to our experiment tracking tool.


    CONFIG = {
        "model": "ResNet18",   # Change name when using a different model
        "batch_size": 512, # run batch size finder to find optimal batch size - this is optimal batch size 
        "learning_rate": 0.001, # Use a small learning rate for finetuning since it is a pre-trained model
        "epochs": 60,  # Train for longer in a real scenario - using 20 epochs since resnet18 is a larger model that will take time to converge
        "num_workers": 0, # Adjust based on your system
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",  # Make sure this directory exists
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
        "patience": 10, # Number of epochs to wait for improvement.
        "dropout_rate": 0.3,  # Increased dropout
        "use_mixup": True,  # Enable mixup augmentation
        "mixup_alpha": 0.4,  # Mixup strength
        "use_tta": True,  # Test-time augmentation
        "label_smoothing": 0.1,  # Label smoothing factor
        "weight_decay": 1e-4,  # Weight decay for regularization
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    #      Data Transformation (Example - You might want to modify) 
    ############################################################################

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), #add rotation.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Increased intensity
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add small translations
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Add perspective changes
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), # CIFAR-100 stats
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),  # Add random erasing
    ])

    ###############
    # validation and test transforms - NO augmentation for validation/test
    ###############

    # Validation and test transforms (NO augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), # CIFAR-100 stats
    ])

    ############################################################################
    #       Data Loading
    ############################################################################

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)

    # Split train into train and validation (80/20 split)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    # Apply test transform to validation set
    valset.dataset.transform = transform_test

    ### define loaders and test set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)  # Added pin_memory for faster data transfer
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"],  pin_memory=True)

    # ... (Create validation and test loaders)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"],  pin_memory=True)
    
    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    model = load_pretrained_resnet18(CONFIG)   # instantiate your model 
    model = model.to(CONFIG["device"])   # move it to target device

    print("\nModel summary:")
    print(f"{model}\n")

    # The following code you can run once to find the batch size that gives you the fastest throughput.
    # You only have to do this once for each machine you use, then you can just
    # set it in CONFIG.
    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        from utils import find_optimal_batch_size
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(model, trainset, CONFIG["device"], CONFIG["num_workers"])
        CONFIG["batch_size"] = optimal_batch_size
        print(f"Using batch size: {CONFIG['batch_size']}")
    

    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])   
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]) 
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["learning_rate"],
        epochs=CONFIG["epochs"],
        steps_per_epoch=len(trainloader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'  # Cosine annealing
    )

    CONFIG["scheduler"] = scheduler  # Store scheduler in CONFIG for access in train function

    # Initialize wandb
    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
    wandb.watch(model)  # watch the model gradients

    ############################################################################
    # --- Training Loop (Example - Students need to complete) ---
    ############################################################################
    best_val_acc = 0.0
    epochs_no_improve = 0 # counter for early stopping.


    for epoch in range(CONFIG["epochs"]):
        # Train with mixup
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)

        # Validate with TTA if configured
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"], use_tta=CONFIG["use_tta"])
        
        # Only step scheduler if not OneCycleLR (which steps every batch)
        if not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"] # Log learning rate
        })

        # Save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth") # Save to wandb as well
            epochs_no_improve = 0 # reset counter.
        else:
            epochs_no_improve += 1 #Increment counter.

        if epochs_no_improve == CONFIG["patience"]: #Early stopping condition.
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break # Stop training.

    wandb.finish()

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################
    # Load the best model
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load("best_model.pth"))

    import eval_cifar100
    import eval_ood

    # --- Evaluation on Clean CIFAR-100 Test Set with TTA ---
    print("Evaluating on CIFAR-100 test set...")
    if CONFIG["use_tta"]:
        # Create custom evaluate function with TTA
        def evaluate_with_tta(model, dataloader, device):
            model.eval()
            correct = 0
            total = 0
            all_predictions = []
            
            with torch.no_grad():
                for data in tqdm(dataloader, desc="Evaluating with TTA"):
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    
                    # Original prediction
                    outputs = model(images)
                    
                    # Flipped prediction
                    flipped = torch.flip(images, dims=[3])
                    outputs_flip = model(flipped)
                    
                    # Average predictions
                    outputs = (outputs + outputs_flip) / 2
                    
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
            return all_predictions, 100 * correct / total
        
        predictions, clean_accuracy = evaluate_with_tta(model, testloader, CONFIG["device"])
    else:
        predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD with TTA ---
    if CONFIG["use_tta"]:
        # Monkey patch the eval_ood module's evaluation function to use TTA
        original_evaluate = eval_ood.evaluate_single_image
        
        def evaluate_with_tta_single(model, image_tensor, device):
            model.eval()
            with torch.no_grad():
                # Original prediction
                image_tensor = image_tensor.to(device)
                outputs = model(image_tensor)
                
                # Flipped prediction
                flipped = torch.flip(image_tensor, dims=[3])
                outputs_flip = model(flipped)
                
                # Average predictions
                outputs = (outputs + outputs_flip) / 2
                
                _, predicted = torch.max(outputs, 1)
                return predicted.item()
        
        # Replace the evaluation function temporarily
        eval_ood.evaluate_single_image = evaluate_with_tta_single
    
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    
    # Restore original function if modified
    if CONFIG["use_tta"]:
        eval_ood.evaluate_single_image = original_evaluate

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()