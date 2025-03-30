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

def modify_resnet18_for_cifar(model, dropout_rate = 0.2):
    # Modify the first convolutional layer
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove the max pooling layer
    model.maxpool = nn.Identity()

     # Add dropout after each block
    # This will apply dropout after each major block in the ResNet architecture
    original_forward = model.forward
    
    def create_forward_with_dropout(dropout_rate):
        def new_forward(x):
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            
            x = model.layer1(x)
            x = F.dropout(x, p=dropout_rate, training=model.training)
            
            x = model.layer2(x)
            x = F.dropout(x, p=dropout_rate, training=model.training)
            
            x = model.layer3(x)
            x = F.dropout(x, p=dropout_rate, training=model.training)
            
            x = model.layer4(x)
            x = F.dropout(x, p=dropout_rate, training=model.training)
            
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

    dropout_rate = CONFIG.get("dropout_rate", 0.2)
    model = modify_resnet18_for_cifar(model)  # Modify the first layers

    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),  # Add dropout before the final classification layer
        nn.Linear(num_ftrs, 100)
    )

    return model.to(CONFIG["device"])

################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can printprogress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):

        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # zero the parameter gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() # get the predicted class from the maximum value in the output-list of class scores
        _, predicted = outputs.max(1) # get the index of the max log-probability (predicted class) 

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval() # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # No need to track gradients
        
        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate throught the validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            
            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) # forward pass (prediction) 
            loss = criterion(outputs, labels) # calculate loss 

            running_loss += loss.item() 
            _, predicted = outputs.max(1) # get the predicted class from the maximum value in the output-list of class scores 

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
        "learning_rate": 0.0005, # Use a small learning rate for finetuning since it is a pre-trained model
        "epochs": 60,  # Train for longer in a real scenario - using 20 epochs since resnet18 is a larger model that will take time to converge
        "num_workers": 0, # Adjust based on your system
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",  # Make sure this directory exists
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
        "patience": 10, # Number of epochs to wait for improvement.
        "dropout_rate": 0.15, 
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    #      Data Transformation (Example - You might want to modify) 
    ############################################################################

    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), #add rotation.
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), #add color jitter.
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), # CIFAR-100 stats
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

    ### define loaders and test set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # ... (Create validation and test loaders)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
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
    criterion = nn.CrossEntropyLoss()   
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=5e-4) #Use Adam optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5) 

    # Initialize wandb
    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
    wandb.watch(model)  # watch the model gradients

    ############################################################################
    # --- Training Loop (Example - Students need to complete) ---
    ############################################################################
    best_val_acc = 0.0
    epochs_no_improve = 0 # counter for early stopping.


    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step(val_loss)

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
    import eval_cifar100
    import eval_ood

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()
