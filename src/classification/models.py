import torch
import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

class CustomCellClassifier(nn.Module):
    """
    A simple Convolutional Neural Network for cell classification.
    Supports both binary (cancerous/non-cancerous) and ternary (cancerous/non-cancerous/false-positive).
    Designed for small input images (e.g., 64x64, 96x96).
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 1, input_image_size: int = 96):
        super().__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (N, 32, H/2, W/2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (N, 64, H/4, W/4)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (N, 128, H/8, W/8)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: (N, 256, H/16, W/16)
        )
        
        # Dynamically determine the size of the flattened features
        self.flattened_features = self._get_flattened_features_size(input_channels, input_image_size)

        # Enhanced classifier for better performance
        if num_classes == 1:  # Binary classification
            self.classifier = nn.Sequential(
                nn.Linear(self.flattened_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes) # BCEWithLogitsLoss
            )
        else:  # Multi-class classification
            self.classifier = nn.Sequential(
                nn.Linear(self.flattened_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes) # CrossEntropyLoss
            )

    def _get_flattened_features_size(self, input_channels: int, input_image_size: int) -> int:
        """Helper to calculate the input size for the linear layer by passing a dummy input."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_image_size, input_image_size)
            dummy_output = self.features(dummy_input)
            return dummy_output.view(dummy_output.size(0), -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x

def get_classification_model(
    model_name: str, 
    pretrained: bool = True, 
    num_classes: int = 1, 
    input_channels: int = 3,
    input_image_size: int = 96, # Required for custom_cnn to calculate flattened size
    classification_mode: str = "binary"  # Add classification mode context
) -> nn.Module:
    """
    Returns a classification model based on the specified name.
    
    Args:
        model_name (str): Name of the model (e.g., 'resnet18', 'custom_cnn').
        pretrained (bool): Whether to use pretrained weights (for torchvision models).
        num_classes (int): Number of output classes (1 for binary, 3 for ternary).
        input_channels (int): Number of input channels (e.g., 3 for RGB).
        input_image_size (int): The square size of the input images (e.g., 96x96).
        classification_mode (str): Classification mode for logging purposes.
        
    Returns:
        nn.Module: The instantiated classification model.
    """
    if model_name == "custom_cnn":
        logger.info(f"Creating CustomCellClassifier for {classification_mode} classification with "
                   f"{input_channels} input channels, {num_classes} output classes, and image size {input_image_size}.")
        return CustomCellClassifier(input_channels=input_channels, num_classes=num_classes, input_image_size=input_image_size)
    
    elif model_name.startswith("resnet"):
        if model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
            
        # Modify the first conv layer if input channels are not 3 (e.g., grayscale)
        if input_channels != 3:
            model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            logger.info(f"Modified ResNet conv1 for {input_channels} input channels.")
            
        # Modify the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        logger.info(f"Created {model_name} model for {classification_mode} classification with "
                   f"pretrained={pretrained} and {num_classes} output class(es).")
        return model
    
    # Example for other torchvision models
    # elif model_name.startswith("vgg"):
    #     if model_name == "vgg11":
    #         model = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1 if pretrained else None)
    #     else:
    #         raise ValueError(f"Unsupported VGG model: {model_name}")
    #     # VGG classifier is a Sequential module, need to modify the last linear layer
    #     num_ftrs = model.classifier[6].in_features
    #     model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    #     logger.info(f"Created {model_name} model for {classification_mode} classification with "
    #                f"pretrained={pretrained} and {num_classes} output class(es).")
    #     return model
    
    else:
        raise ValueError(f"Unsupported model name: {model_name}")