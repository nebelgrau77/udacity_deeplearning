import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        # this model is based on the VGG architecture
        
        # 8 convolutional layers
        
        self.num_classes = num_classes
        self.dropout = dropout
                
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),            
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),            
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2, stride=2),  # image size 112 x 112           
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # image size 56 x 56         
            nn.Dropout(self.dropout)
        )
            
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2, stride=2), # image size 28 x 28         
            nn.Dropout(self.dropout)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2, stride=2), # image size 14 x 14        
            nn.Dropout(self.dropout)
        )
  
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2, stride=2), # image size 7 x 7        
            nn.Dropout(self.dropout)
        )
        
        # 2 fully connected, smaller layers, following one of the suggestions in the forums
        
        self.fc_layers = nn.Sequential(            
            nn.Dropout(self.dropout),
            # nn.Linear(in_features=512*7*7, out_features=512),
            nn.Linear(in_features=512*14*14, out_features=4096),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            #nn.Linear(4096,4096),
            #nn.ReLU(),            
            nn.Linear(4096, self.num_classes))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        
        # flatten for the fully connected layers
        
        x = x.view(x.size(0), -1)
        
        x = self.fc_layers(x)
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"

    