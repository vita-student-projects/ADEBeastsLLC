# DLAV Project Submission Part 3
**ADEBeastsLLC** <br>
 Rafael Garcia Bustillos (377114) and Jeffrey Yu (371327)

## Task

The goal of the third milestone was to create a model to predict future car trajectories using egocentric images taken from real driving situations and simulated egocentric images. In the following sections, we will describe the inputs, model structure, loss function design, training configuration, and instructions to run the code.

## Inputs: 

```python
camera: ndarray (200, 300, 3)      # RGB image

sdc_history_feature: ndarray (21, 3)    # [x, y, heading] from past 21 steps
```
## Outputs:
```python
sdc_future_feature: ndarray (60, 3),      # [x, y, heading] for the next 60 steps
```

## Structure

We designed the architecture with 3 main components:
1. **Pretrained Resnet34 Image feature extractor** <br>
 We use a pre-trained Resnet34 model with weights from the Image1K dataset. We want to use a robust and high-performing model to extract image features. We cropped and interpolated the input RGB image, semantic map, and depth map to (224x224) to respect the ResNet input image architecture. To adjust the model to our dataset, we chose to retrain the last ResNet block, letting the model learn the high-level features relevant to our dataset.
2. **History feature extractor** <br>
 We use a 4-layer fully connected MLP to learn relationships between the past locations of the model. This outputs a vector of size 128.
3. **Future Decoder** <br>
 To predict the future position and heading, we utilize a GRU and a 2-layered fully connected MLP.
 The hidden state of the GRU is initialized by concatenating the image and history feature extractor results; the input state is initialized as the last position in `sdc_history_feture`.
 A 60-step for loop is created, where the current position is encoded with the GRU to a vector of size 384, then the MLP converts it to a vector of size 3 for x-position, y-position, and heading. The results are saved on a list and concatenated at the end, to the shape of [batch, 60, 3].


## Loss Function Design
We used a custom loss criterion for the future trajectories. We scale the L2 loss of x, y, and heading differently. We chose to scale the x- and y-position by 2, to increase the loss value of the model slightly. Next, we scaled the heading loss by 8 times to compensate for the magnitude of the difference between the maximum value of the heading and the maximum displacements of the x and y positions.

## Training Configuration 
The model was trained on an HPC cluster with a V100 GPU.

We trained using the following configurations:

- Optimizer: Adam   
- Learning Rate Scheduler: Cosine annealing with a starting learning rate of 8e-4 and ending learning rate of 5e-5.
- Epochs: 150
- Batch size: 32

During the training loop, for every batch set, 1 random image transformation would be applied to all the images in a batch.
The transformations we chose are:
- **None**: We apply no visual transforms.
- **ColorJitter**: For the model to better extract the geometrical features of the image and not be too affected by the object's colors. Because in the end, not all cars, roads, or buildings will be of the same color.
- **RandomPerspective**: To prepare the model for small variations in perspective.
- **RandomResizedCrop**: Zooms into cropped regions of the image and resizes them to [224,224]. Let the model look at different content in the same image.
- **GaussianBlur**: To prepare the model for different image qualities, or weather conditions with poor visibility(i.e. rain or fog covering the camera's view)
- **ElasticTransform**: Prepares the model for non-sharp images (i.e. in rainy conditions)
- **RandomAffine**: Applies a small horizontal translation to the image to improve the module's robustness in varying camera angles. (i.e. car on left lane vs. right lane)

Note: We save the best-performing model while training, using the criterion of lowest ADE on the validation. This may not correspond to the final model found at the end of training.

## Code Structure and Instructions
The code is comprised of multiple files structured in the manner below:

```
DLAV/
├── main.py # Entry point for training
├── analyze.py # loads a model, visualizes results, and calculates validation ADE
├── kaggle.py # prepares csv for kaggle
├── vis_data.py # visualizes the input data 
├── Weights_V3.pth # Trained weights from Milestone 3 with utils.model.DrivingPlanner_GRU
├── utils/
│ |── dataset.py # dataset and data loader for nuPlan
| ├── evaluate.py # helper functions for validation of the model
│ ├── logger.py # logging class
│ ├── model.py # model architecture definition
│ └── train.py # training functions
```
The weight for Milestone 3 can be found at [this link](https://drive.google.com/file/d/1NP9nO1XP8swJ3fl0U50-KtdU88IxAlQd/view?usp=drive_link)
### Environment Config
Install the required packages using the command: ```pip install -r requirements.txt```

### Training
To train the model, run the ```main.py``` file. The training hyperparameters are modified within the main file.

### Visualization and Validation 
To run the model and visualize the validation set, run the code ```validate.py```. You need to modify the line to load in the desired weights. Below is an image of the visualization of the model outputs.

### Infer  
To generate the ```.csv``` file for submissions, run the file ```kaggle.py```.
