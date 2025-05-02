# DLAV Project Submission Part 1
**ADEBeastsLLC** <br>
 Rafael Garcia Bustillos and Jeffrey Yu

## Inputs: 

```python
camera: ndarray (200, 300, 3)      # RGB image

sdc_history_feature: ndarray (21, 3)    # [x, y, heading] from past 21 steps
```
## Outputs:
```python
sdc_future_feature: ndarray (60, 3),      # [x, y, heading] for the next 60 steps
```

## Structure

We designed the architecture with three main components:
1. **CNN Image feature extractor** <br>
    We use a custom CNN architecture to extract information from the image. This consist of multiple Conv2d layers with a stride of 2 to learn features based on different sizes of the image.  We also use a MaxPool function to ensure that the embedding space at the output of the CNN is not too large. We then flatten and use 2 linear layers at the end. This outputs a vector of size 256.
2. **History feature extractor** <br>
    We use a 4 layer MLP to learn relationships between the past locations of the model. This outputs a vector of size 128.
3. **Decoder** <br>
    The decoder concatenates the outputs of the history feature extractor and then uses 2 hidden layers and then outputs a final prediction, which is a vector of size 180. This is then reshaped at the end of the forward pass to the desired shape of [60, 3].

## Other Changes
We used a custom loss criterion because we wanted scale the weight of the heading loss by 4 times. This is becuase the maximum value of heading is less than the maximal displacements of the x and y positions. 

## Training Configuration 
We trained using the below configurations:


- Optimizer: Adam
    - 1st stage: ```lr = 1e-3```, and ```weight_decay = 1e-4``` for 70 epochs
    - 2nd stage: ```lr = 1e-4```, and ```weight_decay = 1e-4``` for 70 epochs
- Batch size: 32

Note: We save the best performing model while training, using the criterion of lowest ADE on the validation. This may may not correspond to the final model found at the end of training. 

## Code Instructions
The entire code is found under the file ```DLAV-Phase1_final.ipynb```. We have included headers for the sections of the code. The code can be run consecutively, but if run all at once, then the model will be trained again. To avoid this, skip to the section labeled **Load/Save** and set the value ```load_model = True``` to load in the pretrained weights. We have already set this to be the case.

### Training
To train the model, run the code sections ```Set-Up``` and under the heading ```Training```. This includes the data loading, model creation, logger, and training loop.

### Run 
To run the model and visualize on the validation set, run the code sections under the heading ```Validation```

### Infer  
To generate the ```.csv``` file for submissions run the section under ```Generate Submission```.

## Additional Notes
We attempted to train with a GRU, but was not able to get and ADE below ~2.5.