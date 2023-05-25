# Two-Stage-MedImg-Seg
For MSDM6980

present link: https://docs.google.com/presentation/d/1Lzmad5BwLLHfSBbIZjy2d3KWjs6y3EwRXAVD71ZHRoE/edit?usp=sharing

## Usage
### Train
!python train.py -l bce -e 100 -lr 1e-3 -b 4 --load '/path/to/your/model.pth'

-l/--loss: criterion for training, options are:
 'bce': binary cross entropy loss,
 'energy': energy loss,
 'ac': active contour loss,
 'nll': negative log likelihood loss,
 'dice': dice loss,
 'dice_bce': dice loss + binary cross entropy loss
 
optional:
-e/--epochs: number of epochs
-lr/--learning_rate: learning rate
-b/--batch_size: batch size
--load: load model path

### Predict
!python predict.py -m '/path/to/your/model.pth' -i '/path/to/your/image' -im '/path/to/your/mask' -o '/path/to/your/pred_img'

### Fixed
* In energy loss: update torch version from 1.7.0- to 2.0+.
* In energy loss: 