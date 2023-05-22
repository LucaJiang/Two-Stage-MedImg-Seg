# Two-Stage-MedImg-Seg
For MSDM6980


## Usage
### Train
!python train.py -l bce -e 100 -lr 1e-3 -b 4 --load '/path/to/your/model.pth'

-l/--loss: criterion for training, options are:
 'bce': binary cross entropy loss,
 'energy': energy loss,
 'ac': active contour loss,
 'ac2': active contour loss version 2,
 'nll': negative log likelihood loss,
 'lognll': log negative log likelihood loss
 'dice': dice loss,
 'dice_bce': dice loss + binary cross entropy loss
 
optional:
-e/--epochs: number of epochs
-lr/--learning_rate: learning rate
-b/--batch_size: batch size
--load: load model path

### Predict
!python predict.py -m '/path/to/your/model.pth' -i '/path/to/your/image' -im '/path/to/your/mask' -o '/path/to/your/pred_img'