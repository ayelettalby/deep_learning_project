import numpy as np
import matplotlib.pyplot as plt
val_dice_scratch = np.load(r'G:\Deep learning\Datasets_organized\small_dataset\Transfer_exp\Kits\37\exp_3_scratch\dices\val_dice_organ.npy')
val_dice_transfer = np.load(r'G:\Deep learning\Datasets_organized\small_dataset\Transfer_exp\Kits\37\exp_3_transfer\dices\val_dice_organ.npy')

plt.figure()
plt.plot(val_dice_scratch,'b',label='validation dice from scratch')
plt.plot(val_dice_transfer,'r',label='validation dice transfer')
plt.xlabel('epoch number')
plt.ylabel('organ dice')

plt.legend()
plt.show()

