# Hand/arm segmentation model

`hand_segmentation.py` can run a binary TensorFlow Lite model:

```powershell
python hand_segmentation.py --segmentation-model models/hand_arm.tflite
```

The model must accept an RGB image with shape `[1, height, width, 3]` and
return either one foreground-probability channel or classes with foreground at
channel 1. It is run only inside the MediaPipe hand/forearm region.

## Data required

Each training image needs a same-size binary mask:

- `0`: background
- `1`: every visible hand and forearm pixel

Hand-only masks are not enough if arms must be red. Capture and label webcam
frames that cover your expected lighting, skin tones, sleeves, backgrounds,
distances, rotations, and both hands. Existing hand-mask datasets can provide
an initial model, but add your own forearm labels before deployment.

## Training and export

1. Resize image/mask pairs to a fixed size such as 256x256.
2. Train a compact U-Net or MobileNet-U-Net with binary cross-entropy plus
   Dice loss; measure IoU/Dice on held-out frames from your Pixel camera.
3. Augment with rotation, scale, brightness, blur, and background changes.
4. Export the trained Keras model to TensorFlow Lite, preferably with
   float16 or int8 quantization.
5. Check one frame in this module before measuring real-time performance.

The linked article's model can be used only after verifying its input shape,
output class order, and whether its labels include forearms.
