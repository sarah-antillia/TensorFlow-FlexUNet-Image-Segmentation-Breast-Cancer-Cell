<h2>TensorFlow-FlexUNet-Image-Segmentation-Breast-Cancer-Cell (2025/06/10)</h2>

This is the first experiment of Image Segmentation for Breast-Cancer-Cell (Benign and Malignant)
 based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>) and an
 <a href="https://drive.google.com/file/d/18MuYY-Kfx1o3OyWCRjjtkQdf5UGU5vBm/view?usp=sharing">
Augmented-Breast-Cancer-Cell-ImageMask-Dataset.zip</a>, which was derived by us from the 
<a href="https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation">
Breast Cancer Cell Segmentation (58 histopathological images with expert annotations)
</a>
<br>
<hr>
<b>Actual Image Segmentation for 512x512 Breast-Cancer-Cell Dataset</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>
In the following predicted mask images, green regions indicate benign areas, while red regions indicate malignant tumors.<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/images/5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/masks/5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test_output/5.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/images/barrdistorted_1001_0.3_0.3_33.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/masks/barrdistorted_1001_0.3_0.3_33.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test_output/barrdistorted_1001_0.3_0.3_33.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/images/barrdistorted_1003_0.3_0.3_40.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/masks/barrdistorted_1003_0.3_0.3_40.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test_output/barrdistorted_1003_0.3_0.3_40.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<b>1. Dataset Citation</b><br>
We used the following dataset:<br>
<a href="https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation">
Breast Cancer Cell Segmentation (58 histopathological images with expert annotations)
</a>
<br><br>
<b>About Dataset</b><br>
In this dataset, there are 58 H&E stained histopathology images used in breast cancer cell <br
detection with associated ground truth data available. Routine histology uses the stain <br>b
combination of hematoxylin and eosin, commonly referred to as H&E. These images are stained<br>
 since most cells are essentially transparent, with little or no intrinsic pigment. Certain <br>
 special stains, which bind selectively to particular components, are be used to identify <br>
 biological structures such as cells. In those images, the challenging problem is cell segmentation<br>
 for subsequent classification in benign and malignant cells.<br>
<br>
<b>How to Cite this Dataset</b><br>
If you use this dataset in your research, please credit the authors.<br>
<br>
<b>Original Article</b><br>
E. Drelie Gelasca, J. Byun, B. Obara and B. S. Manjunath, <br>
"Evaluation and benchmark for biological image segmentation,"<br>
 2008 15th IEEE International Conference on Image Processing, <br>
 San Diego, CA, 2008, pp. 1816-1819<br>.
<br>
BibTeX<br>
@inproceedings{Drelie08-298,<br>
author = {Elisa Drelie Gelasca and Jiyun Byun and Boguslaw Obara and B.S. Manjunath},<br>
title = {Evaluation and Benchmark for Biological Image Segmentation},<br>
booktitle = {IEEE International Conference on Image Processing},<br>
location = {San Diego, CA},<br>
month = {Oct},<br>
year = {2008},<br>
url = {http://vision.ece.ucsb.edu/publications/elisa_ICIP08.pdf}}<br>

<br>


<h3>
<a id="2">
2 Breast-Cancer-Cell ImageMask Dataset
</a>
</h3>
 If you would like to train this Breast-Cancer-Cell Segmentation model by yourself,
 please download the dataset from the google drive  
 <a href="https://drive.google.com/file/d/18MuYY-Kfx1o3OyWCRjjtkQdf5UGU5vBm/view?usp=sharing">
Augmented-Breast-Cancer-Cell-ImageMask-Dataset.zip</a>
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Breast-Cancer-Cell
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Breast-Cancer-Cell Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/Breast-Cancer-Cell_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not so large to use for the
 training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Breast-Cancer-Cell TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Breast-Cancer-Cell and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 3

base_filters   = 16
base_kernels   = (7,7)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
Specifed rgb color map dict for Breast-Cancer-Cell 8 classes.<br>
<pre>
[mask]
mask_datatyoe    = "categorized"
mask_file_format = ".png"
;Breast-Cancer-Cell rgb color map dict for 1+2 classes.
;        background:black , Benign:green  Malignant: red
rgb_map = {(0,0,0):0,(0,255,0):1, (255,0,0):2 }
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<br> 
As shown below, early in the model training, the predicted masks from our UNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br> 
<br>
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 20,21,22)</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/asset/epoch_change_infer_at_middle.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 42,43,44)</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 44 by EarlyStopping callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/asset/train_console_output_at_epoch44.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Breast-Cancer-Cell</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Breast-Cancer-Cell.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/asset/evaluate_console_output_at_epoch44.png" width="920" height="auto">
<br><br>Image-Segmentation-Breast-Cancer-Cell

<a href="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Breast-Cancer-Cell/test was very low, and dice_coef_multiclass 
was very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0395
dice_coef_multiclass,0.9815
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Breast-Cancer-Cell</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for Breast-Cancer-Cell.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/images/barrdistorted_1001_0.3_0.3_14.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/masks/barrdistorted_1001_0.3_0.3_14.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test_output/barrdistorted_1001_0.3_0.3_14.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/images/barrdistorted_1003_0.3_0.3_20.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/masks/barrdistorted_1003_0.3_0.3_20.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test_output/barrdistorted_1003_0.3_0.3_20.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/images/distorted_0.02_rsigma0.5_sigma40_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/masks/distorted_0.02_rsigma0.5_sigma40_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test_output/distorted_0.02_rsigma0.5_sigma40_8.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/images/50.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/masks/50.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test_output/50.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/images/barrdistorted_1001_0.3_0.3_33.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/masks/barrdistorted_1001_0.3_0.3_33.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test_output/barrdistorted_1001_0.3_0.3_33.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/images/deformed_alpha_1300_sigmoid_8_40.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test/masks/deformed_alpha_1300_sigmoid_8_40.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Breast-Cancer-Cell/mini_test_output/deformed_alpha_1300_sigmoid_8_40.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Breast Cancer Cell Segmentation (58 histopathological images with expert annotations) </b><br>
<a href="https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation">
https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation</a>
</a>
<br>
<br>
<b>2. Tensorflow-Image-Segmentation-Breast-Cancer-Cell</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Breast-Cancer-Cell">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Breast-Cancer-Cell
</a>
<br>
<br>

