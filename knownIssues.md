# Auto Annotate - Known Issues

Sharing below a few of the known issues for the Auto Annotate tool. These known issues are basically the ones arising due to python environment disparity (different version of the packages installed locally) or the bugs which have not been resolved yet and have a workaround.

**❤ </> Please feel free to contribute to this page. </> ❤**

---
## 1. AttributeError: 'str' object has no attribute 'decode'

### Contributer     :  [jarleven](https://github.com/jarleven)
### Fix:
Downgrade h5py. 
```
pip install 'h5py==2.10.0' --force-reinstall
```
### Reference:
https://stackoverflow.com/questions/53740577/does-any-one-got-attributeerror-str-object-has-no-attribute-decode-whi

### Stack Trace:
python3 annotate.py annotateCustom --image_directory=/host/Notes/SalmonModel --label=Salmon --weights=/host/mask_rcnn_coco_tf22_01082021/mask_rcnn_mmodel_0030.h5 --displayMaskedImages=True

```
Loading weights... /host/mask_rcnn_coco_tf22_01082021/mask_rcnn_mmodel_0030.h5
Traceback (most recent call last):
File "annotate.py", line 218, in
model.load_weights(weights_path, by_name=True)
File "/Auto-Annotate/mrcnn/model.py", line 2130, in load_weights
saving.load_weights_from_hdf5_group_by_name(f, layers)
File "/usr/local/lib/python3.6/dist-packages/keras/engine/topology.py", line 3114, in load_weights_from_hdf5_group_by_name
original_keras_version = f.attrs['keras_version'].decode('utf8')
AttributeError: 'str' object has no attribute 'decode'
```
---
