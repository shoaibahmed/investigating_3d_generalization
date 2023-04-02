# Unity3D

This directory contains the Unity project in order to generate both the synthetic paperclip dataset as well as the 3D chairs dataset.
Open the scene, select the main GameObject, and select one of the scripts to either generate the paperclips dataset or the 3D chairs dataset.

Please configure the dataset path (only if generating the 3D chairs dataset) as well as the output path at the following locations:
```
ScreenRecorder.cs:77
ModelLoaderMultiRot.cs:112
```

## Acknowledgements

The code relies on the Unity OBJImport external package for loading 3D models such as 3D chairs from ShapeNet.

## License

MIT
