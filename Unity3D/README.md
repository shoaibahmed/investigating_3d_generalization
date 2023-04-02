# Unity3D

This directory contains the Unity project (named Paperclip) in order to generate both the synthetic paperclip dataset as well as the 3D chairs dataset.
Open the scene, select the main GameObject, and select one of the scripts to either generate the paperclips dataset or the 3D chairs dataset.

Please configure the dataset path (only if generating the 3D chairs dataset) as well as the output path at the following locations:
```
ScreenRecorder.cs:77
ModelLoaderMultiRot.cs:112
```

Since the scripts also generates views of the object rotating along multiple axes simultaneously, we use a stride of 10 degrees (configurable in the script) to generate views in this setting.
The generation stride also needs to be specified to the trainer.

## Conversion to WebDataset

The trainer expects dataset to be provided in the WebDataset format.
Once the dataset generation using the Unity project is completed, simply convert that into a tar file:
```bash
tar -cvf paperclips_v6.tar.xz ./Paperclips_v6/
```

Once the files have been compressed, these files needs to be broken down into proper training and evaluation fails.
Although the main compressed file can be used directly for both training as well as evaluation, discarding samples at training time is extremely slow.
Therefore, we create separate WebDataset files for fast data loading.

`wds/gen` contains two different WDS generators. One generator is for the paperclips, and the other one is for 3D models.
Please do configure the compressed file path correctly in the generators.

## Generated Datasets

The models in 3D chairs dataset are taken from ShapeNet, which requires permission to download.
Therefore, we only directly distribute the tar file for the Paperclips.
This is the main tar file, and requires running the generators in order to obtain the corresponding training and testing files.

[Link to be added soon as the file is huge i.e., ~90GB]

## Acknowledgements

The code relies on the Unity OBJImport external package for loading 3D models such as 3D chairs from ShapeNet.

## License

MIT
