# GMFSS
GMFlow based video frame interpolation

## Requirement

- opencv-python
- pytorch
- numpy
- cupy

## Usage
- Download weights from [Google Drive](https://drive.google.com/drive/folders/1ZmV2KZJd0ywwheqdszOkV2j2DWjj9G2e?usp=sharing) and put them inside the weights folder
- Run the following command and view the result in the output folder
```
python test.py
```

## Benchmarks on the anime test set
     PSNR: 23.839 SSIM: 0.908 LPIPS: 0.107
<img src="https://user-images.githubusercontent.com/68835291/190122330-1f3e0418-5e19-4383-a215-09f944cf5f85.gif" width="60%">

## Reference

Optical Flow:
[GMFlow](https://github.com/haofeixu/gmflow)

Video Interpolation: 
[SoftSplat-Full](https://github.com/JHLew/SoftSplat-Full)  [SoftSplat](https://github.com/sniklaus/softmax-splatting) [AnimeInterp](https://github.com/lisiyao21/AnimeInterp) [EISAI](https://github.com/ShuhongChen/eisai-anime-interpolator)
