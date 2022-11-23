# GMFSS
GMFlow based anime video frame interpolation

## We now provide [GMFupSS](https://github.com/98mxr/GMFupSS) and [GMFSS_union](https://github.com/98mxr/GMFSS_union) as the next generation of GMFSS with better performance. Welcome to try.

## Requirement

- Python 3.10.7 or other versions
- opencv-python
- pytorch
- numpy
- cupy
- pandas

## Usage
- Run the following command and view the result in the 'output' folder
```
python test.py
```

## Benchmarks on anime test set

- Download [Dataset](https://drive.google.com/file/d/1GZ3PwCqhDyD_5-9HCsJdowq2g8Dt31ax/view?usp=sharing) and unzip it to 'dataset' folder
- Run eval.py and see result at 'loss.csv'

```
PSNR: 23.839 SSIM: 0.908 LPIPS: 0.107
```

<img src="https://user-images.githubusercontent.com/68835291/190122330-1f3e0418-5e19-4383-a215-09f944cf5f85.gif" width="60%">

## Reference

Optical Flow:
[GMFlow](https://github.com/haofeixu/gmflow)

Video Interpolation: 
[SoftSplat-Full](https://github.com/JHLew/SoftSplat-Full)  [SoftSplat](https://github.com/sniklaus/softmax-splatting) [AnimeInterp](https://github.com/lisiyao21/AnimeInterp) [EISAI](https://github.com/ShuhongChen/eisai-anime-interpolator) [ECCV2022-RIFE](https://github.com/megvii-research/ECCV2022-RIFE)
