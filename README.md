# GMFSS
GMFlow based anime video frame interpolation

---

**2023-04-03: We now propose [GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna) as a factual basis for training in GMFSS. Please use it. This item will not be updated!**

**2022-11-23: [GMFSS_union](https://github.com/98mxr/GMFSS_union) —— The next generation of GMFSS with better performance.**

**2022-09-18: [GMFupSS](https://github.com/98mxr/GMFupSS) —— A faster GMFSS**

---

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

- Switch to project [GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna)
- Download [ATD-12K](https://drive.google.com/file/d/1GZ3PwCqhDyD_5-9HCsJdowq2g8Dt31ax/view?usp=sharing) and unzip it, rename atd_test2k to dataset
- Put the dataset to the root of eval.py
- Run eval.py and see result at 'loss.csv'

```
PSNR: 29.22 SSIM: 0.932 LPIPS: 0.048
```

<img src="https://user-images.githubusercontent.com/68835291/190122330-1f3e0418-5e19-4383-a215-09f944cf5f85.gif" width="60%">

## Also, have a look on VideoFlow’s First VFI application, [VFSS](https://github.com/hyw-dev/VFSS)!

## Reference

Optical Flow:
[GMFlow](https://github.com/haofeixu/gmflow)

Video Interpolation: 
[SoftSplat-Full](https://github.com/JHLew/SoftSplat-Full)  [SoftSplat](https://github.com/sniklaus/softmax-splatting) [AnimeInterp](https://github.com/lisiyao21/AnimeInterp) [EISAI](https://github.com/ShuhongChen/eisai-anime-interpolator) [ECCV2022-RIFE](https://github.com/megvii-research/ECCV2022-RIFE)

## Acknowledgment
This project is sponsored by [SVFI](https://steamcommunity.com/app/1692080) [Development Team](https://github.com/Justin62628/Squirrel-RIFE) 
