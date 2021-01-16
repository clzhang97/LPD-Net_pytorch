
# A Novel Learned Primal-Dual Network for Image Compressive Sensing (LPD-Net)
Author et al.: C. Zhang, Y. Liu, F. Shang, Y. Li and H. Liu

Code Link: https://github.com/clzhang97/LPD-Net_pytorch

The code is tested in Linux and Windows environments (Python: 3.6.0, pytorch: 1.2.0, CUDA10.0) with Nvidia GTX 2080Ti GPU. 


### Test LPD-Net for Set11

You can test LPD-Net to reconstruct images for dataset Set11 by running the following command with the default cs ratio is 20%. The pre-trained model exists in "./trained_model" and 11 images of Set11 have been placed in "./data/Set11".

```python
python Test_LPD_Net.py
```

**After running the above command, you will see the following print message on the screen.**

> Testing. CS Reconstruction Start
> 
> \[01/11] data\Set11\barbara.tif:  PSNR is 27.88, SSIM is 0.8764
> 
> \[02/11] data\Set11\boats.tif:  PSNR is 32.01, SSIM is 0.9129
> 
> \[03/11] data\Set11\cameraman.tif:  PSNR is 27.89, SSIM is 0.8653
> 
> \[04/11] data\Set11\fingerprint.tif:  PSNR is 26.79, SSIM is 0.9148
> 
> \[05/11] data\Set11\flinstones.tif:  PSNR is 28.87, SSIM is 0.8682
> 
> \[06/11] data\Set11\foreman.tif:  PSNR is 38.15, SSIM is 0.9498
> 
> \[07/11] data\Set11\house.tif:  PSNR is 35.30, SSIM is 0.8953
> 
> \[08/11] data\Set11\lena256.tif:  PSNR is 31.35, SSIM is 0.9159
> 
> \[09/11] data\Set11\Monarch.tif:  PSNR is 31.58, SSIM is 0.9406
> 
> \[10/11] data\Set11\Parrots.tif:  PSNR is 30.63, SSIM is 0.9225
> 
> \[11/11] data\Set11\peppers256.tif:  PSNR is 33.06, SSIM is 0.9181
> 
> Avg PSNR/SSIM: 31.23/0.9073


**And the reconstruction images are saved in "./result"**

## References

J. Zhang and B. Ghanem, “ISTA-Net: Interpretable Optimization Inspired Deep Network for Image Compressive Sensing,” in CVPR. IEEE, 2018, pp. 1828–1837.
The code of ISTA-Net: https://github.com/jianzhangcs/ISTA-Net-PyTorch. 




