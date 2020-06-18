Code Link: https://github.com/clzhang97/LPD-Net_pytorch

The code is tested in Linux and Windows environments (Python: 3.6.0, pytorch: 1.2.0, CUDA10.0) with Nvidia GTX 2080Ti GPU. 

We use the same training data "Training_Data_Img91.mat" as ISTA-Net. Please download it and put it in the folder "./Training_data" first (https://github.com/jianzhangcs/ISTA-Net-PyTorch).

### Training LPD-Net 

You can train LPD-Net by running the following commands with different CS ratios.

```python
python Train_LPD_Net.py --cs_ratio=20
python Train_LPD_Net.py --cs_ratio=30
python Train_LPD_Net.py --cs_ratio=40
python Train_LPD_Net.py --cs_ratio=50
#...
```

### Test LPD-Net for Set11

You can test LPD-Net to reconstruct images for dataset Set11 by running the following command. The pre-trained model exists in "./Pre_trained_model" and 11 images of Set11 have been placed in "./Test_images".

```python
python Test_LPD_Net.py --cs_ratio=20
```

**After running the above command, you will see the following print message on the screen.**

> CS reconstruction for Set11.
> 
> \[01/11]: barbara.tif, PSNR/SSIM:  27.88/0.8764
> 
> \[02/11]: boats.tif, PSNR/SSIM:  32.01/0.9129
> 
> \[03/11]: cameraman.tif, PSNR/SSIM:  27.89/0.8653
> 
> \[04/11]: fingerprint.tif, PSNR/SSIM:  26.79/0.9148
> 
> \[05/11]: flinstones.tif, PSNR/SSIM:  28.87/0.8682
> 
> \[06/11]: foreman.tif, PSNR/SSIM:  38.15/0.9498
> 
> \[07/11]: house.tif, PSNR/SSIM:  35.30/0.8953
> 
> \[08/11]: lena256.tif, PSNR/SSIM:  31.35/0.9159
> 
> \[09/11]: Monarch.tif, PSNR/SSIM:  31.58/0.9406
> 
> \[10/11]: Parrots.tif, PSNR/SSIM:  30.63/0.9225
> 
> \[11/11]: peppers256.tif, PSNR/SSIM:  33.06/0.9181
> 
> Avg PSNR/SSIM is 31.23/0.9073 for Set11 when CS ratio is 20%.

**And the reconstruction images are saved in "./Image_reconstruction_result"**

## References

[1] J. Zhang and B. Ghanem, “ISTA-Net: Interpretable Optimization Inspired Deep Network for Image Compressive Sensing,” in CVPR. IEEE, 2018, pp. 1828–1837.

[2] The code of ISTA-Net: https://github.com/jianzhangcs/ISTA-Net-PyTorch. 





# LPD-Net
# LPD-Net
