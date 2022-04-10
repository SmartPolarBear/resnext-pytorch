# ResNeXt-pytorch
A pytorch implementation for ResNeXt.

## Dependencies

- Pytorch 1.11.0
- CUDA 11.5

## Usage

This project contains a demo for a regression task to train on [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)
to rate human faces.

### Train

To train the model, use `demo/face_rating.py`.

### Demo

To test the trained model, use `demo/face_rating_demo.py`. 

## Citation

If you use ResNeXt in your research, please cite the paper:

```
@article{Xie2016,
  title={Aggregated Residual Transformations for Deep Neural Networks},
  author={Saining Xie and Ross Girshick and Piotr Dollár and Zhuowen Tu and Kaiming He},
  journal={arXiv preprint arXiv:1611.05431},
  year={2016}
}
```