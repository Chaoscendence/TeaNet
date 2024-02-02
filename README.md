# TeaNet

This repository is for **Reconstructing Randomly Masked Spectra Helps DNNs Identify Discriminant Wavenumbers**.

## Datasets

We used **[RRUFF_IR](https://rruff.info/zipped_data_files/infrared/Processed.zip)**, **[USGS](https://www.sciencebase.gov/catalog/item/requestDownload/5807a2a2e4b0841e59e3a18d?filePath=__disk__97%2F9c%2F35%2F979c35f740ed4991a282a918115a6652270462dd)** and **[ReLab](https://drive.google.com/drive/folders/1KNvZSw5nPa75CUG0-FZsaLq29b7B-ZTU)** in our experiment. The classes are listed under `data`  for each dataset.

## Code structure

```
code/
│
├── main.py - main script to start training ang testing
├── train_model.py - the training code
├── test_model.py - the testing code
│
├── configs/ - configuration for trainging and data
│   ├── __init__.py
│   ├── defaults.py - default configuration
│   └── config.yaml - custom configuration
│
├── data_handle/
│   └── dataset.py - split train/valid and mask data
│
├── models/ - contains models and loss function
│   ├── lenet_ir.py - the classifier used for RRUFF_IR and ReLab
│   ├── lenet_usgs.py - the classifier used for USGS and synthetic spectrum
│   ├── unet.py - the generator
│   └── loss_func.py - losses used for trainging generator
│  
└── utils/ - small utility functions
    ├── log.py
    └── utils.py
```

### Train a TeaNet

```bash
python main.py --train --config [your own config file, default: configs/config.yaml]
```

### Test a TeaNet

```bash
python main.py --test --models_dir [path of testing model]
```

### Results

| Model  | RRUFF_IR  | USGS | ReLab |
| :----- | :-------: | :----------------: | :---: |
| Accuracy | 90.07% |        87.37%         |  85.43%  |

## Notes

The default configs are setted in `code/configs/config.yaml`, e.g. data_path, you should modify it first for your own  experiment. The data should be splitted into 'train_valid' and 'test' in advance.

If you have any quesitons, please contact us by liujinchao@nankai.edu.cn

## Citation

Please cite our paper in your publications if you find the code useful:

```
@article{wu2023reconstructing,
  title={Reconstructing Randomly Masked Spectra Helps DNNs Identify Discriminant Wavenumbers},
  author={Wu, Yingying and Liu, Jinchao and Wang, Yan and Gibson, Stuart and Osadchy, Margarita and Fang, Yongchun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```


