Currently on Semcova dataset:
Lowest MAE for Spo2 on test set, paper reported 1.1%:

- Mobilenet_v2: 0.85%
- Resnet18: 1.05%
- VGG13: 0.95%
- VG16: 0.87%

Please be aware these are the best epoch results, before it starts over-fitting. Since we do not have a validation set (data to scarce), this is an overly optimistic results. There is also not enough training data to call this anywhere conclusive, but we have the models available to train once we get more data.

The code is recycled from other GitHub (referenced in file), therefore it probably needs a clean-up.

You can run experiments by setting the models to be tested at 
'''
def get_models():
    return [
        (4, resnet18(num_classes=2)),
        (4, vgg13bn(num_classes=2)),
        (4, mobilenet_v2(False, True))
    ]
'''
It will save the best model so far in the folder specified at 'experiment_path'.