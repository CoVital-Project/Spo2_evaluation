# CoVital Pytorch Dataset and Dataloader

In this folder there is the functionality to load videos from a folder following the structure described in [README](../sample_data/README.md).
A Dataset instance can be created by:
```  
dataset = Spo2Dataset(data_path)
```  
This will itearate the folders at data_path and load their videos and compute the mean and std for each channel per frame and load the ground truth as labels. It also allows to store metadata for each video. Once the dataset is created, it will contain only the mean and std. This process is slow as to preserve memory, it does it frame per frame. Feel free to modify it if you know ways to speed up the process without causing memory issues. Using torchvision.io.read_video ran out of memory in a 32 gb RAM computer. 

Once the dataset is ready, it can be fed to a DataLoader object. 

```  
dataloader = Spo2DataLoader(dataset, batch_size=4, collate_fn= Spo2DataLoader.collate_fn)
```  

The output needs to be batched tensors, and therefore they have to share the same length. Since we have videos of different lengths, it pads the shorted ones to fit the length of the longest one in each frame. This may be an issue for models which require the same length for all batches, but it is convinient for RNN models. The real length of each video is accessible for each batch. Each batch returns three variables, videos_batch, labels_batch and videos_lengths.