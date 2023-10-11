# MFF-GBDT
## An improved method for broiler weight estimation integrating multi-feature with gradient boosting decision tree

![image](https://user-images.githubusercontent.com/50978393/219844459-653e72f3-413f-4001-b659-a44b1f1a791a.png)

### Dataset Download
- [BroilerDataset.zip](https://pan.baidu.com/s/1B2Amc51YuSQZ26j7ve92uw?pwd=bbu6 )

- [Broiler_Segment_205.zip](https://pan.baidu.com/s/1UUoFQDt4Px4VUY3Jq4-MeA?pwd=tt7b)
```
├─BroilerDataset                # Dataset used to extract features and train GBDT
|  ├─20210206体重登记表.xlsx     # Body weight infomation
│  ├─20210206-200-1198-manuals  # extracted artificial features
│  ├─20210206-200-1198-withauto # extracted artificial and learned features
│  ├─bimask                     # Binary image of a single broiler
│  ├─maskImg                    # masked image of a single broiler
│  └─origin                     # image of single broiler
├─mixData                       # Dataset used to train maskrcnn
│  ├─mask                       # Tag images used to train maskrcnn
│  └─origin                     # Raw image used to train maskrcnn
```
