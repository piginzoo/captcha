
## 我照着撸验证码破解

**为毛要撸他**  

本文参考自DataCastle上的一个识别码破解，原作者是第一名诶~，里面很多CNN训练细节，必须学习之，谁让我CNN和深度学习深度迷惑呢。

感谢作者的辛苦劳动，我只是照着把他的方法重新撸了一遍，作者木有贴出训练代码，模型呢，我也简化了，主要是学习嘛，针对初学者滴滴。

我只撸了type5的代码，其他的大同小异，不玩了...

**作者本体**  

这本是datacastle上的比赛，真好，大家都分享，还分享[思路](http://blog.csdn.net/DataCastle/article/details/52148793)，好赞哦，比较难的是type5的，主要是字母数量不定，作者的思路是，先用个网络找个数，再用个网络挨个识别（长度确定就可以挨个切出来了），好屌！

[去比赛遗址](https://github.com/lllcho/CAPTCHA-breaking/tree/master/model)

[瞻仰原作者](https://github.com/lllcho/CAPTCHA-breaking)

**我在撸之前准备好的手纸**  

 - 如何加载数据，网上都是一片片的MINST数据集，如何加载自己的数据集
 - 作者用的是keras 0.x的版本，老了，我要撸成 2.x的
 - 改改他的网络，我先不追求质量，先撸个简单的，训练快
 
 
带着这些手纸，上路

**环境**

环境我用keras事

```
>>> import keras as k
>>> k.__version__
'2.0.5'
>>> import tensorflow as t
>>> t.__version__
'1.2.1'
>>> import cv2
>>> cv2.__version__
'3.4.0'
```

** 测试数据 **
更多测试数据[下载地址](http://pan.baidu.com/s/1hqk6rxa)
