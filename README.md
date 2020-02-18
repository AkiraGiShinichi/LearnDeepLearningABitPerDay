# LearnDeepLearningABitPerDay
Drop-Learning the Deep Learning. Each day learn 1 pomodoro whatever about DL: vision, techniques, coding, ...

### Day 1(20200201): Have Determination on learning DL.
To follow these sources:
 - DL powerful framework fastai from Jeremy Howard: https://www.youtube.com/watch?v=XfoYk_Z5AkI&list=PLfYUBJiXbdtSIJb-Qd3pw0cqCbkGeS0xn
 - DL general view from Siraj Raval: https://www.youtube.com/watch?v=eN9Lb3vXsAw
 - DL techniques from PyImageSearch: https://www.pyimagesearch.com/category/tutorials/
 - VuHuuTiep: https://github.com/aivivn/d2l-vn?fbclid=IwAR2EVSScFf7WgXsS_tguhKdt6EG2o-W98x_Vhz1AruIvUxcCriAaMl7wheo
 
Expert goals:
 - Deep Learning
 - Reinforcement Learning
 - Cloud Service
 - Quantum Computing
 - Blockchain

### Day 2(20200202): Clear Goals of DL.
 - Image processing: computer vision + object detection
  - Neural Network models: DNN, CNN, DRNN, GAN, ResNET, Dense net, Siamese NET, ...
  - Optimization Algorithm: LP, DP, ILP, and all kind of heuristic searching algorithm
 - NLP: ?
 
### Day 3(20200203): DL Optimization.
 - LP: Logic Programs
 - DP: Dynamic Programming
 - GP: Generic Programming
 - ILP: Inductive Logic Programming
 - Quadratic Programming
 - Meta-heuristic algorithms: Genetic Algorithm or Partical Swarm Optimization & variant [Survey of Meta-Heuristic Algorithms for Deep Learning Trainin - 2016](https://www.intechopen.com/books/optimization-algorithms-methods-and-applications/survey-of-meta-heuristic-algorithms-for-deep-learning-training), A*, Deep Learning Heuristic Tree Search
 ![](https://www.intechopen.com/media/chapter/51131/media/fig2.png)

### Day 4(20200204): How to use [Pytorch/Fastai.v3](https://course.fast.ai/)
**Simplest training program**: 4 steps
```Python
from fastai.vision import *

# 1. Load data from path(download from URL to path)
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)

# 2. Define model
model = simple_cnn((3,16,16,2))

# 3. Create learner
learn = Learner(data, model)

# 4. Train
learn.fit(1)
```

**Should read tomorrow:** https://towardsdatascience.com/10-new-things-i-learnt-from-fast-ai-v3-4d79c1f07e33

**Should inquire next next day:** Fastai emphasis: https://www.fast.ai/

### Day 5(20200205): Read half of a article about Fastai v3 course [10 New Things I Learnt from fast.ai v3](https://towardsdatascience.com/10-new-things-i-learnt-from-fast-ai-v3-4d79c1f07e33)

**Summarize smt I learnt from reading it:**
 - Feedfoward Network 1 hidden layer có thể ước lượng mọi hàm ⇒ ứng dụng?
 - Các kiến trúc State of Art: Resnet50, Unet,...
 - Train nên hội tụ ở vùng phẳng, sẽ generalised hơn ⇒ ứng dụng: Cách train tới vùng phẳng? Dùng fit_one_cycle of triangle cyclical lr.
  - 2 new loss functions: **Pixel MSE, Feature loss** ⇒ Ứng dụng:?
  - **Discriminative lr** là gì?
  - Dùng Random Forest search hyperparameters thế nào?
  - **Mix precision training**: Single precision for backpropagation, half precision for feedforward. ⇒ How effective & How to implement?
  - Regularisation: use **magic number 0.1** for weight decay.

### Day 6(20200206): Continue reading half of a article about Fastai v3 course [10 New Things I Learnt from fast.ai v3](https://towardsdatascience.com/10-new-things-i-learnt-from-fast-ai-v3-4d79c1f07e33)
 - Some different kinds of classsification: multi-label classification, multi-class classification/multinomial classification. How loss of multi-label classification work?
 - Language Modeling: can use transfer learning, should check course ULMFiT.
 - Tabular Data: not understand yet :(
 - Collaborative Filtering: a kind of action predicting?
 - Image Generation(use GAN): What is crappification? Why GAN, momentum should be 0? Improve quality by Perceptual loss(AKA feature loss), what exactly is it?
 - Model interpretability: can observe through fastai activation heat map.
 - To penalize the model complexity by sum square and `wd` ratio.
 
 ### Day 7(20200207): Setup Deep Learning Evironment.
 
 **Install Ubuntu on MSI**:
  - [Disable Security Boot & add `nomodeset` before install](https://medium.com/@carlosero/making-ubuntu-18-04-work-on-msi-gs65-8re-9818f4d9dc9d). Should analysize only 2 partitions: swap + root
  - Update then restart(should add `nomodeset` when startting up also).
  - Install: 
   + NVIDIA drivers: `sudo ubuntu-drivers autoinstall`
   
**Install docker -> nvidia-docker -> tensorflow**:
 https://www.tensorflow.org/install/docker
 
*Config to use docker without sudo*: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04

**Install miniconda**:
 - Download & Install from official website
 - Run `source ~/.bashrc`

### Day 8(20200208): Try Fastai DL framework.
**Basic program to train a network:** 3 main steps
```python
# Reference: fastai/course-v3/nbs/dl1/lesson1-pets.ipynb
# 0.a. Import library
from fastai.vision import *
from fastai.metrics import error_rate

# 0.b. Obtain data information here
# path = untar_data(URLs.PETs)
# ... extract path_img, filenames from path
# ... define pattern from data specific

# 1. Define data loader
data = ImageDataBunch.from_name_re(path_img, filenames, pattern, ds_tfms=get_transforms(), size=224, bs=64).normalize(imagenet_stats)

# 2. Define learner
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

# 3. Train
learn.fit_one_cycle(4)
```

### Day 9(20200209): How to develope a DL application.
**Steps to develope a DL application**: 4 main steps
 - create your own data
 - train the model
 - interpretation
 - put model into production
 
#### Create your own data:
#### Train the model:
```python
# 1. Define learner
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

# 2. Train last-layers
learn.fit_one_cycle(4)
learn.save('stage-1')

# 3. Fine-tune first-layers
# 3.a. Find suitable lr
learn.unfreeze()
learn.lr_find()
learn.recorder.plot() # select min&max lr from this graph
# 3.b. Train
learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))
learn.save('stage-2')
```
#### Interpretation:
#### Put model into production:
```python
learn.export() # export to "export.pkl"
```

```python
defaults.device = torch.device('cpu')
learn = load_learner(path) # path of "export.pkl"
pred_class,pred_idx,outputs = learn.predict(img)
```

### Day 10(20200210): Create your own image-data from google.

**Steps to get urls of google images**:
 - type `label` of images into google images
 - press `ctrl+shift+J`(chrome) or `ctrl+shift+K`(firefox) to open Javascript console.
 - paste `urls=Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));` into console to download urls of images.
 - there will be a popup on the top of browser, select `show..` to show saving dialog and to download the image urls.
 
**Steps to download**:
 - create a folder such as `data`. Inside, copy csv of urls into and make the same name folder for each csv file.
 - in Jupyter, declare the path such as `path = Path('data')`.
 - download images by running `download_images(path/file, dest, max_pics=200)`

*Steps to remove broken images(Optional)*
 - in Jupyter run:
 ```python
 for c in classes:
   print(c)
   verify_images(path/c, delete=True, max_size=500)
 ```
*Steps to view images(Optional)*:
```python
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(7,8))
```


### Day 11(20200211): Practice simple Pytorch.
```python
%matplotlib inline
from fastai.basics import *

n=100
x = torch.ones(n, 2)
x[:, 0].uniform_(-1., 1)
# print(x)
a = tensor(3., 2)
# print(a)
y = x@a + torch.rand(n) # make func: y = a*x + noise

def mse(y_hat, y): return((y_hat - y) ** 2).mean()

def update(): # linear regression
    y_hat = x@aa
    loss = mse(y_hat, y)
    print(loss)
    loss.backward() # to measure aa.grad(=dloss/daa)
    with torch.no_grad():
        aa.sub_(lr * aa.grad) # update aa
        aa.grad.zero_() # clear grad of aa
    
aa = nn.Parameter(tensor(1., 1))
lr = 1e-1
for t in range(100): update() # adjust aa step by step
```

### Day 12(20200212): Practice simple Pytorch - [backpropagation](https://pytorch.org/tutorials/beginner/nn_tutorial.html#refactor-using-nn-module).

[Practice code](day12/Untitled20.ipynb)

*y = A*x + B*
```python
# 1st way
y = A.matmul(x) + B
# 2nd way
y = A @ x + B
```

*backpropagation*:
```python
for epochs:
    for batches:
        # xb
        # yb
        yb_hat = pred = model(xb)
        
        loss = nll(yb_hat, yb)
        
        loss.backward() # to measure all grads of all tensors that required_grad=True
        with torch.no_grad():
            # update hyper-parameter tensors
            weights -= weights.grad * lr
            biases -= biases.grad * lr
            # clear grad
            weights.grad.zero_()
            biases.grad.zero_()
```

### Day 13(20200213): Read Fastai [Lesson 1 Notes](https://forums.fast.ai/t/deep-learning-lesson-1-notes/27748)

**Some important functions of Fastai lesson 1**:

 1. `fastai.vision.data.ImageDataBunch`
![](https://forums.fast.ai/uploads/default/optimized/2X/c/ce82725c7c3e8c9d2fb828b2a4bd859018bc4cce_2_1035x540.png)

 2. `ds_tfms`
	 - applying to images on the fly
	 - changes all the image sizes to 224 X 224
	 - images get centered, cropped and zoomed a little bit by transformation functions
 
	**Q**: Why 224 not 256?
		- models are designed so that the final layer is of size 7 by 7
		- we want something where if you go 7 times 2 a bunch of times (224 = 7 2222*2)
 
 3. `data.normalize(imagenet_stats)`
	[Why normalize images by subtracting dataset's image mean, instead of the current image mean in deep learning?](https://stats.stackexchange.com/questions/211436/why-normalize-images-by-subtracting-datasets-image-mean-instead-of-the-current)

	*Advice*: If you have accuracy trouble while training a model one thing to verify is that whether you have normalized the data properly.

 4. `fastai.vision.learner.create_cnn`
	- resnet34 was trained on half million images of 1000 categories.

[Practice code using nn.functional.crossentropy](day13/22Using torch.nn.functional.ipynb)
[Practice code using nn.Module](day13/23Refactor using nn.Module.ipynb)

### Day 14(20200214): Practice simple Pytorch - [refactor nn.linear](https://pytorch.org/tutorials/beginner/nn_tutorial.html#refactor-using-nn-module).

Tree of essential modules:
 - torch:
   + .nn:
	1. Module
	2. Parameter
	3. Linear
   + .optim: SGD, ...
   + .utils.data:
	1. TensorDataset
	2. DataLoader

*Answer a question*: what is torch.no_grad():
	- temporarily set all requires_flag to false
	- because, when requires_flag=True, it's not able to change values
	- when it's False, the variable is able to be updated.
[Practice code using nn.functional.crossentropy](day13/22Using torch.nn.functional.ipynb)

[Practice code using nn.Linear, optim, utils.data](day14)

### Day 17(20200217): Read [Deep Learning cho những máy tính thiếu RAM](https://viblo.asia/p/deep-learning-cho-nhung-may-tinh-thieu-ram-Qbq5Q3VmZD8?fbclid=IwAR1SqfZsE7Zxw6p_fs_uz16fBqbiwOqPxSeNw4Frz0XDD3jCkH9_NySNxaM)

**Problem**:
 - Khi train network, dung lượng dành cho network model rất nhỏ. 1 phần là cho Optimizer(lưu Gradient). Phần lớn là cho Activation(nodes, trọng số, delta...).

![](https://images.viblo.asia/0acefde1-c29f-4990-b22f-8dbe4670b758.png)
*Q*: What is `Transfer network`.

**Solution**:
 - mua thêm RAM
 - giảm batchsize
 - Nén mô hình mạng(model compression): dùng kỹ thuật `Pruning` để cắt tỉa kết nối dư thừa(trọng số = 0); hoặc kỹ thuật `quantization` để kết hợp và phân cụm trọng số để biểu diễn *cùng một số lượng liên kết với ít bộ nhớ hơn* => đánh đổi: accuracy giảm.
 - Gradient Checkpointing: giảm Activation memory bằng cách thay đổi cách cập nhật trọng số nhưng số lượng tính toán, thời gian train tăng lên nhiều(O(n) -> O(n^2)).
![](https://miro.medium.com/max/541/0*NARheCDvdoPc4A8z.)
![](https://miro.medium.com/max/676/0*udMSiPD0kZHum-sZ.)
 - Online learning:
![](https://miro.medium.com/max/4385/1*eSSU6uX7NR5kPK7ZbfnnVw.png)
 - Thay format dữ liệu














