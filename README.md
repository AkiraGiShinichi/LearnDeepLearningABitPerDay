# LearnDeepLearningABitPerDay
Drop-Learning the Deep Learning. Each day learn 1 pomodoro whatever about DL: vision, techniques, coding, ...

**Rule of challenge*:
> Each day learn a bit. But if there is a day that you can not follow, don't be down and don't give up.
> Do it additionally the next day.
> DON'T be off 2 days consecutively

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

### Day 15(20200215): Practice simple Pytorch(ngày 15 nghỉ, làm bù vào ngày 18).

[Practice code - Create model, fit and get_data](day15)
```python
class ModelX(nn.Module):
    ...


def get_data(...):
   ...
   return (DataLoader(train_ds, bs, shuffle=True),
	   DataLoader(valid_ds, bs))

def create_model():
   return model, opt

def batch_loss(model, loss_func, xb, yb, opt=None):
   ...

def fit(...):
   ...

```

### Day 16(20200216): Practice simple Pytorch(ngày 16 nghỉ, làm bù vào ngày 18).

[Practice code - Switch to CNN](day16/29Switch to CNN.ipynb)
[Practice code - use Sequence](day16/30nn.Sequential.ipynb)

**Pytorch flexible computation**:
 - x.view(...)

**Run model on GPU**:
```python
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
...
x.view(...).to(dev)
y.to(dev)
model.to(dev)
```

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

### Day 18(20200218): Read [Tạo chatbot trên Chatwork tự động giải đáp thông tin về dịch COVID-2020](https://viblo.asia/p/tao-chatbot-tren-chatwork-tu-dong-giai-dap-thong-tin-ve-dich-covid-2020-924lJq9XZPM?fbclid=IwAR04xoSkAA3z9dZzEV12flP9tN_vpU_x1H3jeIavgirhWLSOFA8nJpovyF8)

**Main problems**:
 - update dữ liệu thường xuyên
 - hệ thống chat
 - hệ thống backend

**System Design**:
![](https://images.viblo.asia/1acf30f4-e621-4804-98b3-fa66df8e2fba.png)
 - Chatwork get question from user, it request Webhook -> Webhook parse message then forward to NLU of RASA
 - RASA analyses message intent then executes logic operation.
 - base on intent, Python Pandas analyses data then responses by Django.
 - Chatwork gets response message than forward it to user.

### Day 19(20200219): Practice expression classification - Làm bù vào ngày 20.

**Done**:
 - Download expression data from Kaggle.
 - build program to train classification network, and test prediction

**Not Done Yet**:
 - evaluate accuracy
 - not show comparison matrix

[Practice code - expression classification](day19/prac-lesson21.ipynb)

### Day 20(20200220): Synthetize kinds of Neural Network - Làm bù vào ngày 21.

**Network types**:
 - Forward: 
	+ Perceptron
	+ Feed Forward
	+ **Deep Feed Forward(DFF) ~ Multi-layer Perceptron(MLP)**
 - Recurrent:
	+ Recurrent Neural Network(RNN)
	+ **Long/Short Term Memory(LSTM)**
	+ Gated Recurrent Unit(GRU)
 - Auto-Encoder:
	+ **Auto-Encoder(AE)**
	+ Variantional Auto-Encoder(VAE)
	+ Denoising Auto-Encoder(DAE)
	+ Sparse Auto-Encoder(SAE)
 - Chain:
	+ Markov Chain(MC)
	+ Hopfield Network(HN)
	+ Boltzmann Machine(BM)
	+ Restricted Boltzmann Machine(RBM)
	+ **Deep Belief Network(DBN)**
 - Convolutional:
	+ **Deep Convolutional Network(DCN)**
	+ Deconvotional Network(DN)
	+ Deep Convolutional Inverse Graphics Network(DCIGN)
 - Others:
	+ **Generative Adversarial Network(GAN)**
	+ **Deep Residual Network(DRN)**
	+ Support Vector Machine(SVM)
	+ LSM, ELM, ESN, KN, **Neural Turing Machine(NTM)**

![](https://miro.medium.com/max/2500/1*gccuMDV8fXjcvz1RSk4kgQ.png)

**Source Article**: [Cheat Sheets for AI, Neural Networks, Machine Learning, Deep Learning & Big Data](https://becominghuman.ai/cheat-sheets-for-ai-neural-networks-machine-learning-deep-learning-big-data-678c51b4b463)

### Day 21(20200221): Backpropagation & Weight Update.

**Network**:
![](https://miro.medium.com/max/1793/1*fnU_3MGmFp0LBIzRPx42-w.png)

**Activations**:
 - ReLU:
![](https://miro.medium.com/max/1440/1*Zydya39KBX6JoLUTRIneWQ.png)
![](https://miro.medium.com/max/1430/1*pWYG8v_vepzrvSkbV8z6lw.png)
 - Sigmoid:
![](https://miro.medium.com/max/1394/1*bnRzdcuXyQn5WHFCmcM36Q.png)
![](https://miro.medium.com/max/1563/1*7BV4iqZRHrkGmWRrmYOuFw.png)
 - Softmax:
![](https://miro.medium.com/max/1611/1*JgoFiHOQauknwhMd1_hjRg.png)
![](https://miro.medium.com/max/1649/1*7BzpaIXwjL8CIB4f5PHg_w.png)

**Error**:
![](https://miro.medium.com/max/1166/1*6YcTN8sQfJB4P6NamglBUg.png)
![](https://miro.medium.com/max/1725/1*dRX0O_3p5FICxp1AhGVLLw.png)

**Derivatives**:
 - ReLU:
![](https://miro.medium.com/max/328/1*vuXXfYips0L51d3yfKrf6w.png)
 - Sigmoid:
![](https://miro.medium.com/max/678/1*9gk0k_cqUREhs_2CCjKmLw.png)
 - Softmax:
![](https://miro.medium.com/max/768/1*l6GNTFihUu0EuUMUGHMwpA.png)

**Backpropagation**:
 - Hidden 2 - Output:
 - Hidden 1 - Hidden 2:
 - Input - Hidden 1:
 
**Results**:
 - Initial weights:
 - Final weights:
 
**Source article**: [Back-Propagation is very simple. Who made it Complicated ?](https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c)
 
### Day 24(20200224): Read [DeepMind just released Haiku and RLax for neural networks and reinforcement learning](https://towardsdatascience.com/deepmind-just-released-haiku-and-rlax-for-neural-networks-and-reinforcement-learning-a6468f9352cc?gi=85fd0bd7214f)

#### Main content:
 - JAX: bộ gia tốc code
 - Haiku & RLax: bộ core thư viện cho JAX
 - Haiku: simple NN for JAX(developed by author of Sonnet)
 - RLax: simple RL for JAX
 - What is [Sonnet](https://github.com/deepmind/sonnet/blob/v2/README.md)? NN based on Tensorflow but pytorch-looked-like.

### Day 26(20200226): Read [A 2019 Guide to Object Detection](https://heartbeat.fritz.ai/a-2019-guide-to-object-detection-9509987954c3#2836)

#### Main networks:
 - R-CNN
 - Fast R-CNN
 - Faster R-CNN
 - Mask R-CNN
 - SSD
 - Yolo
 - R-FCN [Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409)
 - DSSD(Deconvolutional Single Shot Detector)
 
#### New networks:
 - Retinanet
 - Centernet
 - Tridentnet
 - LSDA [Large Scale Detection through Adaptation by jhoffman](http://lsda.berkeleyvision.org/)
 - ...
![](https://miro.medium.com/max/913/1*vfPZ2PuvTIQnnHsGjzfU_Q.png)

**Good Introduction about Object Detection** [Object Detection with Deep Learning: The Definitive Guide](https://tryolabs.com/blog/2017/08/30/object-detection-an-overview-in-the-age-of-deep-learning/)

#### New technique: Objects as Points
 - Use key point estimation to find center points
 - From center, regress other properties: 3D location, size, orientation,..
 - Advantages: faster & more accurate than bounding box methods

### Day 27(20200227): Read [Monocular 3D Object Detection in Autonomous Driving — A Review](https://towardsdatascience.com/monocular-3d-object-detection-in-autonomous-driving-2476a3c7f57e)

**Comment**:
 - honestly, this article is f-cking hard to understand
 - but it introduces a lot of new methods & terminologies.
 - should reference it after have a retively deep knowledge about 3D Object Detection

#### Main contents:
 - Monocular 3D object detection(mono3DOD): is Doing 3D Object Detection from 2D images.
 - 4 groups of development for mono3DOD: 
	+ representation transformation: is Converting perspective images to Birds-Eye-View(BEV) or Pseudo-lidar
	+ keypoints and shape: object detection networks
	+ geometric reasoning based on 2D/3D constraint:?
	+ direct generation of 3D bbox:?
 - 3D vehicle information can be recovered using monocular images:?

### Day 28(20200228): Best source to update Deep Learning

#### Most recommended Newsletters:
 - https://machinelearningmastery.com/category/deep-learning-for-computer-vision/
 - https://www.pyimagesearch.com/
 - https://www.machinelearningisfun.com/

or Blogs of Frameworks:
 - https://blog.keras.io/
 - https://pytorch.org/tutorials/
 - https://towardsdatascience.com/tagged/fastai
 
#### Twitter:
 - @GoogleAI, @OpenAI, @AndrewYNg, @KDNuggets, @Goodfellow_Ian, @YLeCun, @Karpathy
 
#### Blogs:
 - https://netflixtechblog.com/
 - https://research.fb.com/category/computer-vision/
 - https://ai.googleblog.com/
 - https://medium.com/airbnb-engineering

#### Youtube:
 - https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A
 - https://www.youtube.com/user/keeroyz/featured
 - https://www.youtube.com/user/andrewyantakng/
 - https://www.youtube.com/user/googlecloudplatform/featured
 - https://www.youtube.com/playlist?list=PLqFaTIg4myu8t5ycqvp7I07jTjol3RCl9
 - https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw/playlists

### Day 29(20200229): Practice backpropagation on Matlab (Learnt at day 29, but note at day 32)

**Practice**: [Code](day29/backpropagation-example.m)

**Source Article**: [BACKPROPAGATION EXAMPLE WITH NUMBERS STEP BY STEP](https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/)

### Day 30(20200301): 3D Representations (Learnt at day 30, but note at day 32)

#### 4 main representations:
 - Point Cloud
 - Mesh
 - Volumetric
 - Projected View RGB(D)
 
#### 4 main approach:
 - Directly process: PointNet
 - Edge-based: GraphCNN, SPH, ...
 - VoxNet, 3DShapeNets, SubVolume, ...
 - LFD, MVCNN, ...
 
#### DL methods for Point Clouds:
![](https://pbs.twimg.com/media/ENbxLfIU4AAemSA.jpg)

### Day 31(20200302): Reinforcement Learning Frameworks (Làm bù ngày 32)

#### Main contents - Keywords:
 - OpenAI Gym
 - Google Dopamine
 - Keras-RL
 - RLLib
 - MAgent
 - Facebook Horizon
 
#### Great Tutorial:
 - [A Free course in Deep Reinforcement Learning from beginner to expert](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)

**From source**: [A Comparison of Reinforcement Learning Frameworks: Dopamine, RLLib, Keras-RL, Coach, TRFL, Tensorforce, Coach and more](https://winderresearch.com/a-comparison-of-reinforcement-learning-frameworks-dopamine-rllib-keras-rl-coach-trfl-tensorforce-coach-and-more/)

#### Best RL Framework:
 - Stable Baselines [Source Benmark Article](https://medium.com/data-from-the-trenches/choosing-a-deep-reinforcement-learning-library-890fb0307092)

### Day 32(20200303): Reinforcement Learning for Drug Engineering

#### Main contents - Keywords:
 - AlphaFold: system to predict 3D structure of protein from protein sequence [source](https://deepmind.com/blog/article/AlphaFold-Using-AI-for-scientific-discovery)
 - SMILES([Simplified Molecular Input Line Entry System](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html)): is string representation of a molecule(based off structure & components of a givin molecule) [source](https://towardsdatascience.com/lets-make-some-molecules-with-machine-learning-%EF%B8%8F-429b8838e3ef)
 - MOSES([MOlecular SEtS](https://github.com/molecularsets/moses)): sets of molecules.
 - DDR1: a human gene what is associated to a lot of diseases. Inhibit it to cure disease by medical molecular.
 - ZINC15 dataset: sets of chemical compounds.

**From source**: [Drug Engineering](https://www.youtube.com/watch?v=ya3AdrfKYzc)

#### Practice:
 - Did follow Drug Engineering([GENTRL](https://github.com/insilicomedicine/GENTRL)) but unsuccessful because of memory limitation.




