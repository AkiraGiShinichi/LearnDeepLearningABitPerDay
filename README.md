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
**Basic program to train a network**
```python
# 1. Import library
from fastai.vision import *
from fastai.metrics import error_rate

# 2. Get data
path = untar_data(URLs.PETs)
# ... extract path_img, filenames from path
# ... define pattern from data specific

# 3. Define data object
data = ImageDataBunch.from_name_re(path_img, filenames, pattern, ds_tfms=get_transforms(), size=224, bs=64).normalize(imagenet_stats)

# 4. Define learner
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

# 5. Train
learn.fit_one_cycle(4)
```
