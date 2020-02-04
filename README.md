# LearnDeepLearningABitPerDay
Drop-Learning the Deep Learning. Each day learn 1 pomodoro whatever about DL: vision, techniques, coding, ...

### Day 1(20200201): Have Determination on learning DL.
To follow these sources:
 - DL powerful framework fastai from Jeremy Howard: https://www.youtube.com/watch?v=XfoYk_Z5AkI&list=PLfYUBJiXbdtSIJb-Qd3pw0cqCbkGeS0xn
 - DL general view from Siraj Raval: https://www.youtube.com/watch?v=eN9Lb3vXsAw
 - DL techniques from PyImageSearch: https://www.pyimagesearch.com/category/tutorials/
 
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
