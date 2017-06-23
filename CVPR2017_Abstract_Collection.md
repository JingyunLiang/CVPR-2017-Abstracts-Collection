# CVPR 2017 Abstracts Collection
_Collection of CVPR 2017, including titles, links, authors, abstracts and my own comments. Created by Michael Liang, NUDT. Note that ** and Comment<  > are my own comments._

## Machine Learning 1

Spotlight 1-1A

#### Exclusivity-Consistency Regularized Multi-View Subspace Clustering

Xiaobo Wang, Xiaojie Guo, Zhen Lei, Changqing Zhang, Stan Z. Li

_Abstract[]_

Comment<>

#### *Borrowing Treasures From the Wealthy: Deep Transfer Learning Through Selective Joint Fine-TuningAbstract[PDF](https://arxiv.org/abs/1702.08690)

Weifeng Ge, Yizhou Yu

_Abstract[Deep neural networks require a large amount of labeled training data during supervised learning. However, collecting and labeling so much data might be infeasible in many cases. In this paper, we introduce a source-target selective joint fine-tuning scheme for improving the performance of deep learning tasks with insufficient training data. In this scheme, a target learning task with insufficient training data is carried out simultaneously with another source learning task with abundant training data. However, the source learning task does not use all existing training data. Our core idea is to identify and use a subset of training images from the original source learning task whose low-level characteristics are similar to those from the target learning task, and jointly fine-tune shared convolutional layers for both tasks. Specifically, we compute descriptors from linear or nonlinear filter bank responses on training images from both tasks, and use such descriptors to search for a desired subset of training samples for the source learning task. Experiments demonstrate that our selective joint fine-tuning scheme achieves state-of-the-art performance on multiple visual classification tasks with insufficient training data for deep learning. Such tasks include Caltech 256, MIT Indoor 67, Oxford Flowers 102 and Stanford Dogs 120. In comparison to fine-tuning without a source domain, the proposed method can improve the classification accuracy by 2% - 10% using a single model.]_

Comment<a source-target selective joint fine-tuning scheme with insufficient training data; insufficient & abundant task with similar low-level feature trained simultaneously; share conv layers; help labelling data,>

#### **The More You Know: Using Knowledge Graphs for Image ClassificationAbstract[PDF](https://arxiv.org/abs/1612.04844)

Kenneth Marino, Ruslan Salakhutdinov, Abhinav Gupta

_Abstract[One characteristic that sets humans apart from modern learning-based computer vision algorithms is the ability to acquire knowledge about the world and use that knowledge to reason about the visual world. Humans can learn about the characteristics of objects and the relationships that occur between them to learn a large variety of visual concepts, often with few examples. This paper investigates the use of structured prior knowledge in the form of knowledge graphs and shows that using this knowledge improves performance on image classification. We build on recent work on end-to-end learning on graphs, introducing the Graph Search Neural Network as a way of efficiently incorporating large knowledge graphs into a vision classification pipeline. We show in a number of experiments that our method outperforms standard neural network baselines for multi-label classification.]_

Comment< the real way to AI >

#### Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on GraphsAbstract[PDF](https://arxiv.org/abs/1704.02901)

Martin Simonovsky, Nikos Komodakis
_Abstract[A number of problems can be formulated as prediction on graph-structured data. In this work, we generalize the convolution operator from regular grids to arbitrary graphs while avoiding the spectral domain, which allows us to handle graphs of varying size and connectivity. To move beyond a simple diffusion, filter weights are conditioned on the specific edge labels in the neighborhood of a vertex. Together with the proper choice of graph coarsening, we explore constructing deep neural networks for graph classification. In particular, we demonstrate the generality of our formulation in point cloud classification, where we set the new state of the art, and on a graph classification dataset, where we outperform other deep learning approaches.]_

Comment< graph-structured data >

#### *Convolutional Neural Network Architecture for Geometric MatchingAbstract[PDF](https://arxiv.org/abs/1703.05593)

Ignacio Rocco, Relja ArandjeloviÄ‡, Josef Sivic

_Abstract[We address the problem of determining correspondences between two images in agreement with a geometric model such as an affine or thin-plate spline transformation, and estimating its parameters. The contributions of this work are three-fold. First, we propose a convolutional neural network architecture for geometric matching. The architecture is based on three main components that zmimic the standard steps of feature extraction, matching and simultaneous inlier detection and model parameter estimation, while being trainable end-to-end. Second, we demonstrate that the network parameters can be trained from synthetically generated imagery without the need for manual annotation and that our matching layer significantly increases generalization capabilities to never seen before images. Finally, we show that the same model can perform both instance-level and category-level matching giving state-of-the-art results on the challenging Proposal Flow dataset.]_

Comment< geomatric matching; synthetically generated image; instance-level& category-level; matching layer,> 

#### Deep Affordance-Grounded Sensorimotor Object RecognitionAbstract[PDF](https://arxiv.org/abs/1704.02787)

Spyridon Thermos, Georgios Th. Papadopoulos, Petros Daras, Gerasimos Potamianos

_Abstract[It is well-established by cognitive neuroscience that human perception of objects constitutes a complex process, where object appearance information is combined with evidence about the so-called object "affordances", namely the types of actions that humans typically perform when interacting with them. This fact has recently motivated the "sensorimotor" approach to the challenging task of automatic object recognition, where both information sources are fused to improve robustness. In this work, the aforementioned paradigm is adopted, surpassing current limitations of sensorimotor object recognition research. Specifically, the deep learning paradigm is introduced to the problem for the first time, developing a number of novel neuro-biologically and neuro-physiologically inspired architectures that utilize state-of-the-art neural networks for fusing the available information sources in multiple ways. The proposed methods are evaluated using a large RGB-D corpus, which is specifically collected for the task of sensorimotor object recognition and is made publicly available. Experimental results demonstrate the utility of affordance information to object recognition, achieving an up to 29% relative error reduction by its inclusion.]_

Comment< Sensorimotor Object Recognition >

#### Discovering Causal Signals in ImagesAbstract[PDF](https://arxiv.org/abs/1605.08179)

David Lopez-Paz, Robert Nishihara, Soumith Chintala, Bernhard SchÃ¶lkopf, Léon Bottou

_Abstract[The purpose of this paper is to point out and assay observable causal signals within collections of static images. We achieve this goal in two steps. First, we take a learning approach to observational causal inference, and build a classifier that achieves state-of-the-art performance on finding the causal direction between pairs of random variables, when given samples from their joint distribution. Second, we use our causal direction finder to effectively distinguish between features of objects and features of their contexts in collections of static images. Our experiments demonstrate the existence of Abstract[PDF](1) a relation between the direction of causality and the difference between objects and their contexts, and Abstract[PDF](2) observable causal signals in collections of static images.]_

Comment< causal singals>

#### *On Compressing Deep Models by Low Rank and Sparse Decomposition

Xiyu Yu, Tongliang Liu, Xinchao Wang, Dacheng Tao

_Abstract[]_

Comment< >

Oral 1-1A

#### PointNet: Deep Learning on Point Sets for 3D Classification and SegmentationAbstract[PDF](https://arxiv.org/abs/1612.00593)

Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas

_Abstract[Point cloud is an important type of geometric data structure. Due to its irregular format, most researchers transform such data to regular 3D voxel grids or collections of images. This, however, renders data unnecessarily voluminous and causes issues. In this paper, we design a novel type of neural network that directly consumes point clouds and well respects the permutation invariance of points in the input. Our network, named PointNet, provides a unified architecture for applications ranging from object classification, part segmentation, to scene semantic parsing. Though simple, PointNet is highly efficient and effective. Empirically, it shows strong performance on par or even better than state of the art. Theoretically, we provide analysis towards understanding of what the network has learnt and why the network is robust with respect to input perturbation and corruption.]_

Comment< point cloud, 3D>

#### ***Universal Adversarial PerturbationsAbstract[PDF](https://arxiv.org/abs/1610.08401)

Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, Pascal Frossard

_Abstract[Given a state-of-the-art deep neural network classifier, we show the existence of a universal Abstract[PDF](image-agnostic) and very small perturbation vector that causes natural images to be misclassified with high probability. We propose a systematic algorithm for computing universal perturbations, and show that state-of-the-art deep neural networks are highly vulnerable to such perturbations, albeit being quasi-imperceptible to the human eye. We further empirically analyze these universal perturbations and show, in particular, that they generalize very well across neural networks. The surprising existence of universal perturbations reveals important geometric correlations among the high-dimensional decision boundary of classifiers. It further outlines potential security breaches with the existence of single directions in the input space that adversaries can possibly exploit to break a classifier on most natural images.]_

Comment< using unseen perturbations to break a classifier, destory DL>

#### *Unsupervised Pixel-Level Domain Adaptation With Generative Adversarial NetworksAbstract[PDF](https://arxiv.org/abs/1612.05424)

Konstantinos Bousmalis, Nathan Silberman, David Dohan, Dumitru Erhan, Dilip Krishnan

_Abstract[Collecting well-annotated image datasets to train modern machine learning algorithms is prohibitively expensive for many tasks. One appealing alternative is rendering synthetic data where ground-truth annotations are generated automatically. Unfortunately, models trained purely on rendered images often fail to generalize to real images. To address this shortcoming, prior work introduced unsupervised domain adaptation algorithms that attempt to map representations between the two domains or learn to extract features that are domain-invariant. In this work, we present a new approach that learns, in an unsupervised manner, a transformation in the pixel space from one domain to the other. Our generative adversarial network Abstract[PDF](GAN)-based method adapts source-domain images to appear as if drawn from the target domain. Our approach not only produces plausible samples, but also outperforms the state-of-the-art on a number of unsupervised domain adaptation scenarios by large margins. Finally, we demonstrate that the adaptation process generalizes to object classes unseen during training.]_

Comment< generate domain-invariant annotations, adapts source-domain images to appear as if drawn from the target domain ;GAN>

#### Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network Abstract[PDF](PDF, code)Abstract[PDF](https://arxiv.org/pdf/1609.04802.pdf)Abstract[PDF](https://github.com/leehomyc/Photo-Realistic-Super-Resoluton)

Christian Ledig, Lucas Theis, Ferenc HuszÃƒÂ¡r, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi

_Abstract[Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this paper, we present SRGAN, a generative adversarial network Abstract[PDF](GAN) for image super-resolution Abstract[PDF](SR). To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. An extensive mean-opinion-score Abstract[PDF](MOS) test shows hugely significant gains in perceptual quality using SRGAN. The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method.]_

Comment< Super-Resolution, GAN>

## 3D Vision 1

Spotlight 1-1B

#### Context-Aware Captions From Context-Agnostic SupervisionAbstract[PDF](https://arxiv.org/abs/1701.02870)

Ramakrishna Vedantam, Samy Bengio, Kevin Murphy, Devi Parikh, Gal Chechik

_Abstract[We introduce an inference technique to produce discriminative context-aware image captions Abstract[PDF](captions that describe differences between images or visual concepts) using only generic context-agnostic training data Abstract[PDF](captions that describe a concept or an image in isolation). For example, given images and captions of "siamese cat" and "tiger cat", we generate language that describes the "siamese cat" in a way that distinguishes it from "tiger cat". Our key novelty is that we show how to do joint inference over a language model that is context-agnostic and a listener which distinguishes closely-related concepts. We first apply our technique to a justification task, namely to describe why an image contains a particular fine-grained category as opposed to another closely-related category of the CUB-200-2011 dataset. We then study discriminative image captioning to generate language that uniquely refers to one of two semantically-similar images in the COCO dataset. Evaluations with discriminative ground truth for justification and human studies for discriminative image captioning reveal that our approach outperforms baseline generative and speaker-listener approaches for discrimination. ]_

Comment< distinguished image caption>

#### Global Hypothesis Generation for 6D Object Pose EstimationAbstract[PDF](https://arxiv.org/abs/1612.02287)

Frank Michel, Alexander Kirillov, Eric Brachmann, Alexander Krull, Stefan Gumhold, Bogdan Savchynskyy, Carsten Rother

_Abstract[This paper addresses the task of estimating the 6D pose of a known 3D object from a single RGB-D image. Most modern approaches solve this task in three steps: i) Compute local features; ii) Generate a pool of pose-hypotheses; iii) Select and refine a pose from the pool. This work focuses on the second step. While all existing approaches generate the hypotheses pool via local reasoning, e.g. RANSAC or Hough-voting, we are the first to show that global reasoning is beneficial at this stage. In particular, we formulate a novel fully-connected Conditional Random Field Abstract[PDF](CRF) that outputs a very small number of pose-hypotheses. Despite the potential functions of the CRF being non-Gaussian, we give a new and efficient two-step optimization procedure, with some guarantees for optimality. We utilize our global hypotheses generation procedure to produce results that exceed state-of-the-art for the challenging "Occluded Object Dataset".]_

Comment< pose estimation>

#### A Practical Method for Fully Automatic Intrinsic Camera Calibration Using Directionally Encoded Light

Mahdi Abbaspour Tehrani, Thabo Beeler, Anselm GrundhÃ¶fer

_Abstract[]_

Comment< camera calibration >

#### CATS: A Color and Thermal Stereo Benchmark

Wayne Treible, Philip Saponaro, Scott Sorensen, Abhishek Kolagunda, Michael O'Neal, Brian Phelan, Kelly Sherbondy, Chandra Kambhamettu

_Abstract[]_

Comment< >

#### Elastic Shape-From-Template With Spatially Sparse Deforming Forces

Abed Malti, Cédric Herzet

_Abstract[]_

Comment< >

#### Distinguishing the Indistinguishable: Exploring Structural Ambiguities via Geodesic Context

Qingan Yan, Long Yang, Ling Zhang, Chunxia Xiao

_Abstract[]_

Comment< >

#### Multi-Scale Continuous CRFs as Sequential Deep Networks for Monocular Depth EstimationAbstract[PDF](https://arxiv.org/abs/1704.02157)

Dan Xu, Elisa Ricci, Wanli Ouyang, Xiaogang Wang, Nicu Sebe

_Abstract[This paper addresses the problem of depth estimation from a single still image. Inspired by recent works on multi- scale convolutional neural networks Abstract[PDF](CNN), we propose a deep model which fuses complementary information derived from multiple CNN side outputs. Different from previous methods, the integration is obtained by means of continuous Conditional Random Fields Abstract[PDF](CRFs). In particular, we propose two different variations, one based on a cascade of multiple CRFs, the other on a unified graphical model. By designing a novel CNN implementation of mean-field updates for continuous CRFs, we show that both proposed models can be regarded as sequential deep networks and that training can be performed end-to-end. Through extensive experimental evaluation we demonstrate the effective- ness of the proposed approach and establish new state of the art results on publicly available datasets. ]_

Comment< depth estimation from a single still image; using multi-scale CNN's side output, integrated by continuous Conditional Random Fields Abstract[PDF](CRFs).>

#### Dynamic Time-Of-Flight

Michael Schober, Amit Adam, Omer Yair, Shai Mazor, Sebastian Nowozin

_Abstract[]_

Comment< >

Oral 1-1B

#### Semantic Scene Completion From a Single Depth ImageAbstract[PDF](https://arxiv.org/abs/1611.08974)

Shuran Song, Fisher Yu, Andy Zeng, Angel X. Chang, Manolis Savva, Thomas Funkhouser

_Abstract[This paper focuses on semantic scene completion, a task for producing a complete 3D voxel representation of volumetric occupancy and semantic labels for a scene from a single-view depth map observation. Previous work has considered scene completion and semantic labeling of depth maps separately. However, we observe that these two problems are tightly intertwined. To leverage the coupled nature of these two tasks, we introduce the semantic scene completion network Abstract[PDF](SSCNet), an end-to-end 3D convolutional network that takes a single depth image as input and simultaneously outputs occupancy and semantic labels for all voxels in the camera view frustum. Our network uses a dilation-based 3D context module to efficiently expand the receptive field and enable 3D context learning. To train our network, we construct SUNCG - a manually created large-scale dataset of synthetic 3D scenes with dense volumetric annotations. Our experiments demonstrate that the joint model outperforms methods addressing each task in isolation and outperforms alternative approaches on the semantic scene completion task. ]_

Comment< >

#### 3DMatch: Learning Local Geometric Descriptors From RGB-D ReconstructionsAbstract[PDF](https://arxiv.org/abs/1603.08182)Abstract[PDF](http://3dmatch.cs.princeton.edu/)

Andy Zeng, Shuran Song, Matthias NieÃŸner, Matthew Fisher, Jianxiong Xiao, Thomas Funkhouser

_Abstract[Matching local geometric features on real-world depth images is a challenging task due to the noisy, low-resolution, and incomplete nature of 3D scan data. These difficulties limit the performance of current state-of-art methods, which are typically based on histograms over geometric properties. In this paper, we present 3DMatch, a data-driven model that learns a local volumetric patch descriptor for establishing correspondences between partial 3D data. To amass training data for our model, we propose a self-supervised feature learning method that leverages the millions of correspondence labels found in existing RGB-D reconstructions. Experiments show that our descriptor is not only able to match local geometry in new scenes for reconstruction, but also generalize to different tasks and spatial scales Abstract[PDF](e.g. instance-level object model alignment for the Amazon Picking Challenge, and mesh surface correspondence). Results show that 3DMatch consistently outperforms other state-of-the-art approaches by a significant margin. Code, data, benchmarks, and pre-trained models are available online at this http URL]_

Comment< >

#### Multi-View Supervision for Single-View Reconstruction via Differentiable Ray Consistency Abstract[PDF](PDF, project,code)Abstract[PDF](https://arxiv.org/pdf/1704.06254.pdf)Abstract[PDF](https://shubhtuls.github.io/drc/)

Shubham Tulsiani, Tinghui Zhou, Alexei A. Efros, Jitendra Malik

_Abstract[We study the notion of consistency between a 3D shape and a 2D observation and propose a differentiable formulation which allows computing gradients of the 3D shape given an observation from an arbitrary view. We do so by reformulating view consistency using a differentiable ray consistency Abstract[PDF](DRC) term. We show that this formulation can be incorporated in a learning framework to leverage different types of multi-view observations e.g. foreground masks, depth, color images, semantics etc. as supervision for learning single-view 3D prediction. We present empirical analysis of our technique in a controlled setting. We also show that this approach allows us to improve over existing techniques for single-view reconstruction of objects from the PASCAL VOC dataset. ]_

Comment< >

#### On-The-Fly Adaptation of Regression Forests for Online Camera Relocalisation

Tommaso Cavallari, Stuart Golodetz, Nicholas A. Lord, Julien Valentin, Luigi Di Stefano, Philip H. S. Torr

_Abstract[]_

Comment< >

## Low- & Mid-Level Vision

Spotlight 1-1C

#### Designing Effective Inter-Pixel Information Flow for Natural Image Matting

YaÄŸiz Aksoy, TunÃ§ Ozan Aydin, Marc Pollefeys
Deep Video Deblurring for Hand-Held Cameras
Shuochen Su, Mauricio Delbracio, Jue Wang, Guillermo Sapiro, Wolfgang Heidrich, Oliver Wang
Instance-Level Salient Object Segmentation
Guanbin Li, Yuan Xie, Liang Lin, Yizhou Yu
Deep Multi-Scale Convolutional Neural Network for Dynamic Scene Deblurring
Seungjun Nah, Tae Hyun Kim, Kyoung Mu Lee
Diversified Texture Synthesis With Feed-Forward Networks
Yijun Li, Chen Fang, Jimei Yang, Zhaowen Wang, Xin Lu, Ming-Hsuan Yang
Radiometric Calibration for Internet Photo Collections Abstract[PDF](PDF)
Zhipeng Mo, Boxin Shi, Sai-Kit Yeung, Yasuyuki Matsushita
Deeply Aggregated Alternating Minimization for Image Restoration
Youngjung Kim, Hyungjoo Jung, Dongbo Min, Kwanghoon Sohn
End-To-End Instance Segmentation With Recurrent Attention
Mengye Ren, Richard S. Zemel
Oral 1-1C
SRN: Side-output Residual Network for Object Symmetry Detection in the Wild
Wei Ke, Jie Chen, Jianbin Jiao, Guoying Zhao, Qixiang Ye
Deep Image Matting Abstract[PDF](PDF, abstract)
Ning Xu, Brian Price, Scott Cohen, Thomas Huang
Wetness and Color From a Single Multispectral Image
Mihoko Shimano, Hiroki Okawa, Yuta Asano, Ryoma Bise, Ko Nishino, Imari Sato
FC4: Fully Convolutional Color Constancy With Confidence-Weighted Pooling
Yuanming Hu, Baoyuan Wang, Stephen Lin
Poster 1-1
3D Computer Vision
Face Normals â€œIn-The-Wildâ€ Using Fully Convolutional Networks
George Trigeorgis, Patrick Snape, Iasonas Kokkinos, Stefanos Zafeiriou
A Non-Convex Variational Approach to Photometric Stereo Under Inaccurate Lighting
Yvain Quéau, Tao Wu, FranÃ§ois Lauze, Jean-Denis Durou, Daniel Cremers
A Linear Extrinsic Calibration of Kaleidoscopic Imaging System From Single 3D Point
Kosuke Takahashi, Akihiro Miyata, Shohei Nobuhara, Takashi Matsuyama
Polarimetric Multi-View Stereo
Zhaopeng Cui, Jinwei Gu, Boxin Shi, Ping Tan, Jan Kautz
An Exact Penalty Method for Locally Convergent Maximum Consensus Abstract[PDF](PDF, code)
Huu Le, Tat-Jun Chin, David Suter
Deep Supervision With Shape Concepts for Occlusion-Aware 3D Object Parsing
Chi Li, M. Zeeshan Zia, Quoc-Huy Tran, Xiang Yu, Gregory D. Hager, Manmohan Chandraker
Amodal Detection of 3D Objects: Inferring 3D Bounding Boxes From 2D Ones in RGB-Depth Images
Zhuo Deng, Longin Jan Latecki
Analyzing Humans in Images
Transition Forests: Learning Discriminative Temporal Transitions for Action Recognition and Detection
Guillermo Garcia-Hernando, Tae-Kyun Kim
Scene Flow to Action Map: A New Representation for RGB-D Based Action Recognition With Convolutional Neural Networks
Pichao Wang, Wanqing Li, Zhimin Gao, Yuyao Zhang, Chang Tang, Philip Ogunbona
Detecting Masked Faces in the Wild With LLE-CNNs
Shiming Ge, Jia Li, Qiting Ye, Zhao Luo
A Domain Based Approach to Social Relation Recognition
Qianru Sun, Bernt Schiele, Mario Fritz
Spatio-Temporal Naive-Bayes Nearest-Neighbor Abstract[PDF](ST-NBNN) for Skeleton-Based Action Recognition
Junwu Weng, Chaoqun Weng, Junsong Yuan
Personalizing Gesture Recognition Using Hierarchical Bayesian Neural Networks
Ajjen Joshi, Soumya Ghosh, Margrit Betke, Stan Sclaroff, Hanspeter Pfister
Applications
Real-Time 3D Model Tracking in Color and Depth on a Single CPU Core
Wadim Kehl, Federico Tombari, Slobodan Ilic, Nassir Navab
Multi-Scale FCN With Cascaded Instance Aware Segmentation for Arbitrary Oriented Word Spotting in the Wild
Dafang He, Xiao Yang, Chen Liang, Zihan Zhou, Alexander G. Ororbi II, Daniel Kifer, C. Lee Giles
Viraliency: Pooling Local Virality
Xavier Alameda-Pineda, Andrea Pilzer, Dan Xu, Nicu Sebe, Elisa Ricci
Biomedical Image/Video Analysis
A Non-Local Low-Rank Framework for Ultrasound Speckle Reduction
Lei Zhu, Chi-Wing Fu, Michael S. Brown, Pheng-Ann Heng
Image Motion & Tracking
Video Acceleration Magnification
Yichao Zhang, Silvia L. Pintea, Jan C. van Gemert
Superpixel-Based Tracking-By-Segmentation Using Markov Chains
Donghun Yeo, Jeany Son, Bohyung Han, Joon Hee Han
BranchOut: Regularization for Online Ensemble Tracking With Convolutional Neural Networks
Bohyung Han, Jack Sim, Hartwig Adam
Learning Motion Patterns in Videos
Pavel Tokmakov, Karteek Alahari, Cordelia Schmid
Low- & Mid-Level Vision
Deep Level Sets for Salient Object Detection
Ping Hu, Bing Shuai, Jun Liu, Gang Wang
Binary Constraint Preserving Graph Matching
Bo Jiang, Jin Tang, Chris Ding, Bin Luo
From Local to Global: Edge Profiles to Camera Motion in Blurred Images
Subeesh Vasu, A. N. Rajagopalan
What Is the Space of Attenuation Coefficients in Underwater Computer Vision?
Derya Akkaynak, Tali Treibitz, Tom Shlesinger, Yossi Loya, Raz Tamir, David Iluz
Robust Energy Minimization for BRDF-Invariant Shape From Light Fields
Zhengqin Li, Zexiang Xu, Ravi Ramamoorthi, Manmohan Chandraker
Boundary-Aware Instance Segmentation
Zeeshan Hayder, Xuming He, Mathieu Salzmann
Spatially-Varying Blur Detection Based on Multiscale Fused and Sorted Transform Coefficients of Gradient Magnitudes
S. Alireza Golestaneh, Lina J. Karam
Model-Based Iterative Restoration for Binary Document Image Compression With Dictionary Learning
Yandong Guo, Cheng Lu, Jan P. Allebach, Charles A. Bouman
FCSS: Fully Convolutional Self-Similarity for Dense Semantic Correspondence
Seungryong Kim, Dongbo Min, Bumsub Ham, Sangryul Jeon, Stephen Lin, Kwanghoon Sohn
Machine Learning
Learning by Association â€” A Versatile Semi-Supervised Training Method for Neural Networks
Philip Haeusser, Alexander Mordvintsev, Daniel Cremers
Dilated Residual Networks
Fisher Yu, Vladlen Koltun, Thomas Funkhouser
Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction
Richard Zhang, Phillip Isola, Alexei A. Efros
Nonnegative Matrix Underapproximation for Robust Multiple Model Fitting
Mariano Tepper, Guillermo Sapiro
Truncated Max-Of-Convex Models
Pankaj Pansari, M. Pawan Kumar
Additive Component Analysis
Calvin Murdock, Fernando De la Torre
Subspace Clustering via Variance Regularized Ridge Regression
Chong Peng, Zhao Kang, Qiang Cheng
The Incremental Multiresolution Matrix Factorization Algorithm
Vamsi K. Ithapu, Risi Kondor, Sterling C. Johnson, Vikas Singh
Transformation-Grounded Image Generation Network for Novel 3D View Synthesis
Eunbyung Park, Jimei Yang, Ersin Yumer, Duygu Ceylan, Alexander C. Berg
Learning Dynamic Guidance for Depth Image Enhancement Abstract[PDF](PDF)
Shuhang Gu, Wangmeng Zuo, Shi Guo, Yunjin Chen, Chongyu Chen, Lei Zhang
A-Lamp: Adaptive Layout-Aware Multi-Patch Deep Convolutional Neural Network for Photo Aesthetic Assessment Abstract[PDF](PDF)
Shuang Ma, Jing Liu, Chang Wen Chen
Teaching Compositionality to CNNs
Austin Stone, Huayan Wang, Michael Stark, Yi Liu, D. Scott Phoenix, Dileep George
Using Ranking-CNN for Age Estimation
Shixing Chen, Caojin Zhang, Ming Dong, Jialiang Le, Mike Rao
Accurate Single Stage Detector Using Recurrent Rolling Convolution
Jimmy Ren, Xiaohao Chen, Jianbo Liu, Wenxiu Sun, Jiahao Pang, Qiong Yan, Yu-Wing Tai, Li Xu
A Compact DNN: Approaching GoogLeNet-Level Accuracy of Classification and Domain Adaptation
Chunpeng Wu, Wei Wen, Tariq Afzal, Yongmei Zhang, Yiran Chen, Hai Abstract[PDF](Helen) Li
The Impact of Typicality for Informative Representative Selection
Jawadul H. Bappy, Sujoy Paul, Ertem Tuncel, Amit K. Roy-Chowdhury
Infinite Variational Autoencoder for Semi-Supervised Learning
M. Ehsan Abbasnejad, Anthony Dick, Anton van den Hengel
SurfNet: Generating 3D Shape Surfaces Using Deep Residual Networks
Ayan Sinha, Asim Unmesh, Qixing Huang, Karthik Ramani
Intrinsic Grassmann Averages for Online Linear and Robust Subspace Learning
Rudrasis Chakraborty, SÃ¸ren Hauberg, Baba C. Vemuri
Variational Bayesian Multiple Instance Learning With Gaussian Processes
Manuel HauÃŸmann, Fred A. Hamprecht, Melih Kandemir
Temporal Attention-Gated Model for Robust Sequence Classification
Wenjie Pei, Tadas BaltruÅ¡aitis, David M.J. Tax, Louis-Philippe Morency
Non-Uniform Subset Selection for Active Learning in Structured Data
Sujoy Paul, Jawadul H. Bappy, Amit K. Roy-Chowdhury
Colorization as a Proxy Task for Visual Understanding
Gustav Larsson, Michael Maire, Gregory Shakhnarovich
Shading Annotations in the Wild
Balazs Kovacs, Sean Bell, Noah Snavely, Kavita Bala
LCNN: Lookup-Based Convolutional Neural Network
Hessam Bagherinezhad, Mohammad Rastegari, Ali Farhadi
Object Recognition & Scene Understanding
Physics Inspired Optimization on Semantic Transfer Features: An Alternative Method for Room Layout Estimation
Hao Zhao, Ming Lu, Anbang Yao, Yiwen Guo, Yurong Chen, Li Zhang
Pixelwise Instance Segmentation With a Dynamically Instantiated Network
Anurag Arnab, Philip H. S. Torr
Object Detection in Videos With Tubelet Proposal Networks
Kai Kang, Hongsheng Li, Tong Xiao, Wanli Ouyang, Junjie Yan, Xihui Liu, Xiaogang Wang
AMVH: Asymmetric Multi-Valued Hashing
Cheng Da, Shibiao Xu, Kun Ding, Gaofeng Meng, Shiming Xiang, Chunhong Pan
Spindle Net: Person Re-Identification With Human Body Region Guided Feature Decomposition and Fusion
Haiyu Zhao, Maoqing Tian, Shuyang Sun, Jing Shao, Junjie Yan, Shuai Yi, Xiaogang Wang, Xiaoou Tang
Deep Visual-Semantic Quantization for Efficient Image Retrieval
Yue Cao, Mingsheng Long, Jianmin Wang, Shichen Liu
Efficient Diffusion on Region Manifolds: Recovering Small Objects With Compact CNN Representations
Ahmet Iscen, Giorgos Tolias, Yannis Avrithis, Teddy Furon, OndÅ™ej Chum
Feature Pyramid Networks for Object Detection
Tsung-Yi Lin, Piotr DollÃ¡r, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie
Mind the Class Weight Bias: Weighted Maximum Mean Discrepancy for Unsupervised Domain Adaptation
Hongliang Yan, Yukang Ding, Peihua Li, Qilong Wang, Yong Xu, Wangmeng Zuo
StyleNet: Generating Attractive Visual Captions With Styles
Chuang Gan, Zhe Gan, Xiaodong He, Jianfeng Gao, Li Deng
Fine-Grained Recognition of Thousands of Object Categories With Single-Example Training
Leonid Karlinsky, Joseph Shtok, Yochay Tzur, Asaf Tzadok
Improving Interpretability of Deep Neural Networks With Semantic Information
Yinpeng Dong, Hang Su, Jun Zhu, Bo Zhang
Video Captioning With Transferred Semantic Attributes
Yingwei Pan, Ting Yao, Houqiang Li, Tao Mei
Fast Boosting Based Detection Using Scale Invariant Multimodal Multiresolution Filtered Features
Arthur Daniel Costea, Robert Varga, Sergiu Nedevschi
Video Analytics
Temporal Convolutional Networks for Action Segmentation and Detection
Colin Lea, Michael D. Flynn, René Vidal, Austin Reiter, Gregory D. Hager
Surveillance Video Parsing With Single Frame Supervision
Si Liu, Changhu Wang, Ruihe Qian, Han Yu, Renda Bao, Yao Sun
Weakly Supervised Actor-Action Segmentation via Robust Multi-Task Ranking
Yan Yan, Chenliang Xu, Dawen Cai, Jason J. Corso
Unsupervised Visual-Linguistic Reference Resolution in Instructional Videos
De-An Huang, Joseph J. Lim, Li Fei-Fei, Juan Carlos Niebles
Zero-Shot Action Recognition With Error-Correcting Output Codes
Jie Qin, Li Liu, Ling Shao, Fumin Shen, Bingbing Ni, Jiaxin Chen, Yunhong Wang
Enhancing Video Summarization via Vision-Language Embedding
Bryan A. Plummer, Matthew Brown, Svetlana Lazebnik
Synthesizing Dynamic Patterns by Spatial-Temporal Generative ConvNet
Jianwen Xie, Song-Chun Zhu, Ying Nian Wu
Object Recognition & Scene Understanding - Computer Vision & Language
Discriminative Bimodal Networks for Visual Localization and Detection With Natural Language Queries
Yuting Zhang, Luyao Yuan, Yijie Guo, Zhiyuan He, I-An Huang, Honglak Lee
Automatic Understanding of Image and Video Advertisements
Zaeem Hussain, Mingda Zhang, Xiaozhong Zhang, Keren Ye, Christopher Thomas, Zuha Agha, Nathan Ong, Adriana Kovashka
Deep Sketch Hashing: Fast Free-Hand Sketch-Based Image Retrieval
Li Liu, Fumin Shen, Yuming Shen, Xianglong Liu, Ling Shao
Discover and Learn New Objects From Documentaries
Kai Chen, Hang Song, Chen Change Loy, Dahua Lin
Spatial-Semantic Image Search by Visual Feature Synthesis
Long Mai, Hailin Jin, Zhe Lin, Chen Fang, Jonathan Brandt, Feng Liu
Fully-Adaptive Feature Sharing in Multi-Task Networks With Applications in Person Attribute Classification
Yongxi Lu, Abhishek Kumar, Shuangfei Zhai, Yu Cheng, Tara Javidi, Rogerio Feris
Semantic Compositional Networks for Visual Captioning
Zhe Gan, Chuang Gan, Xiaodong He, Yunchen Pu, Kenneth Tran, Jianfeng Gao, Lawrence Carin, Li Deng
Training Object Class Detectors With Click Supervision
Dim P. Papadopoulos, Jasper R. R. Uijlings, Frank Keller, Vittorio Ferrari
Oral 1-2A
Deep Reinforcement Learning-Based Image Captioning With Embedding Reward
Zhou Ren, Xiaoyu Wang, Ning Zhang, Xutao Lv, Li-Jia Li
From Red Wine to Red Tomato: Composition With Context
Ishan Misra, Abhinav Gupta, Martial Hebert
Captioning Images With Diverse Objects
Subhashini Venugopalan, Lisa Anne Hendricks, Marcus Rohrbach, Raymond Mooney, Trevor Darrell, Kate Saenko
Self-Critical Sequence Training for Image Captioning
Steven J. Rennie, Etienne Marcheret, Youssef Mroueh, Jerret Ross, Vaibhava Goel
Analyzing Humans 1
Spotlight 1-2B
Crossing Nets: Combining GANs and VAEs With a Shared Latent Space for Hand Pose Estimation
Chengde Wan, Thomas Probst, Luc Van Gool, Angela Yao
Predicting Behaviors of Basketball Players From First Person Videos
Shan Su, Jung Pyo Hong, Jianbo Shi, Hyun Soo Park
LCR-Net: Localization-Classification-Regression for Human Pose
Grégory Rogez, Philippe Weinzaepfel, Cordelia Schmid
Learning Residual Images for Face Attribute Manipulation
Wei Shen, Rujie Liu
Seeing What Is Not There: Learning Context to Determine Where Objects Are Missing
Jin Sun, David W. Jacobs
Deep Learning on Lie Groups for Skeleton-Based Action Recognition
Zhiwu Huang, Chengde Wan, Thomas Probst, Luc Van Gool
Harvesting Multiple Views for Marker-Less 3D Human Pose Annotations
Georgios Pavlakos, Xiaowei Zhou, Konstantinos G. Derpanis, Kostas Daniilidis
Coarse-To-Fine Volumetric Prediction for Single-Image 3D Human Pose
Georgios Pavlakos, Xiaowei Zhou, Konstantinos G. Derpanis, Kostas Daniilidis
Oral 1-2B
Weakly Supervised Action Learning With RNN Based Fine-To-Coarse Modeling
Alexander Richard, Hilde Kuehne, Juergen Gall
Disentangled Representation Learning GAN for Pose-Invariant Face Recognition
Luan Tran, Xi Yin, Xiaoming Liu
ArtTrack: Articulated Multi-Person Tracking in the Wild
Eldar Insafutdinov, Mykhaylo Andriluka, Leonid Pishchulin, Siyu Tang, Evgeny Levinkov, Bjoern Andres, Bernt Schiele
Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields Abstract[PDF](PDF, code)
Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh
Image Motion & Tracking; Video Analysis
Spotlight 1-2C
Template Matching With Deformable Diversity Similarity
Itamar Talmi, Roey Mechrez, Lihi Zelnik-Manor
Beyond Triplet Loss: A Deep Quadruplet Network for Person Re-Identification
Weihua Chen, Xiaotang Chen, Jianguo Zhang, Kaiqi Huang
Agent-Centric Risk Assessment: Accident Anticipation and Risky Region Localization
Kuo-Hao Zeng, Shih-Han Chou, Fu-Hsiang Chan, Juan Carlos Niebles, Min Sun
Bidirectional Multirate Reconstruction for Temporal Modeling in Videos
Linchao Zhu, Zhongwen Xu, Yi Yang
Action-Decision Networks for Visual Tracking With Deep Reinforcement Learning
Sangdoo Yun, Jongwon Choi, Youngjoon Yoo, Kimin Yun, Jin Young Choi
TGIF-QA: Toward Spatio-Temporal Reasoning in Visual Question Answering
Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, Gunhee Kim
Making 360° Video Watchable in 2D: Learning Videography for Click Free Viewing
Yu-Chuan Su, Kristen Grauman
Unsupervised Adaptive Re-Identification in Open World Dynamic Camera Networks
Rameswar Panda, Amran Bhuiyan, Vittorio Murino, Amit K. Roy-Chowdhury
Oral 1-2C
Context-Aware Correlation Filter Tracking
Matthias Mueller, Neil Smith, Bernard Ghanem
Deep 360 Pilot: Learning a Deep Agent for Piloting Through 360° Sports Videos
Hou-Ning Hu, Yen-Chen Lin, Ming-Yu Liu, Hsien-Tzu Cheng, Yung-Ju Chang, Min Sun
Slow Flow: Exploiting High-Speed Cameras for Accurate and Diverse Optical Flow Reference Data
Joel Janai, Fatma Güney, Jonas Wulff, Michael J. Black, Andreas Geiger
CDC: Convolutional-De-Convolutional Networks for Precise Temporal Action Localization in Untrimmed Videos
Zheng Shou, Jonathan Chan, Alireza Zareian, Kazuyuki Miyazawa, Shih-Fu Chang
Poster 1-2
3D Computer Vision
Exploiting 2D Floorplan for Building-Scale Panorama RGBD Alignment
Erik Wijmans, Yasutaka Furukawa
A Combinatorial Solution to Non-Rigid 3D Shape-To-Image Matching
Florian Bernard, Frank R. Schmidt, Johan Thunberg, Daniel Cremers
NID-SLAM: Robust Monocular SLAM Using Normalised Information Distance
Geoffrey Pascoe, Will Maddern, Michael Tanner, Pedro Piniés, Paul Newman
End-To-End Training of Hybrid CNN-CRF Models for Stereo
Patrick KnÃ¶belreiter, Christian Reinbacher, Alexander Shekhovtsov, Thomas Pock
Learning Shape Abstractions by Assembling Volumetric Primitives
Shubham Tulsiani, Hao Su, Leonidas J. Guibas, Alexei A. Efros, Jitendra Malik
Locality-Sensitive Deconvolution Networks With Gated Fusion for RGB-D Indoor Semantic Segmentation
Yanhua Cheng, Rui Cai, Zhiwei Li, Xin Zhao, Kaiqi Huang
Acquiring Axially-Symmetric Transparent Objects Using Single-View Transmission Imaging
Jaewon Kim, Ilya Reshetouski, Abhijeet Ghosh
Regressing Robust and Discriminative 3D Morphable Models With a Very Deep Neural Network
Anh Tuáº¥n Tráº§n, Tal Hassner, Iacopo Masi, Gérard Medioni
End-To-End 3D Face Reconstruction With Deep Neural Networks
Pengfei Dou, Shishir K. Shah, Ioannis A. Kakadiaris
DUST: Dual Union of Spatio-Temporal Subspaces for Monocular Multiple Object 3D Reconstruction
Antonio Agudo, Francesc Moreno-Noguer
Analyzing Humans in Images
Finding Tiny Faces
Peiyun Hu, Deva Ramanan
Dynamic Facial Analysis: From Bayesian Filtering to Recurrent Neural Network
Jinwei Gu, Xiaodong Yang, Shalini De Mello, Jan Kautz
Deep Temporal Linear Encoding Networks
Ali Diba, Vivek Sharma, Luc Van Gool
Joint Registration and Representation Learning for Unconstrained Face Identification
Munawar Hayat, Salman H. Khan, Naoufel Werghi, Roland Goecke
3D Human Pose Estimation From a Single Image via Distance Matrix Regression
Francesc Moreno-Noguer
One-Shot Metric Learning for Person Re-Identification
Slawomir BÄ…k, Peter Carr
Generalized Rank Pooling for Activity Recognition
Anoop Cherian, Basura Fernando, Mehrtash Harandi, Stephen Gould
Deep Representation Learning for Human Motion Prediction and Classification
Judith BÃ¼tepage, Michael J. Black, Danica Kragic, Hedvig KjellstrÃ¶m
Interspecies Knowledge Transfer for Facial Keypoint Detection
Maheen Rashid, Xiuye Gu, Yong Jae Lee
Recurrent Convolutional Neural Networks for Continuous Sign Language Recognition by Staged Optimization
Runpeng Cui, Hu Liu, Changshui Zhang
Applications
Modeling Sub-Event Dynamics in First-Person Action Recognition
Hasan F. M. Zaki, Faisal Shafait, Ajmal Mian
Computational Photography
Turning an Urban Scene Video Into a Cinemagraph
Hang Yan, Yebin Liu, Yasutaka Furukawa
Light Field Reconstruction Using Deep Convolutional Network on EPI
Gaochang Wu, Mandan Zhao, Liangyong Wang, Qionghai Dai, Tianyou Chai, Yebin Liu
Image Motion & Tracking
FlowNet 2.0: Evolution of Optical Flow Estimation With Deep Networks
Eddy Ilg, Nikolaus Mayer, Tonmoy Saikia, Margret Keuper, Alexey Dosovitskiy, Thomas Brox
Low- & Mid-Level Vision
Attention-Aware Face Hallucination via Deep Reinforcement Learning
Qingxing Cao, Liang Lin, Yukai Shi, Xiaodan Liang, Guanbin Li
Simple Does It: Weakly Supervised Instance and Semantic Segmentation
Anna Khoreva, Rodrigo Benenson, Jan Hosang, Matthias Hein, Bernt Schiele
Anti-Glare: Tightly Constrained Optimization for Eyeglass Reflection Removal
Tushar Sandhan, Jin Young Choi
Deep Joint Rain Detection and Removal From a Single Image
Wenhan Yang, Robby T. Tan, Jiashi Feng, Jiaying Liu, Zongming Guo, Shuicheng Yan
Radiometric Calibration From Faces in Images
Chen Li, Stephen Lin, Kun Zhou, Katsushi Ikeuchi
Webly Supervised Semantic Segmentation
Bin Jin, Maria V. Ortiz Segovia, Sabine SÃ¼sstrunk
Removing Rain From Single Images via a Deep Detail Network
Xueyang Fu, Jiabin Huang, Delu Zeng, Yue Huang, Xinghao Ding, John Paisley
Deep Crisp Boundaries
Yupei Wang, Xin Zhao, Kaiqi Huang
Coarse-To-Fine Segmentation With Shape-Tailored Continuum Scale Spaces
Naeemullah Khan, Byung-Woo Hong, Anthony Yezzi, Ganesh Sundaramoorthi
Large Kernel Matters â€” Improve Semantic Segmentation by Global Convolutional Network
Chao Peng, Xiangyu Zhang, Gang Yu, Guiming Luo, Jian Sun
Single Image Reflection Suppression
Nikolaos Arvanitopoulos, Radhakrishna Achanta, Sabine SÃ¼sstrunk
CASENet: Deep Category-Aware Semantic Edge Detection
Zhiding Yu, Chen Feng, Ming-Yu Liu, Srikumar Ramalingam
Reflectance Adaptive Filtering Improves Intrinsic Image Estimation
Thomas Nestmeyer, Peter V. Gehler
Machine Learning
Conditional Similarity Networks
Andreas Veit, Serge Belongie, Theofanis Karaletsos
Spatially Adaptive Computation Time for Residual Networks
Michael Figurnov, Maxwell D. Collins, Yukun Zhu, Li Zhang, Jonathan Huang, Dmitry Vetrov, Ruslan Salakhutdinov
Xception: Deep Learning With Depthwise Separable Convolutions
FranÃ§ois Chollet
Feedback Networks
Amir R. Zamir, Te-Lin Wu, Lin Sun, William B. Shen, Bertram E. Shi, Jitendra Malik, Silvio Savarese
Online Summarization via Submodular and Convex Optimization
Ehsan Elhamifar, M. Clara De Paolis Kaluza
Deep MANTA: A Coarse-To-Fine Many-Task Network for Joint 2D and 3D Vehicle Analysis From Monocular Image
Florian Chabot, Mohamed Chaouch, Jaonary Rabarisoa, Céline Teulière, Thierry Chateau
Improving Pairwise Ranking for Multi-Label Image Classification
Yuncheng Li, Yale Song, Jiebo Luo
Active Convolution: Learning the Shape of Convolution for Image Classification
Yunho Jeon, Junmo Kim
Linking Image and Text With 2-Way Nets
Aviv Eisenschtat, Lior Wolf
Stacked Generative Adversarial Networks
Xun Huang, Yixuan Li, Omid Poursaeed, John Hopcroft, Serge Belongie
Image Splicing Detection via Camera Response Function Analysis
Can Chen, Scott McCloskey, Jingyi Yu
Building a Regular Decision Boundary With Deep Networks
Edouard Oyallon
More Is Less: A More Complicated Network With Less Inference Complexity
Xuanyi Dong, Junshi Huang, Yi Yang, Shuicheng Yan
Joint Graph Decomposition and Node Labeling: Problem, Algorithms, Applications
Evgeny Levinkov, Jonas Uhrig, Siyu Tang, Mohamed Omran, Eldar Insafutdinov, Alexander Kirillov, Carsten Rother, Thomas Brox, Bernt Schiele, Bjoern Andres
Scale-Aware Face Detection
Zekun Hao, Yu Liu, Hongwei Qin, Junjie Yan, Xiu Li, Xiaolin Hu
Deep Unsupervised Similarity Learning Using Partially Ordered Sets
Miguel A. Bautista, Artsiom Sanakoyeu, BjÃ¶rn Ommer
Generative Hierarchical Learning of Sparse FRAME Models
Jianwen Xie, Yifei Xu, Erik Nijkamp, Ying Nian Wu, Song-Chun Zhu
Object Recognition & Scene Understanding
Generating Holistic 3D Scene Abstractions for Text-Based Image Retrieval
Ang Li, Jin Sun, Joe Yue-Hei Ng, Ruichi Yu, Vlad I. Morariu, Larry S. Davis
Perceptual Generative Adversarial Networks for Small Object Detection
Jianan Li, Xiaodan Liang, Yunchao Wei, Tingfa Xu, Jiashi Feng, Shuicheng Yan
Emotion Recognition in Context
Ronak Kosti, Jose M. Alvarez, Adria Recasens, Agata Lapedriza
Deep Learning of Human Visual Sensitivity in Image Quality Assessment Framework
Jongyoo Kim, Sanghoon Lee
Dense Captioning With Joint Inference and Visual Context
Linjie Yang, Kevin Tang, Jianchao Yang, Li-Jia Li
CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning
Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, Ross Girshick
Cross-View Image Matching for Geo-Localization in Urban Environments
Yicong Tian, Chen Chen, Mubarak Shah
Matrix Tri-Factorization With Manifold Regularizations for Zero-Shot Learning
Xing Xu, Fumin Shen, Yang Yang, Dongxiang Zhang, Heng Tao Shen, Jingkuan Song
Self-Supervised Learning of Visual Features Through Embedding Images Into Text Topic Spaces
Lluis Gomez, Yash Patel, MarÃ§al Rusiñol, Dimosthenis Karatzas, C. V. Jawahar
Learning Spatial Regularization With Image-Level Supervisions for Multi-Label Image Classification
Feng Zhu, Hongsheng Li, Wanli Ouyang, Nenghai Yu, Xiaogang Wang
Semantically Consistent Regularization for Zero-Shot Recognition
Pedro Morgado, Nuno Vasconcelos
Can Walking and Measuring Along Chord Bunches Better Describe Leaf Shapes?
Bin Wang, Yongsheng Gao, Changming Sun, Michael Blumenstein, John La Salle
Video Analytics
Self-Learning Scene-Specific Pedestrian Detectors Using a Progressive Latent Model
Qixiang Ye, Tianliang Zhang, Wei Ke, Qiang Qiu, Jie Chen, Guillermo Sapiro, Baochang Zhang
Predictive-Corrective Networks for Action Detection
Achal Dave, Olga Russakovsky, Deva Ramanan
Budget-Aware Deep Semantic Video Segmentation
Behrooz Mahasseni, Sinisa Todorovic, Alan Fern
Unified Embedding and Metric Learning for Zero-Exemplar Event Detection
Noureldien Hussein, Efstratios Gavves, Arnold W.M. Smeulders
Spatiotemporal Pyramid Network for Video Action Recognition
Yunbo Wang, Mingsheng Long, Jianmin Wang, Philip S. Yu
ER3: A Unified Framework for Event Retrieval, Recognition and Recounting
Zhanning Gao, Gang Hua, Dongqing Zhang, Nebojsa Jojic, Le Wang, Jianru Xue, Nanning Zheng
FusionSeg: Learning to Combine Motion and Appearance for Fully Automatic Segmentation of Generic Objects in Videos
Suyog Dutt Jain, Bo Xiong, Kristen Grauman
Query-Focused Video Summarization: Dataset, Evaluation, and a Memory Network Based Approach
Aidean Sharghi, Jacob S. Laurel, Boqing Gong
Flexible Spatio-Temporal Networks for Video Prediction
Chaochao Lu, Michael Hirsch, Bernhard SchÃ¶lkopf
Temporal Action Co-Segmentation in 3D Motion Capture Data and Videos
Konstantinos Papoutsakis, Costas Panagiotakis, Antonis A. Argyros
Machine Learning 2
Spotlight 2-1A
Dual Attention Networks for Multimodal Reasoning and Matching
Hyeonseob Nam, Jung-Woo Ha, Jeonghee Kim
DESIRE: Distant Future Prediction in Dynamic Scenes With Interacting Agents
Namhoon Lee, Wongun Choi, Paul Vernaza, Christopher B. Choy, Philip H. S. Torr, Manmohan Chandraker
Interpretable Structure-Evolving LSTM
Xiaodan Liang, Liang Lin, Xiaohui Shen, Jiashi Feng, Shuicheng Yan, Eric P. Xing
ShapeOdds: Variational Bayesian Learning of Generative Shape Models
Shireen Elhabian, Ross Whitaker
Fast Video Classification via Adaptive Cascading of Deep Models
Haichen Shen, Seungyeop Han, Matthai Philipose, Arvind Krishnamurthy
Deep Metric Learning via Facility Location
Hyun Oh Song, Stefanie Jegelka, Vivek Rathod, Kevin Murphy
Semi-Supervised Deep Learning for Monocular Depth Map Prediction
Yevhen Kuznietsov, JÃ¶rg StÃ¼ckler, Bastian Leibe
Weakly Supervised Semantic Segmentation Using Web-Crawled Videos
Seunghoon Hong, Donghun Yeo, Suha Kwak, Honglak Lee, Bohyung Han
Oral 2-1A
Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach
Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, Lizhen Qu
Learning From Simulated and Unsupervised Images Through Adversarial Training
Ashish Shrivastava, Tomas Pfister, Oncel Tuzel, Joshua Susskind, Wenda Wang, Russell Webb
Inverse Compositional Spatial Transformer Networks
Chen-Hsuan Lin, Simon Lucey
Densely Connected Convolutional Networks
Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
Computational Photography
Spotlight 2-1B
Visual Dialog
Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, José M. F. Moura, Devi Parikh, Dhruv Batra
Video Frame Interpolation via Adaptive Convolution
Simon Niklaus, Long Mai, Feng Liu
FastMask: Segment Multi-Scale Object Candidates in One Shot
Hexiang Hu, Shiyi Lan, Yuning Jiang, Zhimin Cao, Fei Sha
Reconstructing Transient Images From Single-Photon Sensors
Matthew O'Toole, Felix Heide, David B. Lindell, Kai Zang, Steven Diamond, Gordon Wetzstein
DeshadowNet: A Multi-Context Embedding Deep Network for Shadow Removal
Liangqiong Qu, Jiandong Tian, Shengfeng He, Yandong Tang, Rynson W. H. Lau
Illuminant-Camera Communication to Observe Moving Objects Under Strong External Light by Spread Spectrum Modulation
Ryusuke Sagawa, Yutaka Satoh
Photorealistic Facial Texture Inference Using Deep Neural Networks
Shunsuke Saito, Lingyu Wei, Liwen Hu, Koki Nagano, Hao Li
The Geometry of First-Returning Photons for Non-Line-Of-Sight Imaging
Chia-Yin Tsai, Kiriakos N. Kutulakos, Srinivasa G. Narasimhan, Aswin C. Sankaranarayanan
Oral 2-1B
Unrolling the Shutter: CNN to Correct Motion Distortions
Vijay Rengarajan, Yogesh Balaji, A. N. Rajagopalan
Light Field Blind Motion Deblurring
Pratul P. Srinivasan, Ren Ng, Ravi Ramamoorthi
Computational Imaging on the Electric Grid
Mark Sheinin, Yoav Y. Schechner, Kiriakos N. Kutulakos
Deep Outdoor Illumination Estimation
Yannick Hold-Geoffroy, Kalyan Sunkavalli, Sunil Hadap, Emiliano Gambaretto, Jean-FranÃ§ois Lalonde
3D Vision 2
Spotlight 2-1C
Efficient Solvers for Minimal Problems by Syzygy-Based Reduction
Viktor Larsson, Kalle Ã…strÃ¶m, Magnus Oskarsson
HSfM: Hybrid Structure-from-Motion
Hainan Cui, Xiang Gao, Shuhan Shen, Zhanyi Hu
Efficient Global Point Cloud Alignment Using Bayesian Nonparametric Mixtures
Julian Straub, Trevor Campbell, Jonathan P. How, John W. Fisher III
A New Rank Constraint on Multi-View Fundamental Matrices, and Its Application to Camera Location Recovery
Soumyadip Sengupta, Tal Amir, Meirav Galun, Tom Goldstein, David W. Jacobs, Amit Singer, Ronen Basri
IM2CAD
Hamid Izadinia, Qi Shan, Steven M. Seitz
ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes
Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias NieÃŸner
Noise Robust Depth From Focus Using a Ring Difference Filter
Jaeheung Surh, Hae-Gon Jeon, Yunwon Park, Sunghoon Im, Hyowon Ha, In So Kweon
Group-Wise Point-Set Registration Based on Rényi's Second Order Entropy
Luis G. Sanchez Giraldo, Erion Hasanbelliu, Murali Rao, Jose C. Principe
Oral 2-1C
A Point Set Generation Network for 3D Object Reconstruction From a Single Image
Haoqiang Fan, Hao Su, Leonidas J. Guibas
3D Point Cloud Registration for Localization Using a Deep Neural Network Auto-Encoder
Gil Elbaz, Tamar Avraham, Anath Fischer
Flight Dynamics-Based Recovery of a UAV Trajectory Using Ground Cameras
Artem Rozantsev, Sudipta N. Sinha, Debadeepta Dey, Pascal Fua
DSAC - Differentiable RANSAC for Camera Localization
Eric Brachmann, Alexander Krull, Sebastian Nowozin, Jamie Shotton, Frank Michel, Stefan Gumhold, Carsten Rother
Poster 2-1
3D Computer Vision
Scalable Surface Reconstruction From Point Clouds With Extreme Scale and Density Diversity
Christian Mostegel, Rudolf Prettenthaler, Friedrich Fraundorfer, Horst Bischof
Synthesizing 3D Shapes via Modeling Multi-View Depth Maps and Silhouettes With Deep Generative Networks
Amir Arsalan Soltani, Haibin Huang, Jiajun Wu, Tejas D. Kulkarni, Joshua B. Tenenbaum
General Models for Rational Cameras and the Case of Two-Slit Projections
Matthew Trager, Bernd Sturmfels, John Canny, Martial Hebert, Jean Ponce
Accurate Depth and Normal Maps From Occlusion-Aware Focal Stack Symmetry
Michael Strecke, Anna Alperovich, Bastian Goldluecke
A Multi-View Stereo Benchmark With High-Resolution Images and Multi-Camera Videos
Thomas SchÃ¶ps, Johannes L. SchÃ¶nberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, Andreas Geiger
Non-Contact Full Field Vibration Measurement Based on Phase-Shifting
Hiroyuki Kayaba, Yuji Kokumai
A Minimal Solution for Two-View Focal-Length Estimation Using Two Affine Correspondences
Daniel Barath, Tekla Toth, Levente Hajder
PoseAgent: Budget-Constrained 6D Object Pose Estimation via Reinforcement Learning
Alexander Krull, Eric Brachmann, Sebastian Nowozin, Frank Michel, Jamie Shotton, Carsten Rother
An Efficient Background Term for 3D Reconstruction and Tracking With Smooth Surface Models
Mariano Jaimez, Thomas J. Cashman, Andrew Fitzgibbon, Javier Gonzalez-Jimenez, Daniel Cremers
Analyzing Humans in Images
Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild
Shan Li, Weihong Deng, JunPing Du
Procedural Generation of Videos to Train Deep Action Recognition Networks
César Roberto de Souza, Adrien Gaidon, Yohann Cabon, Antonio Manuel LÃ³pez
BigHand2.2M Benchmark: Hand Pose Dataset and State of the Art Analysis
Shanxin Yuan, Qi Ye, BjÃ¶rn Stenger, Siddhant Jain, Tae-Kyun Kim
DenseReg: Fully Convolutional Dense Shape Regression In-The-Wild
RÄ±za Alp GÃ¼ler, George Trigeorgis, Epameinondas Antonakos, Patrick Snape, Stefanos Zafeiriou, Iasonas Kokkinos
Adaptive Class Preserving Representation for Image Classification
Jian-Xun Mi, Qiankun Fu, Weisheng Li
Applications
Generalized Semantic Preserving Hashing for N-Label Cross-Modal Retrieval
Devraj Mandal, Kunal N. Chaudhury, Soma Biswas
EAST: An Efficient and Accurate Scene Text Detector
Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, Jiajun Liang
VidLoc: A Deep Spatio-Temporal Model for 6-DoF Video-Clip Relocalization
Ronald Clark, Sen Wang, Andrew Markham, Niki Trigoni, Hongkai Wen
Biomedical Image/Video Analysis
Improving RANSAC-Based Segmentation Through CNN Encapsulation
Dustin Morley, Hassan Foroosh
Computational Photography
Position Tracking for Virtual Reality Using Commodity WiFi
Manikanta Kotaru, Sachin Katti
Designing Illuminant Spectral Power Distributions for Surface Classification
Henryk Blasinski, Joyce Farrell, Brian Wandell
One-Shot Hyperspectral Imaging Using Faced Reflectors
Tsuyoshi Takatani, Takahito Aoto, Yasuhiro Mukaigawa
Image Motion & Tracking
Direct Photometric Alignment by Mesh Deformation
Kaimo Lin, Nianjuan Jiang, Shuaicheng Liu, Loong-Fah Cheong, Minh Do, Jiangbo Lu
CNN-Based Patch Matching for Optical Flow With Thresholded Hinge Embedding Loss
Christian Bailer, Kiran Varanasi, Didier Stricker
Optical Flow Estimation Using a Spatial Pyramid Network
Anurag Ranjan, Michael J. Black
Deep Network Flow for Multi-Object Tracking
Manmohan Chandraker, Paul Vernaza, Wongun Choi, Samuel Schulter
Low- & Mid-Level Vision
Material Classification Using Frequency- and Depth-Dependent Time-Of-Flight Distortion
Kenichiro Tanaka, Yasuhiro Mukaigawa, Takuya Funatomi, Hiroyuki Kubo, Yasuyuki Matsushita, Yasushi Yagi
Benchmarking Denoising Algorithms With Real Photographs
Tobias PlÃ¶tz, Stefan Roth
A Unified Approach of Multi-Scale Deep and Hand-Crafted Features for Defocus Estimation Abstract[PDF](PDF, project)
Jinsun Park, Yu-Wing Tai, Donghyeon Cho, In So Kweon
StyleBank: An Explicit Representation for Neural Image Style Transfer
Dongdong Chen, Lu Yuan, Jing Liao, Nenghai Yu, Gang Hua
Specular Highlight Removal in Facial Images
Chen Li, Stephen Lin, Kun Zhou, Katsushi Ikeuchi
Image Super-Resolution via Deep Recursive Residual Network
Ying Tai, Jian Yang, Xiaoming Liu
Deep Image Harmonization
Yi-Hsuan Tsai, Xiaohui Shen, Zhe Lin, Kalyan Sunkavalli, Xin Lu, Ming-Hsuan Yang
Learning Deep CNN Denoiser Prior for Image Restoration Abstract[PDF](PDF, code)
Kai Zhang, Wangmeng Zuo, Shuhang Gu, Lei Zhang
A Novel Tensor-Based Video Rain Streaks Removal Approach via Utilizing Discriminatively Intrinsic Priors
Tai-Xiang Jiang, Ting-Zhu Huang, Xi-Le Zhao, Liang-Jian Deng, Yao Wang
GMS: Grid-based Motion Statistics for Fast, Ultra-Robust Feature Correspondence
JiaWang Bian, Wen-Yan Lin, Yasuyuki Matsushita, Sai-Kit Yeung, Tan-Dat Nguyen, Ming-Ming Cheng
Video Desnowing and Deraining Based on Matrix Decomposition
Weihong Ren, Jiandong Tian, Zhi Han, Antoni Chan, Yandong Tang
Real-Time Video Super-Resolution With Spatio-Temporal Networks and Motion Compensation
Jose Caballero, Christian Ledig, Andrew Aitken, Alejandro Acosta, Johannes Totz, Zehan Wang, Wenzhe Shi
Deep Watershed Transform for Instance Segmentation
Min Bai, Raquel Urtasun
AnchorNet: A Weakly Supervised Network to Learn Geometry-Sensitive Features for Semantic Matching
David Novotny, Diane Larlus, Andrea Vedaldi
Learning Diverse Image Colorization
Aditya Deshpande, Jiajun Lu, Mao-Chuang Yeh, Min Jin Chong, David Forsyth
Awesome Typography: Statistics-Based Text Effects Transfer
Shuai Yang, Jiaying Liu, Zhouhui Lian, Zongming Guo
Machine Learning
Unsupervised Video Summarization With Adversarial LSTM Networks
Behrooz Mahasseni, Michael Lam, Sinisa Todorovic
Deep TEN: Texture Encoding Network
Hang Zhang, Jia Xue, Kristin Dana
Order-Preserving Wasserstein Distance for Sequence Matching
Bing Su, Gang Hua
A Dual Ascent Framework for Lagrangean Decomposition of Combinatorial Problems
Paul Swoboda, Jan Kuske, Bogdan Savchynskyy
Attend in Groups: A Weakly-Supervised Deep Learning Framework for Learning From Web Data
Bohan Zhuang, Lingqiao Liu, Yao Li, Chunhua Shen, Ian Reid
Hierarchical Multimodal Metric Learning for Multimodal Classification
Heng Zhang, Vishal M. Patel, Rama Chellappa
Efficient Linear Programming for Dense CRFs
Thalaiyasingam Ajanthan, Alban Desmaison, Rudy Bunel, Mathieu Salzmann, Philip H. S. Torr, M. Pawan Kumar
Variational Autoencoded Regression: High Dimensional Regression of Visual Data on Complex Manifold
YoungJoon Yoo, Sangdoo Yun, Hyung Jin Chang, Yiannis Demiris, Jin Young Choi
Learning Random-Walk Label Propagation for Weakly-Supervised Semantic Segmentation
Paul Vernaza, Manmohan Chandraker
Low-Rank-Sparse Subspace Representation for Robust Regression
Yongqiang Zhang, Daming Shi, Junbin Gao, Dansong Cheng
Object Recognition & Scene Understanding
Generating the Future With Adversarial Transformers
Carl Vondrick, Antonio Torralba
Semantic Amodal Segmentation
Yan Zhu, Yuandong Tian, Dimitris Metaxas, Piotr DollÃ¡r
Learning a Deep Embedding Model for Zero-Shot Learning
Li Zhang, Tao Xiang, Shaogang Gong
BIND: Binary Integrated Net Descriptors for Texture-Less Object Recognition
Jacob Chan, Jimmy Addison Lee, Qian Kemao
Growing a Brain: Fine-Tuning by Increasing Model Capacity
Yu-Xiong Wang, Deva Ramanan, Martial Hebert
A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection
Xiaolong Wang, Abhinav Shrivastava, Abhinav Gupta
Multiple Instance Detection Network With Online Instance Classifier Refinement
Peng Tang, Xinggang Wang, Xiang Bai, Wenyu Liu
Kernel Pooling for Convolutional Neural Networks
Yin Cui, Feng Zhou, Jiang Wang, Xiao Liu, Yuanqing Lin, Serge Belongie
Learning Cross-Modal Embeddings for Cooking Recipes and Food Images
Amaia Salvador, Nicholas Hynes, Yusuf Aytar, Javier Marin, Ferda Ofli, Ingmar Weber, Antonio Torralba
Zero-Shot Learning - the Good, the Bad and the Ugly
Yongqin Xian, Bernt Schiele, Zeynep Akata
DeepNav: Learning to Navigate Large Cities
Samarth Brahmbhatt, James Hays
Scene Graph Generation by Iterative Message Passing
Danfei Xu, Yuke Zhu, Christopher B. Choy, Li Fei-Fei
Visual Translation Embedding Network for Visual Relation Detection
Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, Tat-Seng Chua
Unsupervised Part Learning for Visual Recognition
Ronan Sicre, Yannis Avrithis, Ewa Kijak, Frédéric Jurie
Comprehension-Guided Referring Expressions
Ruotian Luo, Gregory Shakhnarovich
Top-Down Visual Saliency Guided by Captions
Vasili Ramanishka, Abir Das, Jianming Zhang, Kate Saenko
Theory
Grassmannian Manifold Optimization Assisted Sparse Spectral Clustering
Qiong Wang, Junbin Gao, Hong Li
Video Analytics
Video Propagation Networks
Varun Jampani, Raghudeep Gadde, Peter V. Gehler
ActionVLAD: Learning Spatio-Temporal Aggregation for Action Classification
Rohit Girdhar, Deva Ramanan, Abhinav Gupta, Josef Sivic, Bryan Russell
SCC: Semantic Context Cascade for Efficient Action Detection
Fabian Caba Heilbron, Wayner Barrios, Victor Escorcia, Bernard Ghanem
Hierarchical Boundary-Aware Neural Encoder for Video Captioning
Lorenzo Baraldi, Costantino Grana, Rita Cucchiara
HOPE: Hierarchical Object Prototype Encoding for Efficient Object Instance Search in Videos
Tan Yu, Yuwei Wu, Junsong Yuan
Spatio-Temporal Vector of Locally Max Pooled Features for Action Recognition in Videos
Ionut Cosmin Duta, Bogdan Ionescu, Kiyoharu Aizawa, Nicu Sebe
Temporal Action Localization by Structured Maximal Sums
Zehuan Yuan, Jonathan C. Stroud, Tong Lu, Jia Deng
Predicting Salient Face in Multiple-Face Videos
Yufan Liu, Songyang Zhang, Mai Xu, Xuming He
Object Recognition & Scene Understanding 1
Spotlight 2-2A
Graph-Structured Representations for Visual Question Answering
Damien Teney, Lingqiao Liu, Anton van den Hengel
Knowing When to Look: Adaptive Attention via a Visual Sentinel for Image Captioning
Jiasen Lu, Caiming Xiong, Devi Parikh, Richard Socher
Learned Contextual Feature Reweighting for Image Geo-Localization
Hyo Jin Kim, Enrique Dunn, Jan-Michael Frahm
End-To-End Concept Word Detection for Video Captioning, Retrieval, and Question Answering
Youngjae Yu, Hyungjin Ko, Jongwook Choi, Gunhee Kim
Deep Cross-Modal Hashing
Qing-Yuan Jiang, Wu-Jun Li
Unambiguous Text Localization and Retrieval for Cluttered Scenes
Xuejian Rong, Chucai Yi, Yingli Tian
Bayesian Supervised Hashing
Zihao Hu, Junxuan Chen, Hongtao Lu, Tongzhen Zhang
Speed/Accuracy Trade-Offs for Modern Convolutional Object Detectors
Jonathan Huang, Vivek Rathod, Chen Sun, Menglong Zhu, Anoop Korattikara, Alireza Fathi, Ian Fischer, Zbigniew Wojna, Yang Song, Sergio Guadarrama, Kevin Murphy
Oral 2-2A
Detecting Visual Relationships With Deep Relational Networks
Bo Dai, Yuqi Zhang, Dahua Lin
Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes
Tobias Pohlen, Alexander Hermans, Markus Mathias, Bastian Leibe
Network Dissection: Quantifying Interpretability of Deep Visual Representations
David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, Antonio Torralba
AGA: Attribute-Guided Augmentation
Mandar Dixit, Roland Kwitt, Marc Niethammer, Nuno Vasconcelos
Analyzing Humans 2
Spotlight 2-2B
A Hierarchical Approach for Generating Descriptive Image Paragraphs
Jonathan Krause, Justin Johnson, Ranjay Krishna, Li Fei-Fei
Person Re-Identification in the Wild
Liang Zheng, Hengheng Zhang, Shaoyan Sun, Manmohan Chandraker, Yi Yang, Qi Tian
Scalable Person Re-Identification on Supervised Smoothed Manifold
Song Bai, Xiang Bai, Qi Tian
Binge Watching: Scaling Affordance Learning From Sitcoms
Xiaolong Wang, Rohit Girdhar, Abhinav Gupta
Joint Detection and Identification Feature Learning for Person Search
Tong Xiao, Shuang Li, Bochao Wang, Liang Lin, Xiaogang Wang
Synthesizing Normalized Faces From Facial Identity Features
Forrester Cole, David Belanger, Dilip Krishnan, Aaron Sarna, Inbar Mosseri, William T. Freeman
Consistent-Aware Deep Learning for Person Re-Identification in a Camera Network
Ji Lin, Liangliang Ren, Jiwen Lu, Jianjiang Feng, Jie Zhou
Level Playing Field for Million Scale Face Recognition
Aaron Nech, Ira Kemelmacher-Shlizerman
Oral 2-2B
Re-Sign: Re-Aligned End-To-End Sequence Modelling With Deep Recurrent CNN-HMMs
Oscar Koller, Sepehr Zargaran, Hermann Ney
Social Scene Understanding: End-To-End Multi-Person Action Localization and Collective Activity Recognition
Timur Bagautdinov, Alexandre Alahi, FranÃ§ois Fleuret, Pascal Fua, Silvio Savarese
Detangling People: Individuating Multiple Close People and Their Body Parts via Region Assembly
Hao Jiang, Kristen Grauman
Lip Reading Sentences in the Wild
Joon Son Chung, Andrew Senior, Oriol Vinyals, Andrew Zisserman
Applications
Spotlight 2-2C
Deep Matching Prior Network: Toward Tighter Multi-Oriented Text Detection
Yuliang Liu, Lianwen Jin
ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases
Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald M. Summers
Attentional Push: A Deep Convolutional Network for Augmenting Image Salience With Shared Attention Modeling in Social Scenes
Siavash Gorji, James J. Clark
Detecting Oriented Text in Natural Images by Linking Segments
Baoguang Shi, Xiang Bai, Serge Belongie
Learning Video Object Segmentation From Static Images
Federico Perazzi, Anna Khoreva, Rodrigo Benenson, Bernt Schiele, Alexander Sorkine-Hornung
Seeing Invisible Poses: Estimating 3D Body Pose From Egocentric Video
Hao Jiang, Kristen Grauman
Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space
Anh Nguyen, Jeff Clune, Yoshua Bengio, Alexey Dosovitskiy, Jason Yosinski
A Joint Speaker-Listener-Reinforcer Model for Referring Expressions
Licheng Yu, Hao Tan, Mohit Bansal, Tamara L. Berg
Oral 2-2C
End-To-End Learning of Driving Models From Large-Scale Video Datasets
Huazhe Xu, Yang Gao, Fisher Yu, Trevor Darrell
Deep Future Gaze: Gaze Anticipation on Egocentric Videos Using Adversarial Networks
Mengmi Zhang, Keng Teck Ma, Joo Hwee Lim, Qi Zhao, Jiashi Feng
MDNet: A Semantically and Visually Interpretable Medical Image Diagnosis Network
Zizhao Zhang, Yuanpu Xie, Fuyong Xing, Mason McGough, Lin Yang
Poster 2-2
3D Computer Vision
Surface Motion Capture Transfer With Gaussian Process Regression
Adnane Boukhayma, Jean-Sébastien Franco, Edmond Boyer
Visual-Inertial-Semantic Scene Representation for 3D Object Detection
Jingming Dong, Xiaohan Fei, Stefano Soatto
Template-Based Monocular 3D Recovery of Elastic Shapes Using Lagrangian Multipliers
Nazim Haouchine, Stephane Cotin
Learning Category-Specific 3D Shape Models From Weakly Labeled 2D Images
Dingwen Zhang, Junwei Han, Yang Yang, Dong Huang
Simultaneous Geometric and Radiometric Calibration of a Projector-Camera Pair
Marjan Shahpaski, Luis Ricardo Sapaico, Gaspard Chevassus, Sabine SÃ¼sstrunk
Learning Barycentric Representations of 3D Shapes for Sketch-Based 3D Shape Retrieval
Jin Xie, Guoxian Dai, Fan Zhu, Yi Fang
Geodesic Distance Descriptors
Gil Shamai, Ron Kimmel
Analyzing Humans in Images
Modeling Temporal Dynamics and Spatial Configurations of Actions Using Two-Stream Recurrent Neural Networks
Hongsong Wang, Liang Wang
Forecasting Human Dynamics From Static Images
Yu-Wei Chao, Jimei Yang, Brian Price, Scott Cohen, Jia Deng
Re-Ranking Person Re-Identification With k-Reciprocal Encoding
Zhun Zhong, Liang Zheng, Donglin Cao, Shaozi Li
Deep Sequential Context Networks for Action Prediction
Yu Kong, Zhiqiang Tao, Yun Fu
Global Context-Aware Attention LSTM Networks for 3D Action Recognition
Jun Liu, Gang Wang, Ping Hu, Ling-Yu Duan, Alex C. Kot
Dynamic Attention-Controlled Cascaded Shape Regression Exploiting Training Data Augmentation and Fuzzy-Set Sample Weighting
Zhen-Hua Feng, Josef Kittler, William Christmas, Patrik Huber, Xiao-Jun Wu
A Deep Regression Architecture With Two-Stage Re-Initialization for High Performance Facial Landmark Detection
Jiangjing Lv, Xiaohu Shao, Junliang Xing, Cheng Cheng, Xi Zhou
Multiple People Tracking by Lifted Multicut and Person Re-Identification
Siyu Tang, Mykhaylo Andriluka, Bjoern Andres, Bernt Schiele
Towards Accurate Multi-Person Pose Estimation in the Wild
George Papandreou, Tyler Zhu, Nori Kanazawa, Alexander Toshev, Jonathan Tompson, Chris Bregler, Kevin Murphy
Applications
Towards a Quality Metric for Dense Light Fields
Vamsi Kiran Adhikarla, Marek Vinkler, Denis Sumin, RafaÅ‚ K. Mantiuk, Karol Myszkowski, Hans-Peter Seidel, Piotr Didyk
Controlling Perceptual Factors in Neural Style Transfer
Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, Aaron Hertzmann, Eli Shechtman
Biomedical Image/Video Analysis
Joint Sequence Learning and Cross-Modality Convolution for 3D Biomedical Segmentation
Kuan-Lun Tseng, Yen-Liang Lin, Winston Hsu, Chung-Yang Huang
LSTM Self-Supervision for Detailed Behavior Analysis
Biagio Brattoli, Uta BÃ¼chler, Anna-Sophia Wahl, Martin E. Schwab, BjÃ¶rn Ommer
Computational Photography
A Wide-Field-Of-View Monocentric Light Field Camera
Donald G. Dansereau, Glenn Schuster, Joseph Ford, Gordon Wetzstein
Image Motion & Tracking
S2F: Slow-To-Fast Interpolator Flow
Yanchao Yang, Stefano Soatto
CLKN: Cascaded Lucas-Kanade Networks for Image Alignment
Che-Han Chang, Chun-Nan Chou, Edward Y. Chang
Multi-Object Tracking With Quadruplet Convolutional Neural Networks
Mooyeol Baek, Jeany Son, Minsu Cho, Bohyung Han
Low- & Mid-Level Vision
Learning to Detect Salient Objects With Image-Level Supervision
Lijun Wang, Huchuan Lu, Yifan Wang, Mengyang Feng, Dong Wang, Baocai Yin, Xiang Ruan
From Motion Blur to Motion Flow: A Deep Learning Solution for Removing Heterogeneous Motion Blur
Dong Gong, Jie Yang, Lingqiao Liu, Yanning Zhang, Ian Reid, Chunhua Shen, Anton van den Hengel, Qinfeng Shi
Co-Occurrence Filter
Roy J. Jevnisek, Shai Avidan
Fractal Dimension Invariant Filtering and Its CNN-Based Implementation
Hongteng Xu, Junchi Yan, Nils Persson, Weiyao Lin, Hongyuan Zha
Noise-Blind Image Deblurring
Meiguang Jin, Stefan Roth, Paolo Favaro
Simultaneous Visual Data Completion and Denoising Based on Tensor Rank and Total Variation Minimization and Its Primal-Dual Splitting Algorithm
Tatsuya Yokota, Hidekata Hontani
HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors
Vassileios Balntas, Karel Lenc, Andrea Vedaldi, Krystian Mikolajczyk
Hyperspectral Image Super-Resolution via Non-Local Sparse Tensor Factorization
Renwei Dian, Leyuan Fang, Shutao Li
Reflection Removal Using Low-Rank Matrix Completion
Byeong-Ju Han, Jae-Young Sim
Object Co-Skeletonization With Co-Segmentation
Koteswar Rao Jerripothula, Jianfei Cai, Jiangbo Lu, Junsong Yuan
Machine Learning
Mining Object Parts From CNNs via Active Question-Answering
Quanshi Zhang, Ruiming Cao, Ying Nian Wu, Song-Chun Zhu
PolyNet: A Pursuit of Structural Diversity in Very Deep Networks
Xingcheng Zhang, Zhizhong Li, Chen Change Loy, Dahua Lin
The VQA-Machine: Learning How to Use Existing Vision Algorithms to Answer New Questions
Peng Wang, Qi Wu, Chunhua Shen, Anton van den Hengel
Joint Discriminative Bayesian Dictionary and Classifier Learning
Naveed Akhtar, Ajmal Mian, Fatih Porikli
A Study of Lagrangean Decompositions and Dual Ascent Solvers for Graph Matching
Paul Swoboda, Carsten Rother, Hassan Abu Alhaija, Dagmar KainmÃ¼ller, Bogdan Savchynskyy
Quad-Networks: Unsupervised Learning to Rank for Interest Point Detection
Nikolay Savinov, Akihito Seki, Ä½ubor LadickÃ½, Torsten Sattler, Marc Pollefeys
Outlier-Robust Tensor PCA
Pan Zhou, Jiashi Feng
Learning Adaptive Receptive Fields for Deep Image Parsing Network
Zhen Wei, Yao Sun, Jinqiao Wang, Hanjiang Lai, Si Liu
Learning an Invariant Hilbert Space for Domain Adaptation
Samitha Herath, Mehrtash Harandi, Fatih Porikli
Fixed-Point Factorized Networks
Peisong Wang, Jian Cheng
Discriminative Optimization: Theory and Applications to Point Cloud Registration
Jayakorn Vongkulbhisal, Fernando De la Torre, JoÃ£o P. Costeira
Online Asymmetric Similarity Learning for Cross-Modal Retrieval
Yiling Wu, Shuhui Wang, Qingming Huang
Improving Training of Deep Neural Networks via Singular Value Bounding
Kui Jia, Dacheng Tao, Shenghua Gao, Xiangmin Xu
S3Pool: Pooling With Stochastic Spatial Sampling
Shuangfei Zhai, Hui Wu, Abhishek Kumar, Yu Cheng, Yongxi Lu, Zhongfei Zhang, Rogerio Feris
Sports Field Localization via Deep Structured Models
Namdar Homayounfar, Sanja Fidler, Raquel Urtasun
Noisy Softmax: Improving the Generalization Ability of DCNN via Postponing the Early Softmax Saturation
Binghui Chen, Weihong Deng, Junping Du
Switching Convolutional Neural Network for Crowd Counting
Deepak Babu Sam, Shiv Surya, R. Venkatesh Babu
Network Sketching: Exploiting Binary Structure in Deep CNNs
Yiwen Guo, Anbang Yao, Hao Zhao, Yurong Chen
Multi-Task Clustering of Human Actions by Sharing Information
Xiaoqiang Yan, Shizhe Hu, Yangdong Ye
Soft-Margin Mixture of Regressions
Dong Huang, Longfei Han, Fernando De la Torre
Multigrid Neural Architectures
Tsung-Wei Ke, Michael Maire, Stella X. Yu
High-Resolution Image Inpainting Using Multi-Scale Neural Patch Synthesis
Chao Yang, Xin Lu, Zhe Lin, Eli Shechtman, Oliver Wang, Hao Li
Deep Quantization: Encoding Convolutional Activations With Deep Generative Model
Zhaofan Qiu, Ting Yao, Tao Mei
DOPE: Distributed Optimization for Pairwise Energies
Jose Dolz, Ismail Ben Ayed, Christian Desrosiers
Improved Texture Networks: Maximizing Quality and Diversity in Feed-Forward Stylization and Texture Synthesis
Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky
Object Recognition & Scene Understanding
Polyhedral Conic Classifiers for Visual Object Detection and Classification
Hakan Cevikalp, Bill Triggs
Incremental Kernel Null Space Discriminant Analysis for Novelty Detection
Juncheng Liu, Zhouhui Lian, Yi Wang, Jianguo Xiao
Predicting Ground-Level Scene Layout From Aerial Imagery
Menghua Zhai, Zachary Bessinger, Scott Workman, Nathan Jacobs
Deep Feature Flow for Video Recognition
Xizhou Zhu, Yuwen Xiong, Jifeng Dai, Lu Yuan, Yichen Wei
Object-Aware Dense Semantic Correspondence
Fan Yang, Xin Li, Hong Cheng, Jianping Li, Leiting Chen
Semantic Regularisation for Recurrent Image Annotation
Feng Liu, Tao Xiang, Timothy M. Hospedales, Wankou Yang, Changyin Sun
Video2Shop: Exact Matching Clothes in Videos to Online Shopping Images
Zhi-Qi Cheng, Xiao Wu, Yang Liu, Xian-Sheng Hua
Fast-At: Fast Automatic Thumbnail Generation Using Deep Neural Networks
Seyed A. Esmaeili, Bharat Singh, Larry S. Davis
Multi-Level Attention Networks for Visual Question Answering
Dongfei Yu, Jianlong Fu, Tao Mei, Yong Rui
Generating Descriptions With Grounded and Co-Referenced People
Anna Rohrbach, Marcus Rohrbach, Siyu Tang, Seong Joon Oh, Bernt Schiele
Straight to Shapes: Real-Time Detection of Encoded Shapes
Saumya Jetley, Michael Sapienza, Stuart Golodetz, Philip H. S. Torr
Simultaneous Feature Aggregating and Hashing for Large-Scale Image Search
Thanh-Toan Do, Dang-Khoa Le Tan, Trung T. Pham, Ngai-Man Cheung
Improving Facial Attribute Prediction Using Semantic Segmentation
Mahdi M. Kalayeh, Boqing Gong, Mubarak Shah
Video Analytics
Learning Cross-Modal Deep Representations for Robust Pedestrian Detection
Dan Xu, Wanli Ouyang, Elisa Ricci, Xiaogang Wang, Nicu Sebe
Spatio-Temporal Self-Organizing Map Deep Network for Dynamic Object Detection From Videos
Yang Du, Chunfeng Yuan, Bing Li, Weiming Hu, Stephen Maybank
CERN: Confidence-Energy Recurrent Network for Group Activity Recognition
Tianmin Shu, Sinisa Todorovic, Song-Chun Zhu
Understanding Traffic Density From Large-Scale Web Camera Data
Shanghang Zhang, Guanhang Wu, JoÃ£o P. Costeira, José M. F. Moura
Collaborative Summarization of Topic-Related Videos
Rameswar Panda, Amit K. Roy-Chowdhury
Machine Learning 3
Spotlight 3-1A
Local Binary Convolutional Neural Networks
Felix Juefei-Xu, Vishnu Naresh Boddeti, Marios Savvides
Deep Self-Taught Learning for Weakly Supervised Object Localization
Zequn Jie, Yunchao Wei, Xiaojie Jin, Jiashi Feng, Wei Liu
Multi-Modal Mean-Fields via Cardinality-Based Clamping
Pierre Baqué, FranÃ§ois Fleuret, Pascal Fua
Probabilistic Temporal Subspace Clustering
Behnam Gholami, Vladimir Pavlovic
Provable Self-Representation Based Outlier Detection in a Union of Subspaces
Chong You, Daniel P. Robinson, René Vidal
Latent Multi-View Subspace Clustering
Changqing Zhang, Qinghua Hu, Huazhu Fu, Pengfei Zhu, Xiaochun Cao
Learning to Extract Semantic Structure From Documents Using Multimodal Fully Convolutional Neural Networks
Xiao Yang, Ersin Yumer, Paul Asente, Mike Kraley, Daniel Kifer, C. Lee Giles
Age Progression/Regression by Conditional Adversarial Autoencoder
Zhifei Zhang, Yang Song, Hairong Qi
Oral 3-1A
Compact Matrix Factorization With Dependent Subspaces
Viktor Larsson, Carl Olsson
FFTLasso: Large-Scale LASSO in the Fourier Domain
Adel Bibi, Hani Itani, Bernard Ghanem
On the Global Geometry of Sphere-Constrained Sparse Blind Deconvolution
Yuqian Zhang, Yenson Lau, Han-wen Kuo, Sky Cheung, Abhay Pasupathy, John Wright
Global Optimality in Neural Network Training
Benjamin D. Haeffele, René Vidal
Object Recognition & Scene Understanding 2
Spotlight 3-1B
What Is and What Is Not a Salient Object? Learning Salient Object Detector by Ensembling Linear Exemplar Regressors
Changqun Xia, Jia Li, Xiaowu Chen, Anlin Zheng, Yu Zhang
Deep Variation-Structured Reinforcement Learning for Visual Relationship and Attribute Detection
Xiaodan Liang, Lisa Lee, Eric P. Xing
Modeling Relationships in Referential Expressions With Compositional Modular Networks
Ronghang Hu, Marcus Rohrbach, Jacob Andreas, Trevor Darrell, Kate Saenko
Counting Everyday Objects in Everyday Scenes
Prithvijit Chattopadhyay, Ramakrishna Vedantam, Ramprasaath R. Selvaraju, Dhruv Batra, Devi Parikh
Fully Convolutional Instance-Aware Semantic Segmentation
Yi Li, Haozhi Qi, Jifeng Dai, Xiangyang Ji, Yichen Wei
Semantic Autoencoder for Zero-Shot Learning
Elyor Kodirov, Tao Xiang, Shaogang Gong
CityPersons: A Diverse Dataset for Pedestrian Detection
Shanshan Zhang, Rodrigo Benenson, Bernt Schiele
GuessWhat?! Visual Object Discovery Through Multi-Modal Dialogue
Harm de Vries, Florian Strub, Sarath Chandar, Olivier Pietquin, Hugo Larochelle, Aaron Courville
Oral 3-1B
Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-Grained Image Recognition
Jianlong Fu, Heliang Zheng, Tao Mei
Annotating Object Instances With a Polygon-RNN
LluÃ­s CastrejÃ³n, Kaustav Kundu, Raquel Urtasun, Sanja Fidler
Connecting Look and Feel: Associating the Visual and Tactile Properties of Physical Materials
Wenzhen Yuan, Shaoxiong Wang, Siyuan Dong, Edward Adelson
Deep Learning Human Mind for Automated Visual Classification
Concetto Spampinato, Simone Palazzo, Isaak Kavasidis, Daniela Giordano, Nasim Souly, Mubarak Shah
Poster 3-1
3D Computer Vision
Self-Calibration-Based Approach to Critical Motion Sequences of Rolling-Shutter Structure From Motion
Eisuke Ito, Takayuki Okatani
Semi-Calibrated Near Field Photometric Stereo
Fotios Logothetis, Roberto Mecca, Roberto Cipolla
Semantic Multi-View Stereo: Jointly Estimating Objects and Voxels
Ali Osman Ulusoy, Michael J. Black, Andreas Geiger
Learning to Predict Stereo Reliability Enforcing Local Consistency of Confidence Maps
Matteo Poggi, Stefano Mattoccia
The Misty Three Point Algorithm for Relative Pose
Tobias Palmér, Kalle Ã…strÃ¶m, Jan-Michael Frahm
The Surfacing of Multiview 3D Drawings via Lofting and Occlusion Reasoning
Anil Usumezbas, Ricardo Fabbri, Benjamin B. Kimia
A New Representation of Skeleton Sequences for 3D Action Recognition
Qiuhong Ke, Mohammed Bennamoun, Senjian An, Ferdous Sohel, Farid Boussaid
A General Framework for Curve and Surface Comparison and Registration With Oriented Varifolds
Irène Kaltenmark, Benjamin Charlier, Nicolas Charon
Learning to Align Semantic Segmentation and 2.5D Maps for Geolocalization
Anil Armagan, Martin Hirzer, Peter M. Roth, Vincent Lepetit
A Generative Model for Depth-Based Robust 3D Facial Pose Tracking
Lu Sheng, Jianfei Cai, Tat-Jen Cham, Vladimir Pavlovic, King Ngi Ngan
Fast 3D Reconstruction of Faces With Glasses
Fabio Maninchedda, Martin R. Oswald, Marc Pollefeys
An Efficient Algebraic Solution to the Perspective-Three-Point Problem
Tong Ke, Stergios I. Roumeliotis
Analyzing Humans in Images
Learning From Synthetic Humans
GÃ¼l Varol, Javier Romero, Xavier Martin, Naureen Mahmood, Michael J. Black, Ivan Laptev, Cordelia Schmid
Forecasting Interactive Dynamics of Pedestrians With Fictitious Play
Wei-Chiu Ma, De-An Huang, Namhoon Lee, Kris M. Kitani
Hand Keypoint Detection in Single Images Using Multiview Bootstrapping
Tomas Simon, Hanbyul Joo, Iain Matthews, Yaser Sheikh
PoseTrack: Joint Multi-Person Pose Estimation and Tracking
Umar Iqbal, Anton Milan, Juergen Gall
Expecting the Unexpected: Training Detectors for Unusual Pedestrians With Adversarial Imposters
Shiyu Huang, Deva Ramanan
On Human Motion Prediction Using Recurrent Neural Networks
Julieta Martinez, Michael J. Black, Javier Romero
Learning and Refining of Privileged Information-Based RNNs for Action Recognition From Depth Sequences
Zhiyuan Shi, Tae-Kyun Kim
Quality Aware Network for Set to Set Recognition
Yu Liu, Junjie Yan, Wanli Ouyang
Unite the People: Closing the Loop Between 3D and 2D Human Representations
Christoph Lassner, Javier Romero, Martin Kiefel, Federica Bogo, Michael J. Black, Peter V. Gehler
Deep Multitask Architecture for Integrated 2D and 3D Human Sensing
Alin-Ionut Popa, Mihai Zanfir, Cristian Sminchisescu
Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
JoÃ£o Carreira, Andrew Zisserman
Applications
Identifying First-Person Camera Wearers in Third-Person Videos
Chenyou Fan, Jangwon Lee, Mingze Xu, Krishna Kumar Singh, Yong Jae Lee, David J. Crandall, Michael S. Ryoo
Biomedical Image/Video Analysis
Parsing Images of Overlapping Organisms With Deep Singling-Out Networks
Victor Yurchenko, Victor Lempitsky
Fine-Tuning Convolutional Neural Networks for Biomedical Image Analysis: Actively and Incrementally
Zongwei Zhou, Jae Shin, Lei Zhang, Suryakanth Gurudu, Michael Gotway, Jianming Liang
Computational Photography
Depth From Defocus in the Wild
Huixuan Tang, Scott Cohen, Brian Price, Stephen Schiller, Kiriakos N. Kutulakos
Matting and Depth Recovery of Thin Structures Using a Focal Stack
Chao Liu, Srinivasa G. Narasimhan, Artur W. Dubrawski
Image Motion & Tracking
Robust Interpolation of Correspondences for Large Displacement Optical Flow
Yinlin Hu, Yunsong Li, Rui Song
Large Margin Object Tracking With Circulant Feature Maps
Mengmeng Wang, Yong Liu, Zeyi Huang
Minimum Delay Moving Object Detection
Dong Lao, Ganesh Sundaramoorthi
Multi-Task Correlation Particle Filter for Robust Object Tracking
Tianzhu Zhang, Changsheng Xu, Ming-Hsuan Yang
Attentional Correlation Filter Network for Adaptive Visual Tracking
Jongwon Choi, Hyung Jin Chang, Sangdoo Yun, Tobias Fischer, Yiannis Demiris, Jin Young Choi
The World of Fast Moving Objects
Denys Rozumnyi, Jan Kotera, Filip Å roubek, LukÃ¡Å¡ NovotnÃ½, JiÅ™Ã­ Matas
Discriminative Correlation Filter With Channel and Spatial Reliability
Alan LukeÅ¾iÄ, TomÃ¡Å¡ VojÃ­Å™, Luka ÄŒehovin Zajc, JiÅ™Ã­ Matas, Matej Kristan
Low- & Mid-Level Vision
Learning Deep Binary Descriptor With Multi-Quantization
Yueqi Duan, Jiwen Lu, Ziwei Wang, Jianjiang Feng, Jie Zhou
One-To-Many Network for Visually Pleasing Compression Artifacts Reduction
Jun Guo, Hongyang Chao
Gated Feedback Refinement Network for Dense Image Labeling
Md Amirul Islam, Mrigank Rochan, Neil D. B. Bruce, Yang Wang
BRISKS: Binary Features for Spherical Images on a Geodesic Grid
Hao Guan, William A. P. Smith
Superpixels and Polygons Using Simple Non-Iterative Clustering
Radhakrishna Achanta, Sabine SÃ¼sstrunk
Hardware-Efficient Guided Image Filtering for Multi-Label Problem
Longquan Dai, Mengke Yuan, Zechao Li, Xiaopeng Zhang, Jinhui Tang
Alternating Direction Graph Matching
D. KhuÃª LÃª-Huu, Nikos Paragios
Learning Discriminative and Transformation Covariant Local Feature Detectors
Xu Zhang, Felix X. Yu, Svebor Karaman, Shih-Fu Chang
Machine Learning
Correlational Gaussian Processes for Cross-Domain Visual Recognition
Chengjiang Long, Gang Hua
DeLiGAN : Generative Adversarial Networks for Diverse and Limited Data
Swaminathan Gurumurthy, Ravi Kiran Sarvadevabhatla, R. Venkatesh Babu
Oriented Response Networks
Yanzhao Zhou, Qixiang Ye, Qiang Qiu, Jianbin Jiao
Missing Modalities Imputation via Cascaded Residual Autoencoder
Luan Tran, Xiaoming Liu, Jiayu Zhou, Rong Jin
Efficient Optimization for Hierarchically-structured Interacting Segments Abstract[PDF](HINTS)
Hossam Isack, Olga Veksler, Ipek Oguz, Milan Sonka, Yuri Boykov
A Message Passing Algorithm for the Minimum Cost Multicut Problem
Paul Swoboda, Bjoern Andres
End-To-End Representation Learning for Correlation Filter Based Tracking
Jack Valmadre, Luca Bertinetto, JoÃ£o Henriques, Andrea Vedaldi, Philip H. S. Torr
Filter Flow Made Practical: Massively Parallel and Lock-Free
Sathya N. Ravi, Yunyang Xiong, Lopamudra Mukherjee, Vikas Singh
Online Graph Completion: Multivariate Signal Recovery in Computer Vision
Won Hwa Kim, Mona Jalal, Seongjae Hwang, Sterling C. Johnson, Vikas Singh
Point to Set Similarity Based Deep Feature Learning for Person Re-Identification
Sanping Zhou, Jinjun Wang, Jiayun Wang, Yihong Gong, Nanning Zheng
Exploiting Saliency for Object Segmentation From Image Level Labels
Seong Joon Oh, Rodrigo Benenson, Anna Khoreva, Zeynep Akata, Mario Fritz, Bernt Schiele
Consensus Maximization With Linear Matrix Inequality Constraints
Pablo Speciale, Danda Pani Paudel, Martin R. Oswald, Till Kroeger, Luc Van Gool, Marc Pollefeys
Physically-Based Rendering for Indoor Scene Understanding Using Convolutional Neural Networks
Yinda Zhang, Shuran Song, Ersin Yumer, Manolis Savva, Joon-Young Lee, Hailin Jin, Thomas Funkhouser
Deep Multimodal Representation Learning From Temporal Data
Xitong Yang, Palghat Ramesh, Radha Chitta, Sriganesh Madhvanath, Edgar A. Bernal, Jiebo Luo
All You Need Is Beyond a Good Init: Exploring Better Solution for Training Extremely Deep Convolutional Neural Networks With Orthonormality and Modulation
Di Xie, Jiang Xiong, Shiliang Pu
Hard Mixtures of Experts for Large Scale Weakly Supervised Vision
Sam Gross, Marc'Aurelio Ranzato, Arthur Szlam
A Reinforcement Learning Approach to the View Planning Problem
Mustafa Devrim Kaba, Mustafa Gokhan Uzunbas, Ser Nam Lim
Zero-Shot Classification With Discriminative Semantic Representation Learning
Meng Ye, Yuhong Guo
Adversarial Discriminative Domain Adaptation
Eric Tzeng, Judy Hoffman, Kate Saenko, Trevor Darrell
None of the above
Learning to Rank Retargeted Images
Yang Chen, Yong-Jin Liu, Yu-Kun Lai
Object Recognition & Scene Understanding
Automatic Discovery, Association Estimation and Learning of Semantic Attributes for a Thousand Categories
Ziad Al-Halah, Rainer Stiefelhagen
Scene Parsing Through ADE20K Dataset
Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, Antonio Torralba
Weakly Supervised Cascaded Convolutional Networks
Ali Diba, Vivek Sharma, Ali Pazandeh, Hamed Pirsiavash, Luc Van Gool
Discretely Coding Semantic Rank Orders for Supervised Image Hashing
Li Liu, Ling Shao, Fumin Shen, Mengyang Yu
Joint Geometrical and Statistical Alignment for Visual Domain Adaptation
Jing Zhang, Wanqing Li, Philip Ogunbona
Weakly Supervised Dense Video Captioning
Zhiqiang Shen, Jianguo Li, Zhou Su, Minjun Li, Yurong Chen, Yu-Gang Jiang, Xiangyang Xue
RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
Guosheng Lin, Anton Milan, Chunhua Shen, Ian Reid
Semantic Segmentation via Structured Patch Prediction, Context CRF and Guidance CRF
Falong Shen, Rui Gan, Shuicheng Yan, Gang Zeng
Person Search With Natural Language Description
Shuang Li, Tong Xiao, Hongsheng Li, Bolei Zhou, Dayu Yue, Xiaogang Wang
Weakly Supervised Affordance Detection
Johann Sawatzky, Abhilash Srikantha, Juergen Gall
Zero-Shot Recognition Using Dual Visual-Semantic Mapping Paths
Yanan Li, Donghui Wang, Huanhang Hu, Yuetan Lin, Yueting Zhuang
Neural Aggregation Network for Video Face Recognition
Jiaolong Yang, Peiran Ren, Dongqing Zhang, Dong Chen, Fang Wen, Hongdong Li, Gang Hua
Relationship Proposal Networks
Ji Zhang, Mohamed Elhoseiny, Scott Cohen, Walter Chang, Ahmed Elgammal
Learning Object Interactions and Descriptions for Semantic Image Segmentation
Guangrun Wang, Ping Luo, Liang Lin, Xiaogang Wang
RON: Reverse Connection With Objectness Prior Networks for Object Detection
Tao Kong, Fuchun Sun, Anbang Yao, Huaping Liu, Ming Lu, Yurong Chen
Weakly-Supervised Visual Grounding of Phrases With Linguistic Structures
Fanyi Xiao, Leonid Sigal, Yong Jae Lee
Incorporating Copying Mechanism in Image Captioning for Learning Novel Objects
Ting Yao, Yingwei Pan, Yehao Li, Tao Mei
Beyond Instance-Level Image Retrieval: Leveraging Captions to Learn a Global Visual Representation for Semantic Retrieval
Albert Gordo, Diane Larlus
MuCaLe-Net: Multi Categorical-Level Networks to Generate More Discriminating Features
Youssef Tamaazousti, Hervé Le Borgne, Céline Hudelot
Zero Shot Learning via Multi-Scale Manifold Regularization
Shay Deutsch, Soheil Kolouri, Kyungnam Kim, Yuri Owechko, Stefano Soatto
Theory
Deeply Supervised Salient Object Detection With Short Connections
Qibin Hou, Ming-Ming Cheng, Xiaowei Hu, Ali Borji, Zhuowen Tu, Philip H. S. Torr
A Matrix Splitting Method for Composite Function Minimization
Ganzhao Yuan, Wei-Shi Zheng, Bernard Ghanem
Video Analytics
One-Shot Video Object Segmentation Abstract[PDF](PDF, project, code, code)
Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-TaixÃ©, Daniel Cremers, Luc Van Gool
Fast Person Re-Identification via Cross-Camera Semantic Binary Transformation
Jiaxin Chen, Yunhong Wang, Jie Qin, Li Liu, Ling Shao
SPFTN: A Self-Paced Fine-Tuning Network for Segmenting Objects in Weakly Labelled Videos
Dingwen Zhang, Le Yang, Deyu Meng, Dong Xu, Junwei Han
Machine Learning 4
Spotlight 4-1A
Hidden Layers in Perceptual Learning
Gad Cohen, Daphna Weinshall
Few-Shot Object Recognition From Machine-Labeled Web Images
Zhongwen Xu, Linchao Zhu, Yi Yang
Hallucinating Very Low-Resolution Unaligned and Noisy Face Images by Transformative Discriminative Autoencoders
Xin Yu, Fatih Porikli
Are You Smarter Than a Sixth Grader? Textbook Question Answering for Multimodal Machine Comprehension
Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, Hannaneh Hajishirzi
Deep Hashing Network for Unsupervised Domain Adaptation
Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, Sethuraman Panchanathan
Generalized Deep Image to Image Regression
Venkataraman Santhanam, Vlad I. Morariu, Larry S. Davis
Deep Learning With Low Precision by Half-Wave Gaussian Quantization
Zhaowei Cai, Xiaodong He, Jian Sun, Nuno Vasconcelos
Creativity: Generating Diverse Questions Using Variational Autoencoders
Unnat Jain, Ziyu Zhang, Alexander G. Schwing
Oral 4-1A
Geometric Deep Learning on Graphs and Manifolds Using Mixture Model CNNs
Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele RodolÃ , Jan Svoboda, Michael M. Bronstein
Full Resolution Image Compression With Recurrent Neural Networks
George Toderici, Damien Vincent, Nick Johnston, Sung Jin Hwang, David Minnen, Joel Shor, Michele Covell
Neural Face Editing With Intrinsic Image Disentangling
Zhixin Shu, Ersin Yumer, Sunil Hadap, Kalyan Sunkavalli, Eli Shechtman, Dimitris Samaras
Ubernet: Training a Universal Convolutional Neural Network for Low-, Mid-, and High-Level Vision Using Diverse Datasets and Limited Memory
Iasonas Kokkinos
Analyzing Humans with 3D Vision
Spotlight 4-1B
3D Face Morphable Models â€œIn-The-Wildâ€
James Booth, Epameinondas Antonakos, Stylianos Ploumpis, George Trigeorgis, Yannis Panagakis, Stefanos Zafeiriou
KillingFusion: Non-Rigid 3D Reconstruction Without Correspondences
Miroslava Slavcheva, Maximilian Baust, Daniel Cremers, Slobodan Ilic
Detailed, Accurate, Human Shape Estimation From Clothed 3D Scan Sequences
Chao Zhang, Sergi Pujades, Michael J. Black, Gerard Pons-Moll
POSEidon: Face-From-Depth for Driver Pose Estimation
Guido Borghi, Marco Venturelli, Roberto Vezzani, Rita Cucchiara
Human Shape From Silhouettes Using Generative HKS Descriptors and Cross-Modal Neural Networks
Endri Dibra, Himanshu Jain, Cengiz Ã–ztireli, Remo Ziegler, Markus Gross
Parametric T-Spline Face Morphable Model for Detailed Fitting in Shape Subspace
Weilong Peng, Zhiyong Feng, Chao Xu, Yong Su
3D Menagerie: Modeling the 3D Shape and Pose of Animals
Silvia Zuffi, Angjoo Kanazawa, David W. Jacobs, Michael J. Black
iCaRL: Incremental Classifier and Representation Learning
Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
Oral 4-1B
Recurrent 3D Pose Sequence Machines
Mude Lin, Liang Lin, Xiaodan Liang, Keze Wang, Hui Cheng
Learning Detailed Face Reconstruction From a Single Image
Elad Richardson, Matan Sela, Roy Or-El, Ron Kimmel
Thin-Slicing Network: A Deep Structured Model for Pose Estimation in Videos
Jie Song, Limin Wang, Luc Van Gool, Otmar Hilliges
Dynamic FAUST: Registering Human Bodies in Motion
Federica Bogo, Javier Romero, Gerard Pons-Moll, Michael J. Black
Poster 4-1
3D Computer Vision
Semantically Coherent Co-Segmentation and Reconstruction of Dynamic Scenes
Armin Mustafa, Adrian Hilton
On the Two-View Geometry of Unsynchronized Cameras
Cenek Albl, Zuzana Kukelova, Andrew Fitzgibbon, Jan Heller, Matej Smid, Tomas Pajdla
Using Locally Corresponding CAD Models for Dense 3D Reconstructions From a Single Image
Chen Kong, Chen-Hsuan Lin, Simon Lucey
A Clever Elimination Strategy for Efficient Minimal Solvers
Zuzana Kukelova, Joe Kileel, Bernd Sturmfels, Tomas Pajdla
Convex Global 3D Registration With Lagrangian Duality
Jesus Briales, Javier Gonzalez-Jimenez
DeMoN: Depth and Motion Network for Learning Monocular Stereo
Benjamin Ummenhofer, Huizhong Zhou, Jonas Uhrig, Nikolaus Mayer, Eddy Ilg, Alexey Dosovitskiy, Thomas Brox
3D Bounding Box Estimation Using Deep Learning and Geometry
Arsalan Mousavian, Dragomir Anguelov, John Flynn, Jana KoÅ¡eckÃ¡
A Dataset for Benchmarking Image-Based Localization
Xun Sun, Yuanfan Xie, Pei Luo, Liang Wang
Analyzing Humans in Images
Asynchronous Temporal Fields for Action Recognition
Gunnar A. Sigurdsson, Santosh Divvala, Ali Farhadi, Abhinav Gupta
Sequential Person Recognition in Photo Albums With a Recurrent Network
Yao Li, Guosheng Lin, Bohan Zhuang, Lingqiao Liu, Chunhua Shen, Anton van den Hengel
Multi-Context Attention for Human Pose Estimation
Xiao Chu, Wei Yang, Wanli Ouyang, Cheng Ma, Alan L. Yuille, Xiaogang Wang
3D Convolutional Neural Networks for Efficient and Robust Hand Pose Estimation From Single Depth Images
Liuhao Ge, Hui Liang, Junsong Yuan, Daniel Thalmann
Lifting From the Deep: Convolutional 3D Pose Estimation From a Single Image
Denis Tome, Chris Russell, Lourdes Agapito
AdaScan: Adaptive Scan Pooling in Deep Convolutional Neural Networks for Human Action Recognition in Videos
Amlan Kar, Nishant Rai, Karan Sikka, Gaurav Sharma
Deep Structured Learning for Facial Action Unit Intensity Estimation
Robert Walecki, Ognjen Abstract[PDF](Oggi) Rudovic, Vladimir Pavlovic, BjÃ¶ern Schuller, Maja Pantic
Simultaneous Facial Landmark Detection, Pose and Deformation Estimation Under Facial Occlusion
Yue Wu, Chao Gou, Qiang Ji
Self-Supervised Video Representation Learning With Odd-One-Out Networks
Basura Fernando, Hakan Bilen, Efstratios Gavves, Stephen Gould
Robust Joint and Individual Variance Explained
Christos Sagonas, Yannis Panagakis, Alina Leidinger, Stefanos Zafeiriou
Discriminative Covariance Oriented Representation Learning for Face Recognition With Image Sets
Wen Wang, Ruiping Wang, Shiguang Shan, Xilin Chen
3D Human Pose Estimation = 2D Pose Estimation + Matching
Ching-Hang Chen, Deva Ramanan
Applications
Joint Gap Detection and Inpainting of Line Drawings
Kazuma Sasaki, Satoshi Iizuka, Edgar Simo-Serra, Hiroshi Ishikawa
Biomedical Image/Video Analysis
Riemannian Nonlinear Mixed Effects Models: Analyzing Longitudinal Deformations in Neuroimaging
Hyunwoo J. Kim, Nagesh Adluru, Heemanshu Suri, Baba C. Vemuri, Sterling C. Johnson, Vikas Singh
Simultaneous Super-Resolution and Cross-Modality Synthesis of 3D Medical Images Using Weakly-Supervised Joint Convolutional Sparse Coding
Yawen Huang, Ling Shao, Alejandro F. Frangi
Computational Photography
Multiple-Scattering Microphysics Tomography
Aviad Levis, Yoav Y. Schechner, Anthony B. Davis
Image Motion & Tracking
Accurate Optical Flow via Direct Cost Volume Processing
Jia Xu, René Ranftl, Vladlen Koltun
Event-Based Visual Inertial Odometry
Alex Zihao Zhu, Nikolay Atanasov, Kostas Daniilidis
Robust Visual Tracking Using Oblique Random Forests
Le Zhang, Jagannadan Varadarajan, Ponnuthurai Nagaratnam Suganthan, Narendra Ahuja, Pierre Moulin
Low- & Mid-Level Vision
Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution
Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, Ming-Hsuan Yang
Learning Non-Lambertian Object Intrinsics Across ShapeNet Categories
Jian Shi, Yue Dong, Hao Su, Stella X. Yu
MCMLSD: A Dynamic Programming Approach to Line Segment Detection
Emilio J. AlmazÃ n, Ron Tal, Yiming Qian, James H. Elder
Contour-Constrained Superpixels for Image and Video Processing
Se-Ho Lee, Won-Dong Jang, Chang-Su Kim
Richer Convolutional Features for Edge Detection
Yun Liu, Ming-Ming Cheng, Xiaowei Hu, Kai Wang, Xiang Bai
Non-Local Color Image Denoising With Convolutional Neural Networks
Stamatios Lefkimmiatis
Generative Face Completion
Yijun Li, Sifei Liu, Jimei Yang, Ming-Hsuan Yang
Hyper-Laplacian Regularized Unidirectional Low-Rank Tensor Recovery for Multispectral Image Denoising
Yi Chang, Luxin Yan, Sheng Zhong
Unsupervised Semantic Scene Labeling for Streaming Data
Maggie Wigness, John G. Rogers III
Why You Should Forget Luminance Conversion and Do Something Better
Rang M. H. Nguyen, Michael S. Brown
Deep Semantic Feature Matching
Nikolai Ufer, BjÃ¶rn Ommer
Machine Learning
Revisiting the Variable Projection Method for Separable Nonlinear Least Squares Problems
Je Hyeong Hong, Christopher Zach, Andrew Fitzgibbon
Efficient Multiple Instance Metric Learning Using Weakly Supervised Data
Marc T. Law, Yaoliang Yu, Raquel Urtasun, Richard S. Zemel, Eric P. Xing
WILDCAT: Weakly Supervised Learning of Deep ConvNets for Image Classification, Pointwise Localization and Segmentation
Thibaut Durand, Taylor Mordan, Nicolas Thome, Matthieu Cord
Image-To-Image Translation With Conditional Adversarial Networks
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros
Deep Roots: Improving CNN Efficiency With Hierarchical Filter Groups
Yani Ioannou, Duncan Robertson, Roberto Cipolla, Antonio Criminisi
Aggregated Residual Transformations for Deep Neural Networks
Saining Xie, Ross Girshick, Piotr DollÃ¡r, Zhuowen Tu, Kaiming He
MIML-FCN+: Multi-Instance Multi-Label Learning via Fully Convolutional Networks With Privileged Information
Hao Yang, Joey Tianyi Zhou, Jianfei Cai, Yew Soon Ong
Low-Rank Embedded Ensemble Semantic Dictionary for Zero-Shot Learning
Zhengming Ding, Ming Shao, Yun Fu
Factorized Variational Autoencoders for Modeling Audience Reactions to Movies
Zhiwei Deng, Rajitha Navarathna, Peter Carr, Stephan Mandt, Yisong Yue, Iain Matthews, Greg Mori
Learning Features by Watching Objects Move
Deepak Pathak, Ross Girshick, Piotr DollÃ¡r, Trevor Darrell, Bharath Hariharan
What Can Help Pedestrian Detection?
Jiayuan Mao, Tete Xiao, Yuning Jiang, Zhimin Cao
DeepPermNet: Visual Permutation Learning
Rodrigo Santa Cruz, Basura Fernando, Anoop Cherian, Stephen Gould
Learning the Multilinear Structure of Visual Data
Mengjiao Wang, Yannis Panagakis, Patrick Snape, Stefanos Zafeiriou
Adaptive and Move Making Auxiliary Cuts for Binary Pairwise Energies
Lena Gorelick, Yuri Boykov, Olga Veksler
Designing Energy-Efficient Convolutional Neural Networks Using Energy-Aware Pruning
Tien-Ju Yang, Yu-Hsin Chen, Vivienne Sze
Joint Multi-Person Pose Estimation and Semantic Part Segmentation
Fangting Xia, Peng Wang, Xianjie Chen, Alan L. Yuille
Deep Feature Interpolation for Image Content Changes
Paul Upchurch, Jacob Gardner, Geoff Pleiss, Robert Pless, Noah Snavely, Kavita Bala, Kilian Weinberger
FASON: First and Second Order Information Fusion Network for Texture Recognition
Xiyang Dai, Joe Yue-Hei Ng, Larry S. Davis
Lean Crowdsourcing: Combining Humans and Machines in an Online System
Steve Branson, Grant Van Horn, Pietro Perona
Object Recognition & Scene Understanding
Supervising Neural Attention Models for Video Captioning by Human Gaze Data
Youngjae Yu, Jongwook Choi, Yeonhwa Kim, Kyung Yoo, Sang-Hun Lee, Gunhee Kim
L2-Net: Deep Learning of Discriminative Patch Descriptor in Euclidean Space
Yurun Tian, Bin Fan, Fuchao Wu
Convolutional Random Walk Networks for Semantic Image Segmentation
Gedas Bertasius, Lorenzo Torresani, Stella X. Yu, Jianbo Shi
Knowledge Acquisition for Visual Question Answering via Iterative Querying
Yuke Zhu, Joseph J. Lim, Li Fei-Fei
Memory-Augmented Attribute Manipulation Networks for Interactive Fashion Search
Bo Zhao, Jiashi Feng, Xiao Wu, Shuicheng Yan
From Zero-Shot Learning to Conventional Supervised Classification: Unseen Visual Data Synthesis
Yang Long, Li Liu, Ling Shao, Fumin Shen, Guiguang Ding, Jungong Han
Are Large-Scale 3D Models Really Necessary for Accurate Visual Localization?
Torsten Sattler, Akihiko Torii, Josef Sivic, Marc Pollefeys, Hajime Taira, Masatoshi Okutomi, Tomas Pajdla
Asymmetric Feature Maps With Application to Sketch Based Retrieval
Giorgos Tolias, OndÅ™ej Chum
Diverse Image Annotation
Baoyuan Wu, Fan Jia, Wei Liu, Bernard Ghanem
AMC: Attention guided Multi-modal Correlation Learning for Image Search
Kan Chen, Trung Bui, Chen Fang, Zhaowen Wang, Ram Nevatia
Multi-Attention Network for One Shot Learning
Peng Wang, Lingqiao Liu, Chunhua Shen, Zi Huang, Anton van den Hengel, Heng Tao Shen
Fried Binary Embedding for High-Dimensional Visual Features
Weixiang Hong, Junsong Yuan, Sreyasee Das Bhattacharjee
Pyramid Scene Parsing Network
Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia
Learning Deep Match Kernels for Image-Set Classification
Haoliang Sun, Xiantong Zhen, Yuanjie Zheng, Gongping Yang, Yilong Yin, Shuo Li
Task-Driven Dynamic Fusion: Reducing Ambiguity in Video Description
Xishan Zhang, Ke Gao, Yongdong Zhang, Dongming Zhang, Jintao Li, Qi Tian
Learning Multifunctional Binary Codes for Both Category and Attribute Oriented Retrieval Tasks
Haomiao Liu, Ruiping Wang, Shiguang Shan, Xilin Chen
Indoor Scene Parsing With Instance Segmentation, Semantic Labeling and Support Relationship Inference
Wei Zhuo, Mathieu Salzmann, Xuming He, Miaomiao Liu
Episodic CAMN: Contextual Attention-Based Memory Networks With Iterative Feedback for Scene Labeling
Abrar H. Abdulnabi, Bing Shuai, Stefan Winkler, Gang Wang
Link the Head to the â€œBeakâ€: Zero Shot Learning From Noisy Text Description at Part Precision
Mohamed Elhoseiny, Yizhe Zhu, Han Zhang, Ahmed Elgammal
SCA-CNN: Spatial and Channel-Wise Attention in Convolutional Networks for Image Captioning
Long Chen, Hanwang Zhang, Jun Xiao, Liqiang Nie, Jian Shao, Wei Liu, Tat-Seng Chua
Deep Pyramidal Residual Networks
Dongyoon Han, Jiwhan Kim, Junmo Kim
Product Split Trees
Artem Babenko, Victor Lempitsky
Making the v in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering
Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, Devi Parikh
Commonly Uncommon: Semantic Sparsity in Situation Recognition
Mark Yatskar, Vicente Ordonez, Luke Zettlemoyer, Ali Farhadi
Cross-Modality Binary Code Learning via Fusion Similarity Hashing
Hong Liu, Rongrong Ji, Yongjian Wu, Feiyue Huang, Baochang Zhang
Theory
Saliency Revisited: Analysis of Mouse Movements Versus Fixations
Hamed R. Tavakoli, Fawad Ahmed, Ali Borji, Jorma Laaksonen
InterpoNet, a Brain Inspired Neural Network for Optical Flow Dense Interpolation
Shay Zweig, Lior Wolf
Video Analytics
SST: Single-Stream Temporal Action Proposals
Shyamal Buch, Victor Escorcia, Chuanqi Shen, Bernard Ghanem, Juan Carlos Niebles
Video Segmentation via Multiple Granularity Analysis
Rui Yang, Bingbing Ni, Chao Ma, Yi Xu, Xiaokang Yang
Spatio-Temporal Alignment of Non-Overlapping Sequences From Independently Panning Cameras
Seyed Morteza Safdarnejad, Xiaoming Liu
UntrimmedNets for Weakly Supervised Action Recognition and Detection
Limin Wang, Yuanjun Xiong, Dahua Lin, Luc Van Gool
Object Recognition & Scene Understanding 3
Spotlight 4-2A
Gaze Embeddings for Zero-Shot Image Classification
Nour Karessli, Zeynep Akata, Bernt Schiele, Andreas Bulling
What's in a Question: Using Visual Questions as a Form of Supervision
Siddha Ganju, Olga Russakovsky, Abhinav Gupta
Attend to You: Personalized Image Captioning With Context Sequence Memory Networks
Cesc Chunseong Park, Byeongchang Kim, Gunhee Kim
Adversarially Tuned Scene Generation
VSR Veeravasarapu, Constantin Rothkopf, Ramesh Visvanathan
Residual Attention Network for Image Classification
Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang
Not All Pixels Are Equal: Difficulty-Aware Semantic Segmentation via Deep Layer Cascade
Xiaoxiao Li, Ziwei Liu, Ping Luo, Chen Change Loy, Xiaoou Tang
Learning Non-Maximum Suppression
Jan Hosang, Rodrigo Benenson, Bernt Schiele
The Amazing Mysteries of the Gutter: Drawing Inferences Between Panels in Comic Book Narratives
Mohit Iyyer, Varun Manjunatha, Anupam Guha, Yogarshi Vyas, Jordan Boyd-Graber, Hal Daumé III, Larry S. Davis
Oral 4-2A
Object Region Mining With Adversarial Erasing: A Simple Classification to Semantic Segmentation Approach
Yunchao Wei, Jiashi Feng, Xiaodan Liang, Ming-Ming Cheng, Yao Zhao, Shuicheng Yan
Fine-Grained Recognition as HSnet Search for Informative Image Parts
Michael Lam, Behrooz Mahasseni, Sinisa Todorovic
G2DeNet: Global Gaussian Distribution Embedding Network and Its Application to Visual Recognition
Qilong Wang, Peihua Li, Lei Zhang
YOLO9000: Better, Faster, Stronger
Joseph Redmon, Ali Farhadi
Machine Learning for 3D Vision
Spotlight 4-2B
Multi-View 3D Object Detection Network for Autonomous Driving
Xiaozhi Chen, Huimin Ma, Ji Wan, Bo Li, Tian Xia
UltraStereo: Efficient Learning-Based Matching for Active Stereo Systems
Sean Ryan Fanello, Julien Valentin, Christoph Rhemann, Adarsh Kowdle, Vladimir Tankovich, Philip Davidson, Shahram Izadi
Shape Completion Using 3D-Encoder-Predictor CNNs and Shape Synthesis
Angela Dai, Charles Ruizhongtai Qi, Matthias NieÃŸner
Geometric Loss Functions for Camera Pose Regression With Deep Learning
Alex Kendall, Roberto Cipolla
CNN-SLAM: Real-Time Dense Monocular SLAM With Learned Depth Prediction
Keisuke Tateno, Federico Tombari, Iro Laina, Nassir Navab
Learning From Noisy Large-Scale Datasets With Minimal Supervision
Andreas Veit, Neil Alldrin, Gal Chechik, Ivan Krasin, Abhinav Gupta, Serge Belongie
SyncSpecCNN: Synchronized Spectral CNN for 3D Shape Segmentation
Li Yi, Hao Su, Xingwen Guo, Leonidas J. Guibas
Non-Local Deep Features for Salient Object Detection
Zhiming Luo, Akshaya Mishra, Andrew Achkar, Justin Eichel, Shaozi Li, Pierre-Marc Jodoin
Oral 4-2B
Unsupervised Monocular Depth Estimation With Left-Right Consistency
Clément Godard, Oisin Mac Aodha, Gabriel J. Brostow
Unsupervised Learning of Depth and Ego-Motion From Video
Tinghui Zhou, Matthew Brown, Noah Snavely, David G. Lowe
OctNet: Learning Deep 3D Representations at High Resolutions
Gernot Riegler, Ali Osman Ulusoy, Andreas Geiger
3D Shape Segmentation With Projective Convolutional Networks
Evangelos Kalogerakis, Melinos Averkiou, Subhransu Maji, Siddhartha Chaudhuri
Poster 4-2
3D Computer Vision
SGM-Nets: Semi-Global Matching With Neural Networks
Akihito Seki, Marc Pollefeys
Stereo-Based 3D Reconstruction of Dynamic Fluid Surfaces by Global Optimization
Yiming Qian, Minglun Gong, Yee-Hong Yang
Fine-To-Coarse Global Registration of RGB-D Scans
Maciej Halber, Thomas Funkhouser
Analyzing Computer Vision Data - The Good, the Bad and the Ugly
Oliver Zendel, Katrin Honauer, Markus Murschitz, Martin Humenberger, Gustavo FernÃ¡ndez DomÃ­nguez
Product Manifold Filter: Non-Rigid Shape Correspondence via Kernel Density Estimation in the Product Space
Matthias Vestner, Roee Litman, Emanuele RodolÃ , Alex Bronstein, Daniel Cremers
Unsupervised Vanishing Point Detection and Camera Calibration From a Single Manhattan Image With Radial Distortion
Michel Antunes, JoÃ£o P. Barreto, Djamila Aouada, BjÃ¶rn Ottersten
Toroidal Constraints for Two-Point Localization Under High Outlier Ratios
Federico Camposeco, Torsten Sattler, Andrea Cohen, Andreas Geiger, Marc Pollefeys
4D Light Field Superpixel and Segmentation
Hao Zhu, Qi Zhang, Qing Wang
Exploiting Symmetry and/or Manhattan Properties for 3D Object Structure Estimation From Single and Multiple Images
Yuan Gao, Alan L. Yuille
Analyzing Humans in Images
Binary Coding for Partial Action Analysis With Limited Observation Ratios
Jie Qin, Li Liu, Ling Shao, Bingbing Ni, Chen Chen, Fumin Shen, Yunhong Wang
SphereFace: Deep Hypersphere Embedding for Face Recognition
Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, Le Song
IRINA: Iris Recognition Abstract[PDF](Even) in Inaccurately Segmented Data
Hugo ProenÃ§a, JoÃ£o C. Neves
Look Into Person: Self-Supervised Structure-Sensitive Learning and a New Benchmark for Human Parsing
Ke Gong, Xiaodan Liang, Dongyu Zhang, Xiaohui Shen, Liang Lin
Action Unit Detection With Region Adaptation, Multi-Labeling Learning and Optimal Temporal Fusing
Wei Li, Farnaz Abtahi, Zhigang Zhu
See the Forest for the Trees: Joint Spatial and Temporal Recurrent Neural Networks for Video-Based Person Re-Identification
Zhen Zhou, Yan Huang, Wei Wang, Liang Wang, Tieniu Tan
Joint Intensity and Spatial Metric Learning for Robust Gait Recognition
Yasushi Makihara, Atsuyuki Suzuki, Daigo Muramatsu, Xiang Li, Yasushi Yagi
Pose-Aware Person Recognition
Vijay Kumar, Anoop Namboodiri, Manohar Paluri, C. V. Jawahar
Not Afraid of the Dark: NIR-VIS Face Recognition via Cross-Spectral Hallucination and Low-Rank Embedding
José Lezama, Qiang Qiu, Guillermo Sapiro
Applications
Jointly Learning Energy Expenditures and Activities Using Egocentric Multimodal Signals
Katsuyuki Nakamura, Serena Yeung, Alexandre Alahi, Li Fei-Fei
Binarized Mode Seeking for Scalable Visual Pattern Discovery
Wei Zhang, Xiaochun Cao, Rui Wang, Yuanfang Guo, Zhineng Chen
Scribbler: Controlling Deep Image Synthesis With Sketch and Color
Patsorn Sangkloy, Jingwan Lu, Chen Fang, Fisher Yu, James Hays
Biomedical Image/Video Analysis
Multi-Way Multi-Level Kernel Modeling for Neuroimaging Classification
Lifang He, Chun-Ta Lu, Hao Ding, Shen Wang, Linlin Shen, Philip S. Yu, Ann B. Ragin
WSISA: Making Survival Prediction From Whole Slide Histopathological Images
Xinliang Zhu, Jiawen Yao, Feiyun Zhu, Junzhou Huang
Computational Photography
On the Effectiveness of Visible Watermarks
Tali Dekel, Michael Rubinstein, Ce Liu, William T. Freeman
Snapshot Hyperspectral Light Field Imaging
Zhiwei Xiong, Lizhi Wang, Huiqun Li, Dong Liu, Feng Wu
Semantic Image Inpainting With Deep Generative Models
Raymond A. Yeh, Chen Chen, Teck Yian Lim, Alexander G. Schwing, Mark Hasegawa-Johnson, Minh N. Do
Image Motion & Tracking
Fast Multi-Frame Stereo Scene Flow With Motion Segmentation
Tatsunori Taniai, Sudipta N. Sinha, Yoichi Sato
Improved Stereo Matching With Constant Highway Networks and Reflective Confidence Learning
Amit Shaked, Lior Wolf
Optical Flow in Mostly Rigid Scenes
Jonas Wulff, Laura Sevilla-Lara, Michael J. Black
Optical Flow Requires Multiple Strategies Abstract[PDF](but Only One Network)
Tal Schuster, Lior Wolf, David Gadot
ECO: Efficient Convolution Operators for Tracking
Martin Danelljan, Goutam Bhat, Fahad Shahbaz Khan, Michael Felsberg
Low- & Mid-Level Vision
Differential Angular Imaging for Material Recognition
Jia Xue, Hang Zhang, Kristin Dana, Ko Nishino
Fast Fourier Color Constancy
Jonathan T. Barron, Yun-Ta Tsai
Comparative Evaluation of Hand-Crafted and Learned Local Features
Johannes L. SchÃ¶nberger, Hans Hardmeier, Torsten Sattler, Marc Pollefeys
Learning Fully Convolutional Networks for Iterative Non-Blind Deconvolution
Jiawei Zhang, Jinshan Pan, Wei-Sheng Lai, Rynson W. H. Lau, Ming-Hsuan Yang
Image Deblurring via Extreme Channels Prior
Yanyang Yan, Wenqi Ren, Yuanfang Guo, Rui Wang, Xiaochun Cao
Simultaneous Stereo Video Deblurring and Scene Flow Estimation
Liyuan Pan, Yuchao Dai, Miaomiao Liu, Fatih Porikli
Deep Photo Style Transfer
Fujun Luan, Sylvain Paris, Eli Shechtman, Kavita Bala
Generative Attribute Controller With Conditional Filtered Generative Adversarial Networks
Takuhiro Kaneko, Kaoru Hiramatsu, Kunio Kashino
Fast Haze Removal for Nighttime Image Using Maximum Reflectance Prior
Jing Zhang, Yang Cao, Shuai Fang, Yu Kang, Chang Wen Chen
Machine Learning
Low-Rank Bilinear Pooling for Fine-Grained Classification
Shu Kong, Charless Fowlkes
Neural Scene De-Rendering
Jiajun Wu, Joshua B. Tenenbaum, Pushmeet Kohli
Real-Time Neural Style Transfer for Videos
Haozhi Huang, Hao Wang, Wenhan Luo, Lin Ma, Wenhao Jiang, Xiaolong Zhu, Zhifeng Li, Wei Liu
A Graph Regularized Deep Neural Network for Unsupervised Image Representation Learning
Shijie Yang, Liang Li, Shuhui Wang, Weigang Zhang, Qingming Huang
Collaborative Deep Reinforcement Learning for Joint Object Search
Xiangyu Kong, Bo Xin, Yizhou Wang, Gang Hua
Loss Max-Pooling for Semantic Image Segmentation
Samuel Rota BulÃ², Gerhard Neuhold, Peter Kontschieder
Deep View Morphing
Dinghuang Ji, Junghyun Kwon, Max McFarland, Silvio Savarese
Unsupervised Learning of Long-Term Motion Dynamics for Videos
Zelun Luo, Boya Peng, De-An Huang, Alexandre Alahi, Li Fei-Fei
Revisiting Metric Learning for SPD Matrix Based Visual Representation
Luping Zhou, Lei Wang, Jianjia Zhang, Yinghuan Shi, Yang Gao
Expert Gate: Lifelong Learning With a Network of Experts
Rahaf Aljundi, Punarjay Chakravarty, Tinne Tuytelaars
A Gift From Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
Junho Yim, Donggyu Joo, Jihoon Bae, Junmo Kim
Domain Adaptation by Mixture of Alignments of Second- or Higher-Order Scatter Tensors
Piotr Koniusz, Yusuf Tas, Fatih Porikli
Deep Mixture of Linear Inverse Regressions Applied to Head-Pose Estimation
Stéphane Lathuilière, Rémi Juge, Pablo Mesejo, Rafael Muñoz-Salinas, Radu Horaud
STD2P: RGBD Semantic Segmentation Using Spatio-Temporal Data-Driven Pooling
Yang He, Wei-Chen Chiu, Margret Keuper, Mario Fritz
Harmonic Networks: Deep Translation and Rotation Equivariance
Daniel E. Worrall, Stephan J. Garbin, Daniyar Turmukhambetov, Gabriel J. Brostow
Multimodal Transfer: A Hierarchical Deep Convolutional Neural Network for Fast Artistic Style Transfer
Xin Wang, Geoffrey Oxholm, Da Zhang, Yuan-Fang Wang
Detect, Replace, Refine: Deep Structured Prediction for Pixel Wise Labeling
Spyros Gidaris, Nikos Komodakis
Weighted-Entropy-Based Quantization for Deep Neural Networks
Eunhyeok Park, Junwhan Ahn, Sungjoo Yoo
Residual Expansion Algorithm: Fast and Effective Optimization for Nonconvex Least Squares Problems
Daiki Ikami, Toshihiko Yamasaki, Kiyoharu Aizawa
Bidirectional Beam Search: Forward-Backward Inference in Neural Sequence Models for Fill-In-The-Blank Image Captioning
Qing Sun, Stefan Lee, Dhruv Batra
Newton-Type Methods for Inference in Higher-Order Markov Random Fields
Hariprasad Kannan, Nikos Komodakis, Nikos Paragios
Adaptive Relaxed ADMM: Convergence Theory and Practical Implementation
Zheng Xu, MÃ¡rio A. T. Figueiredo, Xiaoming Yuan, Christoph Studer, Tom Goldstein
Object Recognition & Scene Understanding
ViP-CNN: Visual Phrase Guided Convolutional Neural Network
Yikang Li, Wanli Ouyang, Xiaogang Wang, Xiao'ou Tang
Instance-Aware Image and Sentence Matching With Selective Multimodal LSTM
Yan Huang, Wei Wang, Liang Wang
Kernel Square-Loss Exemplar Machines for Image Retrieval
Rafael S. Rezende, Joaquin Zepeda, Jean Ponce, Francis Bach, Patrick Pérez
Cognitive Mapping and Planning for Visual Navigation
Saurabh Gupta, James Davidson, Sergey Levine, Rahul Sukthankar, Jitendra Malik
Combining Bottom-Up, Top-Down, and Smoothness Cues for Weakly Supervised Image Segmentation
Anirban Roy, Sinisa Todorovic
Seeing Into Darkness: Scotopic Visual Recognition
Bo Chen, Pietro Perona
Deep Co-Occurrence Feature Learning for Visual Object Recognition
Ya-Fang Shih, Yang-Ming Yeh, Yen-Yu Lin, Ming-Fang Weng, Yi-Chang Lu, Yung-Yu Chuang
An Empirical Evaluation of Visual Question Answering for Novel Objects
Santhosh K. Ramakrishnan, Ambar Pal, Gaurav Sharma, Anurag Mittal
InstanceCut: From Edges to Instances With MultiCut
Alexander Kirillov, Evgeny Levinkov, Bjoern Andres, Bogdan Savchynskyy, Carsten Rother
Fine-Grained Image Classification via Combining Vision and Language
Xiangteng He, Yuxin Peng
Mimicking Very Efficient Network for Object Detection
Quanquan Li, Shengying Jin, Junjie Yan
Tracking by Natural Language Specification
Zhenyang Li, Ran Tao, Efstratios Gavves, Cees G. M. Snoek, Arnold W.M. Smeulders
A Dataset and Exploration of Models for Understanding Video Data Through Fill-In-The-Blank Question-Answering
Tegan Maharaj, Nicolas Ballas, Anna Rohrbach, Aaron Courville, Christopher Pal
Learning Detection With Diverse Proposals
Samaneh Azadi, Jiashi Feng, Trevor Darrell
Skeleton Key: Image Captioning by Skeleton-Attribute Decomposition
Yufei Wang, Zhe Lin, Xiaohui Shen, Scott Cohen, Garrison W. Cottrell
Theory
A Low Power, Fully Event-Based Gesture Recognition System
Arnon Amir, Brian Taba, David Berg, Timothy Melano, Jeffrey McKinstry, Carmelo Di Nolfo, Tapan Nayak, Alexander Andreopoulos, Guillaume Garreau, Marcela Mendoza, Jeff Kusnitz, Michael Debole, Steve Esser, Tobi Delbruck, Myron Flickner, Dharmendra Modha
Video Analytics
Learning Deep Context-Aware Features Over Body and Latent Parts for Person Re-Identification
Dangwei Li, Xiaotang Chen, Zhang Zhang, Kaiqi Huang
Recurrent Modeling of Interaction Context for Collective Activity Recognition
Minsi Wang, Bingbing Ni, Xiaokang Yang
Primary Object Segmentation in Videos Based on Region Augmentation and Reduction
Yeong Jun Koh, Chang-Su Kim
ROAM: A Rich Object Appearance Model With Application to Rotoscoping
Ondrej Miksik, Juan-Manuel Pérez-RÃºa, Philip H. S. Torr, Patrick Pérez
Temporal Residual Networks for Dynamic Scene Recognition
Christoph Feichtenhofer, Axel Pinz, Richard P. Wildes
Spatiotemporal Multiplier Networks for Video Action Recognition
Christoph Feichtenhofer, Axel Pinz, Richard P. Wildes
Learning to Learn From Noisy Web Videos
Serena Yeung, Vignesh Ramanathan, Olga Russakovsky, Liyue Shen, Greg Mori, Li Fei-Fei
YouTube-BoundingBoxes: A Large High-Precision Human-Annotated Data Set for Object Detection in Video
Esteban Real, Jonathon Shlens, Stefano Mazzocchi, Xin Pan, Vincent Vanhoucke
Online Video Object Segmentation via Convolutional Trident Network
Won-Dong Jang, Chang-Su Kim
