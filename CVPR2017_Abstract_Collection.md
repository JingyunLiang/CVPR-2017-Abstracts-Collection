# CVPR 2017 Abstracts Collection
_Collection of CVPR 2017, including titles, links, authors, abstracts and my own comments. Created by Michael Liang, NUDT. All my work are based on http://www.cvpapers.com/cvpr2017.html 
It is a convient project for CVPR fast reading. Some information are missing, and I hope we can work together for a better collection._

## Machine Learning 1

Spotlight 1-1A

#### Exclusivity-Consistency Regularized Multi-View Subspace Clustering

Xiaobo Wang, Xiaojie Guo, Zhen Lei, Changqing Zhang, Stan Z. Li

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### *Borrowing Treasures From the Wealthy: Deep Transfer Learning Through Selective Joint Fine-Tuning  [PDF](https://arxiv.org/abs/1702.08690)

Weifeng Ge, Yizhou Yu

_**Abstract**_: &emsp;_[Deep neural networks require a large amount of labeled training data during supervised learning. However, collecting and labeling so much data might be infeasible in many cases. In this paper, we introduce a source-target selective joint fine-tuning scheme for improving the performance of deep learning tasks with insufficient training data. In this scheme, a target learning task with insufficient training data is carried out simultaneously with another source learning task with abundant training data. However, the source learning task does not use all existing training data. Our core idea is to identify and use a subset of training images from the original source learning task whose low-level characteristics are similar to those from the target learning task, and jointly fine-tune shared convolutional layers for both tasks. Specifically, we compute descriptors from linear or nonlinear filter bank responses on training images from both tasks, and use such descriptors to search for a desired subset of training samples for the source learning task. Experiments demonstrate that our selective joint fine-tuning scheme achieves state-of-the-art performance on multiple visual classification tasks with insufficient training data for deep learning. Such tasks include Caltech 256, MIT Indoor 67, Oxford Flowers 102 and Stanford Dogs 120. In comparison to fine-tuning without a source domain, the proposed method can improve the classification accuracy by 2% - 10% using a single model.]_

_**Comment**_: &emsp;_<a source-target selective joint fine-tuning scheme with insufficient training data; insufficient & abundant task with similar low-level feature trained simultaneously; share conv layers; help labelling data,>_

#### **The More You Know: Using Knowledge Graphs for Image Classification  [PDF](https://arxiv.org/abs/1612.04844)

Kenneth Marino, Ruslan Salakhutdinov, Abhinav Gupta

_**Abstract**_: &emsp;_[One characteristic that sets humans apart from modern learning-based computer vision algorithms is the ability to acquire knowledge about the world and use that knowledge to reason about the visual world. Humans can learn about the characteristics of objects and the relationships that occur between them to learn a large variety of visual concepts, often with few examples. This paper investigates the use of structured prior knowledge in the form of knowledge graphs and shows that using this knowledge improves performance on image classification. We build on recent work on end-to-end learning on graphs, introducing the Graph Search Neural Network as a way of efficiently incorporating large knowledge graphs into a vision classification pipeline. We show in a number of experiments that our method outperforms standard neural network baselines for multi-label classification.]_

_**Comment**_: &emsp;_< the real way to AI >_

#### Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs  [PDF](https://arxiv.org/abs/1704.02901)

Martin Simonovsky, Nikos Komodakis

_**Abstract**_: &emsp;_[A number of problems can be formulated as prediction on graph-structured data. In this work, we generalize the convolution operator from regular grids to arbitrary graphs while avoiding the spectral domain, which allows us to handle graphs of varying size and connectivity. To move beyond a simple diffusion, filter weights are conditioned on the specific edge labels in the neighborhood of a vertex. Together with the proper choice of graph coarsening, we explore constructing deep neural networks for graph classification. In particular, we demonstrate the generality of our formulation in point cloud classification, where we set the new state of the art, and on a graph classification dataset, where we outperform other deep learning approaches.]_

_**Comment**_: &emsp;_< graph-structured data >_

#### *Convolutional Neural Network Architecture for Geometric Matching  [PDF](https://arxiv.org/abs/1703.05593)

Ignacio Rocco, Relja ArandjeloviÄ‡, Josef Sivic

_**Abstract**_: &emsp;_[We address the problem of determining correspondences between two images in agreement with a geometric model such as an affine or thin-plate spline transformation, and estimating its parameters. The contributions of this work are three-fold. First, we propose a convolutional neural network architecture for geometric matching. The architecture is based on three main components that zmimic the standard steps of feature extraction, matching and simultaneous inlier detection and model parameter estimation, while being trainable end-to-end. Second, we demonstrate that the network parameters can be trained from synthetically generated imagery without the need for manual annotation and that our matching layer significantly increases generalization capabilities to never seen before images. Finally, we show that the same model can perform both instance-level and category-level matching giving state-of-the-art results on the challenging Proposal Flow dataset.]_

_**Comment**_: &emsp;_< geomatric matching; synthetically generated image; instance-level& category-level; matching layer,>_ 

#### Deep Affordance-Grounded Sensorimotor Object Recognition  [PDF](https://arxiv.org/abs/1704.02787)

Spyridon Thermos, Georgios Th. Papadopoulos, Petros Daras, Gerasimos Potamianos

_**Abstract**_: &emsp;_[It is well-established by cognitive neuroscience that human perception of objects constitutes a complex process, where object appearance information is combined with evidence about the so-called object "affordances", namely the types of actions that humans typically perform when interacting with them. This fact has recently motivated the "sensorimotor" approach to the challenging task of automatic object recognition, where both information sources are fused to improve robustness. In this work, the aforementioned paradigm is adopted, surpassing current limitations of sensorimotor object recognition research. Specifically, the deep learning paradigm is introduced to the problem for the first time, developing a number of novel neuro-biologically and neuro-physiologically inspired architectures that utilize state-of-the-art neural networks for fusing the available information sources in multiple ways. The proposed methods are evaluated using a large RGB-D corpus, which is specifically collected for the task of sensorimotor object recognition and is made publicly available. Experimental results demonstrate the utility of affordance information to object recognition, achieving an up to 29% relative error reduction by its inclusion.]_

_**Comment**_: &emsp;_< Sensorimotor Object Recognition >_

#### Discovering Causal Signals in Images  [PDF](https://arxiv.org/abs/1605.08179)

David Lopez-Paz, Robert Nishihara, Soumith Chintala, Bernhard SchÃ¶lkopf, Léon Bottou

_**Abstract**_: &emsp;_[The purpose of this paper is to point out and assay observable causal signals within collections of static images. We achieve this goal in two steps. First, we take a learning approach to observational causal inference, and build a classifier that achieves state-of-the-art performance on finding the causal direction between pairs of random variables, when given samples from their joint distribution. Second, we use our causal direction finder to effectively distinguish between features of objects and features of their contexts in collections of static images. Our experiments demonstrate the existence of (1) a relation between the direction of causality and the difference between objects and their contexts, and (2) observable causal signals in collections of static images.]_

_**Comment**_: &emsp;_< causal singals>_

#### *On Compressing Deep Models by Low Rank and Sparse Decomposition

Xiyu Yu, Tongliang Liu, Xinchao Wang, Dacheng Tao

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

Oral 1-1A

#### PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation  [PDF](https://arxiv.org/abs/1612.00593)

Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas

_**Abstract**_: &emsp;_[Point cloud is an important type of geometric data structure. Due to its irregular format, most researchers transform such data to regular 3D voxel grids or collections of images. This, however, renders data unnecessarily voluminous and causes issues. In this paper, we design a novel type of neural network that directly consumes point clouds and well respects the permutation invariance of points in the input. Our network, named PointNet, provides a unified architecture for applications ranging from object classification, part segmentation, to scene semantic parsing. Though simple, PointNet is highly efficient and effective. Empirically, it shows strong performance on par or even better than state of the art. Theoretically, we provide analysis towards understanding of what the network has learnt and why the network is robust with respect to input perturbation and corruption.]_

_**Comment**_: &emsp;_< point cloud, 3D>_

#### ***Universal Adversarial Perturbations  [PDF](https://arxiv.org/abs/1610.08401)

Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, Pascal Frossard

_**Abstract**_: &emsp;_[Given a state-of-the-art deep neural network classifier, we show the existence of a universal (image-agnostic) and very small perturbation vector that causes natural images to be misclassified with high probability. We propose a systematic algorithm for computing universal perturbations, and show that state-of-the-art deep neural networks are highly vulnerable to such perturbations, albeit being quasi-imperceptible to the human eye. We further empirically analyze these universal perturbations and show, in particular, that they generalize very well across neural networks. The surprising existence of universal perturbations reveals important geometric correlations among the high-dimensional decision boundary of classifiers. It further outlines potential security breaches with the existence of single directions in the input space that adversaries can possibly exploit to break a classifier on most natural images.]_

_**Comment**_: &emsp;_< using unseen perturbations to break a classifier, destory DL>_

#### *Unsupervised Pixel-Level Domain Adaptation With Generative Adversarial Networks  [PDF](https://arxiv.org/abs/1612.05424)

Konstantinos Bousmalis, Nathan Silberman, David Dohan, Dumitru Erhan, Dilip Krishnan

_**Abstract**_: &emsp;_[Collecting well-annotated image datasets to train modern machine learning algorithms is prohibitively expensive for many tasks. One appealing alternative is rendering synthetic data where ground-truth annotations are generated automatically. Unfortunately, models trained purely on rendered images often fail to generalize to real images. To address this shortcoming, prior work introduced unsupervised domain adaptation algorithms that attempt to map representations between the two domains or learn to extract features that are domain-invariant. In this work, we present a new approach that learns, in an unsupervised manner, a transformation in the pixel space from one domain to the other. Our generative adversarial network (GAN)-based method adapts source-domain images to appear as if drawn from the target domain. Our approach not only produces plausible samples, but also outperforms the state-of-the-art on a number of unsupervised domain adaptation scenarios by large margins. Finally, we demonstrate that the adaptation process generalizes to object classes unseen during training.]_

_**Comment**_: &emsp;_< generate domain-invariant annotations, adapts source-domain images to appear as if drawn from the target domain ;GAN>_

#### Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (PDF, code)  [PDF](https://arxiv.org/pdf/1609.04802.pdf)  [PDF](https://github.com/leehomyc/Photo-Realistic-Super-Resoluton)

Christian Ledig, Lucas Theis, Ferenc HuszÃƒÂ¡r, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi

_**Abstract**_: &emsp;_[Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this paper, we present SRGAN, a generative adversarial network (GAN) for image super-resolution (SR). To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN. The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method.]_

_**Comment**_: &emsp;_< Super-Resolution, GAN>_

## 3D Vision 1

Spotlight 1-1B

#### Context-Aware Captions From Context-Agnostic Supervision  [PDF](https://arxiv.org/abs/1701.02870)

Ramakrishna Vedantam, Samy Bengio, Kevin Murphy, Devi Parikh, Gal Chechik

_**Abstract**_: &emsp;_[We introduce an inference technique to produce discriminative context-aware image captions (captions that describe differences between images or visual concepts) using only generic context-agnostic training data (captions that describe a concept or an image in isolation). For example, given images and captions of "siamese cat" and "tiger cat", we generate language that describes the "siamese cat" in a way that distinguishes it from "tiger cat". Our key novelty is that we show how to do joint inference over a language model that is context-agnostic and a listener which distinguishes closely-related concepts. We first apply our technique to a justification task, namely to describe why an image contains a particular fine-grained category as opposed to another closely-related category of the CUB-200-2011 dataset. We then study discriminative image captioning to generate language that uniquely refers to one of two semantically-similar images in the COCO dataset. Evaluations with discriminative ground truth for justification and human studies for discriminative image captioning reveal that our approach outperforms baseline generative and speaker-listener approaches for discrimination. ]_

_**Comment**_: &emsp;_< distinguished image caption>_

#### Global Hypothesis Generation for 6D Object Pose Estimation  [PDF](https://arxiv.org/abs/1612.02287)

Frank Michel, Alexander Kirillov, Eric Brachmann, Alexander Krull, Stefan Gumhold, Bogdan Savchynskyy, Carsten Rother

_**Abstract**_: &emsp;_[This paper addresses the task of estimating the 6D pose of a known 3D object from a single RGB-D image. Most modern approaches solve this task in three steps: i) Compute local features; ii) Generate a pool of pose-hypotheses; iii) Select and refine a pose from the pool. This work focuses on the second step. While all existing approaches generate the hypotheses pool via local reasoning, e.g. RANSAC or Hough-voting, we are the first to show that global reasoning is beneficial at this stage. In particular, we formulate a novel fully-connected Conditional Random Field (CRF) that outputs a very small number of pose-hypotheses. Despite the potential functions of the CRF being non-Gaussian, we give a new and efficient two-step optimization procedure, with some guarantees for optimality. We utilize our global hypotheses generation procedure to produce results that exceed state-of-the-art for the challenging "Occluded Object Dataset".]_

_**Comment**_: &emsp;_< pose estimation>_

#### A Practical Method for Fully Automatic Intrinsic Camera Calibration Using Directionally Encoded Light

Mahdi Abbaspour Tehrani, Thabo Beeler, Anselm GrundhÃ¶fer

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< camera calibration >_

#### CATS: A Color and Thermal Stereo Benchmark

Wayne Treible, Philip Saponaro, Scott Sorensen, Abhishek Kolagunda, Michael O'Neal, Brian Phelan, Kelly Sherbondy, Chandra Kambhamettu

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Elastic Shape-From-Template With Spatially Sparse Deforming Forces

Abed Malti, Cédric Herzet

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Distinguishing the Indistinguishable: Exploring Structural Ambiguities via Geodesic Context

Qingan Yan, Long Yang, Ling Zhang, Chunxia Xiao

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Multi-Scale Continuous CRFs as Sequential Deep Networks for Monocular Depth Estimation  [PDF](https://arxiv.org/abs/1704.02157)

Dan Xu, Elisa Ricci, Wanli Ouyang, Xiaogang Wang, Nicu Sebe

_**Abstract**_: &emsp;_[This paper addresses the problem of depth estimation from a single still image. Inspired by recent works on multi- scale convolutional neural networks (CNN), we propose a deep model which fuses complementary information derived from multiple CNN side outputs. Different from previous methods, the integration is obtained by means of continuous Conditional Random Fields (CRFs). In particular, we propose two different variations, one based on a cascade of multiple CRFs, the other on a unified graphical model. By designing a novel CNN implementation of mean-field updates for continuous CRFs, we show that both proposed models can be regarded as sequential deep networks and that training can be performed end-to-end. Through extensive experimental evaluation we demonstrate the effective- ness of the proposed approach and establish new state of the art results on publicly available datasets. ]_

_**Comment**_: &emsp;_< depth estimation from a single still image; using multi-scale CNN's side output, integrated by continuous Conditional Random Fields (CRFs).>_

#### Dynamic Time-Of-Flight

Michael Schober, Amit Adam, Omer Yair, Shai Mazor, Sebastian Nowozin

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

Oral 1-1B

#### Semantic Scene Completion From a Single Depth Image  [PDF](https://arxiv.org/abs/1611.08974)

Shuran Song, Fisher Yu, Andy Zeng, Angel X. Chang, Manolis Savva, Thomas Funkhouser

_**Abstract**_: &emsp;_[This paper focuses on semantic scene completion, a task for producing a complete 3D voxel representation of volumetric occupancy and semantic labels for a scene from a single-view depth map observation. Previous work has considered scene completion and semantic labeling of depth maps separately. However, we observe that these two problems are tightly intertwined. To leverage the coupled nature of these two tasks, we introduce the semantic scene completion network (SSCNet), an end-to-end 3D convolutional network that takes a single depth image as input and simultaneously outputs occupancy and semantic labels for all voxels in the camera view frustum. Our network uses a dilation-based 3D context module to efficiently expand the receptive field and enable 3D context learning. To train our network, we construct SUNCG - a manually created large-scale dataset of synthetic 3D scenes with dense volumetric annotations. Our experiments demonstrate that the joint model outperforms methods addressing each task in isolation and outperforms alternative approaches on the semantic scene completion task. ]_

_**Comment**_: &emsp;_< >_

#### 3DMatch: Learning Local Geometric Descriptors From RGB-D Reconstructions  [PDF](https://arxiv.org/abs/1603.08182)  [PDF](http://3dmatch.cs.princeton.edu/)

Andy Zeng, Shuran Song, Matthias NieÃŸner, Matthew Fisher, Jianxiong Xiao, Thomas Funkhouser

_**Abstract**_: &emsp;_[Matching local geometric features on real-world depth images is a challenging task due to the noisy, low-resolution, and incomplete nature of 3D scan data. These difficulties limit the performance of current state-of-art methods, which are typically based on histograms over geometric properties. In this paper, we present 3DMatch, a data-driven model that learns a local volumetric patch descriptor for establishing correspondences between partial 3D data. To amass training data for our model, we propose a self-supervised feature learning method that leverages the millions of correspondence labels found in existing RGB-D reconstructions. Experiments show that our descriptor is not only able to match local geometry in new scenes for reconstruction, but also generalize to different tasks and spatial scales (e.g. instance-level object model alignment for the Amazon Picking Challenge, and mesh surface correspondence). Results show that 3DMatch consistently outperforms other state-of-the-art approaches by a significant margin. Code, data, benchmarks, and pre-trained models are available online at this http URL]_

_**Comment**_: &emsp;_< >_

#### Multi-View Supervision for Single-View Reconstruction via Differentiable Ray Consistency (PDF, project,code)  [PDF](https://arxiv.org/pdf/1704.06254.pdf)  [PDF](https://shubhtuls.github.io/drc/)

Shubham Tulsiani, Tinghui Zhou, Alexei A. Efros, Jitendra Malik

_**Abstract**_: &emsp;_[We study the notion of consistency between a 3D shape and a 2D observation and propose a differentiable formulation which allows computing gradients of the 3D shape given an observation from an arbitrary view. We do so by reformulating view consistency using a differentiable ray consistency (DRC) term. We show that this formulation can be incorporated in a learning framework to leverage different types of multi-view observations e.g. foreground masks, depth, color images, semantics etc. as supervision for learning single-view 3D prediction. We present empirical analysis of our technique in a controlled setting. We also show that this approach allows us to improve over existing techniques for single-view reconstruction of objects from the PASCAL VOC dataset. ]_

_**Comment**_: &emsp;_< >_

#### On-The-Fly Adaptation of Regression Forests for Online Camera Relocalisation

Tommaso Cavallari, Stuart Golodetz, Nicholas A. Lord, Julien Valentin, Luigi Di Stefano, Philip H. S. Torr

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

## Low- & Mid-Level Vision

Spotlight 1-1C

#### Designing Effective Inter-Pixel Information Flow for Natural Image Matting  [PDF](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwjv6dnwjNPUAhVJNbwKHTq5CiEQFggmMAA&url=http%3A%2F%2Fpeople.inf.ethz.ch%2Faksoyy%2Fpapers%2FCVPR17-ifm.pdf&usg=AFQjCNGFpObgwIjYNZiNW0K_Jb6OLtLcuw&cad=rjt)

YaÄŸiz Aksoy, TunÃ§ Ozan Aydin, Marc Pollefeys

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< Natural Image Matting>_

#### Deep Video Deblurring for Hand-Held Cameras  [PDF](https://arxiv.org/abs/1611.08387)

Shuochen Su, Mauricio Delbracio, Jue Wang, Guillermo Sapiro, Wolfgang Heidrich, Oliver Wang

_**Abstract**_: &emsp;_[Motion blur from camera shake is a major problem in videos captured by hand-held devices. Unlike single-image deblurring, video-based approaches can take advantage of the abundant information that exists across neighboring frames. As a result the best performing methods rely on aligning nearby frames. However, aligning images is a computationally expensive and fragile procedure, and methods that aggregate information must therefore be able to identify which regions have been accurately aligned and which have not, a task which requires high level scene understanding. In this work, we introduce a deep learning solution to video deblurring, where a CNN is trained end-to-end to learn how to accumulate information across frames. To train this network, we collected a dataset of real videos recorded with a high framerate camera, which we use to generate synthetic motion blur for supervision. We show that the features learned from this dataset extend to deblurring motion blur that arises due to camera shake in a wide range of videos, and compare the quality of results to a number of other baselines. ]_

_**Comment**_: &emsp;_< Motion blur from camera shake, CNN>_

#### Instance-Level Salient Object Segmentation  [PDF](https://arxiv.org/abs/1704.03604)

Guanbin Li, Yuan Xie, Liang Lin, Yizhou Yu

_**Abstract**_: &emsp;_[Image saliency detection has recently witnessed rapid progress due to deep convolutional neural networks. However, none of the existing methods is able to identify object instances in the detected salient regions. In this paper, we present a salient instance segmentation method that produces a saliency mask with distinct object instance labels for an input image. Our method consists of three steps, estimating saliency map, detecting salient object contours and identifying salient object instances. For the first two steps, we propose a multiscale saliency refinement network, which generates high-quality salient region masks and salient object contours. Once integrated with multiscale combinatorial grouping and a MAP-based subset optimization framework, our method can generate very promising salient object instance segmentation results. To promote further research and evaluation of salient instance segmentation, we also construct a new database of 1000 images and their pixelwise salient instance annotations. Experimental results demonstrate that our proposed method is capable of achieving state-of-the-art performance on all public benchmarks for salient region detection as well as on our new dataset for salient instance segmentation. ]_

_**Comment**_: &emsp;_< estimating saliency map, detecting salient object contours and identifying salient object instances>_

#### Deep Multi-Scale Convolutional Neural Network for Dynamic Scene Deblurring  [PDF](https://arxiv.org/abs/1612.02177)

Seungjun Nah, Tae Hyun Kim, Kyoung Mu Lee

_**Abstract**_: &emsp;_[Non-uniform blind deblurring for general dynamic scenes is a challenging computer vision problem since blurs are caused by camera shake, scene depth as well as multiple object motions. To remove these complicated motion blurs, conventional energy optimization based methods rely on simple assumptions such that blur kernel is partially uniform or locally linear. Moreover, recent machine learning based methods also depend on synthetic blur datasets generated under these assumptions. This makes conventional deblurring methods fail to remove blurs where blur kernel is difficult to approximate or parameterize (e.g. object motion boundaries). In this work, we propose a multi-scale convolutional neural network that restores blurred images caused by various sources in an end-to-end manner. Furthermore, we present multi-scale loss function that mimics conventional coarse-to-fine approaches. Moreover, we propose a new large scale dataset that provides pairs of realistic blurry image and the corresponding ground truth sharp image that are obtained by a high-speed camera. With the proposed model trained on this dataset, we demonstrate empirically that our method achieves the state-of-the-art performance in dynamic scene deblurring not only qualitatively, but also quantitatively. ]_

_**Comment**_: &emsp;_< deblurring>_

#### *Diversified Texture Synthesis With Feed-Forward Networks  [PDF](https://arxiv.org/abs/1703.01664)

Yijun Li, Chen Fang, Jimei Yang, Zhaowen Wang, Xin Lu, Ming-Hsuan Yang

_**Abstract**_: &emsp;_[Recent progresses on deep discriminative and generative modeling have shown promising results on texture synthesis. However, existing feed-forward based methods trade off generality for efficiency, which suffer from many issues, such as shortage of generality (i.e., build one network per texture), lack of diversity (i.e., always produce visually identical output) and suboptimality (i.e., generate less satisfying visual effects). In this work, we focus on solving these issues for improved texture synthesis. We propose a deep generative feed-forward network which enables efficient synthesis of multiple textures within one single network and meaningful interpolation between them. Meanwhile, a suite of important techniques are introduced to achieve better convergence and diversity. With extensive experiments, we demonstrate the effectiveness of the proposed model and techniques for synthesizing a large number of textures and show its applications with the stylization. ]_

_**Comment**_: &emsp;_< Texture Synthesis using one network>_

Radiometric Calibration for Internet Photo Collections (PDF)  [PDF](http://alumni.media.mit.edu/~shiboxin/files/Mo_CVPR17.pdf)

Zhipeng Mo, Boxin Shi, Sai-Kit Yeung, Yasuyuki Matsushita

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< radiometric calibaration>_

#### Deeply Aggregated Alternating Minimization for Image Restoration  [PDF](https://arxiv.org/abs/1612.06508)

Youngjung Kim, Hyungjoo Jung, Dongbo Min, Kwanghoon Sohn

_**Abstract**_: &emsp;_[Regularization-based image restoration has remained an active research topic in computer vision and image processing. It often leverages a guidance signal captured in different fields as an additional cue. In this work, we present a general framework for image restoration, called deeply aggregated alternating minimization (DeepAM). We propose to train deep neural network to advance two of the steps in the conventional AM algorithm: proximal mapping and ?- continuation. Both steps are learned from a large dataset in an end-to-end manner. The proposed framework enables the convolutional neural networks (CNNs) to operate as a prior or regularizer in the AM algorithm. We show that our learned regularizer via deep aggregation outperforms the recent data-driven approaches as well as the nonlocalbased methods. The flexibility and effectiveness of our framework are demonstrated in several image restoration tasks, including single image denoising, RGB-NIR restoration, and depth super-resolution. ]_

_**Comment**_: &emsp;_< image restoration using deep learning>_

#### End-To-End Instance Segmentation With Recurrent Attention  [PDF](https://arxiv.org/abs/1605.09410)

Mengye Ren, Richard S. Zemel

_**Abstract**_: &emsp;_[While convolutional neural networks have gained impressive success recently in solving structured prediction problems such as semantic segmentation, it remains a challenge to differentiate individual object instances in the scene. Instance segmentation is very important in a variety of applications, such as autonomous driving, image captioning, and visual question answering. Techniques that combine large graphical models with low-level vision have been proposed to address this problem; however, we propose an end-to-end recurrent neural network (RNN) architecture with an attention mechanism to model a human-like counting process, and produce detailed instance segmentations. The network is jointly trained to sequentially produce regions of interest as well as a dominant object segmentation within each region. The proposed model achieves competitive results on the CVPPP, KITTI, and Cityscapes datasets. ]_

_**Comment**_: &emsp;_< instance segmentation using RNN with attention mechanism>_

Oral 1-1C

#### *SRN: Side-output Residual Network for Object Symmetry Detection in the Wild  [PDF](https://arxiv.org/abs/1703.02243)

Wei Ke, Jie Chen, Jianbin Jiao, Guoying Zhao, Qixiang Ye

_**Abstract**_: &emsp;_[In this paper, we establish a baseline for object symmetry detection in complex backgrounds by presenting a new benchmark and an end-to-end deep learning approach, opening up a promising direction for symmetry detection in the wild. The new benchmark, named Sym-PASCAL, spans challenges including object diversity, multi-objects, part-invisibility, and various complex backgrounds that are far beyond those in existing datasets. The proposed symmetry detection approach, named Side-output Residual Network (SRN), leverages output Residual Units (RUs) to fit the errors between the object symmetry groundtruth and the outputs of RUs. By stacking RUs in a deep-to-shallow manner, SRN exploits the 'flow' of errors among multiple scales to ease the problems of fitting complex outputs with limited layers, suppressing the complex backgrounds, and effectively matching object symmetry of different scales. Experimental results validate both the benchmark and its challenging aspects related to realworld images, and the state-of-the-art performance of our symmetry detection approach. The benchmark and the code for SRN are publicly available at this https URL]

_**Comment**_: &emsp;_< object symmetry detection by side-output residual net, new benchmark>_

Deep Image Matting   [PDF](https://arxiv.org/abs/1703.03872)

Ning Xu, Brian Price, Scott Cohen, Thomas Huang

_**Abstract**_: &emsp;_[Image matting is a fundamental computer vision problem and has many applications. Previous algorithms have poor performance when an image has similar foreground and background colors or complicated textures. The main reasons are prior methods 1) only use low-level features and 2) lack high-level context. In this paper, we propose a novel deep learning based algorithm that can tackle both these problems. Our deep model has two parts. The first part is a deep convolutional encoder-decoder network that takes an image and the corresponding trimap as inputs and predict the alpha matte of the image. The second part is a small convolutional network that refines the alpha matte predictions of the first network to have more accurate alpha values and sharper edges. In addition, we also create a large-scale image matting dataset including 49300 training images and 1000 testing images. We evaluate our algorithm on the image matting benchmark, our testing set, and a wide variety of real images. Experimental results clearly demonstrate the superiority of our algorithm over previous methods. ]_

_**Comment**_: &emsp;_< Image Matting>_

Wetness and Color From a Single Multispectral Image

Mihoko Shimano, Hiroki Okawa, Yuta Asano, Ryoma Bise, Ko Nishino, Imari Sato

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

FC4: Fully Convolutional Color Constancy With Confidence-Weighted Pooling

Yuanming Hu, Baoyuan Wang, Stephen Lin

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

# Poster 1-1

## 3D Computer Vision
Face Normals â€œIn-The-Wildâ€ Using Fully Convolutional Networks
George Trigeorgis, Patrick Snape, Iasonas Kokkinos, Stefanos Zafeiriou
A Non-Convex Variational Approach to Photometric Stereo Under Inaccurate Lighting
Yvain Quéau, Tao Wu, FranÃ§ois Lauze, Jean-Denis Durou, Daniel Cremers
A Linear Extrinsic Calibration of Kaleidoscopic Imaging System From Single 3D Point
Kosuke Takahashi, Akihiro Miyata, Shohei Nobuhara, Takashi Matsuyama
Polarimetric Multi-View Stereo
Zhaopeng Cui, Jinwei Gu, Boxin Shi, Ping Tan, Jan Kautz
An Exact Penalty Method for Locally Convergent Maximum Consensus (PDF, code)
Huu Le, Tat-Jun Chin, David Suter
Deep Supervision With Shape Concepts for Occlusion-Aware 3D Object Parsing
Chi Li, M. Zeeshan Zia, Quoc-Huy Tran, Xiang Yu, Gregory D. Hager, Manmohan Chandraker
Amodal Detection of 3D Objects: Inferring 3D Bounding Boxes From 2D Ones in RGB-Depth Images
Zhuo Deng, Longin Jan Latecki

## Analyzing Humans in Images

Transition Forests: Learning Discriminative Temporal Transitions for Action Recognition and Detection
Guillermo Garcia-Hernando, Tae-Kyun Kim
Scene Flow to Action Map: A New Representation for RGB-D Based Action Recognition With Convolutional Neural Networks
Pichao Wang, Wanqing Li, Zhimin Gao, Yuyao Zhang, Chang Tang, Philip Ogunbona
Detecting Masked Faces in the Wild With LLE-CNNs
Shiming Ge, Jia Li, Qiting Ye, Zhao Luo
A Domain Based Approach to Social Relation Recognition
Qianru Sun, Bernt Schiele, Mario Fritz
Spatio-Temporal Naive-Bayes Nearest-Neighbor (ST-NBNN) for Skeleton-Based Action Recognition
Junwu Weng, Chaoqun Weng, Junsong Yuan
Personalizing Gesture Recognition Using Hierarchical Bayesian Neural Networks
Ajjen Joshi, Soumya Ghosh, Margrit Betke, Stan Sclaroff, Hanspeter Pfister

## Applications

Real-Time 3D Model Tracking in Color and Depth on a Single CPU Core
Wadim Kehl, Federico Tombari, Slobodan Ilic, Nassir Navab
Multi-Scale FCN With Cascaded Instance Aware Segmentation for Arbitrary Oriented Word Spotting in the Wild
Dafang He, Xiao Yang, Chen Liang, Zihan Zhou, Alexander G. Ororbi II, Daniel Kifer, C. Lee Giles
Viraliency: Pooling Local Virality
Xavier Alameda-Pineda, Andrea Pilzer, Dan Xu, Nicu Sebe, Elisa Ricci

## Biomedical Image/Video Analysis

A Non-Local Low-Rank Framework for Ultrasound Speckle Reduction
Lei Zhu, Chi-Wing Fu, Michael S. Brown, Pheng-Ann Heng

## Image Motion & Tracking

Video Acceleration Magnification
Yichao Zhang, Silvia L. Pintea, Jan C. van Gemert
Superpixel-Based Tracking-By-Segmentation Using Markov Chains
Donghun Yeo, Jeany Son, Bohyung Han, Joon Hee Han
BranchOut: Regularization for Online Ensemble Tracking With Convolutional Neural Networks
Bohyung Han, Jack Sim, Hartwig Adam
Learning Motion Patterns in Videos
Pavel Tokmakov, Karteek Alahari, Cordelia Schmid

## Low- & Mid-Level Vision

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

## Machine Learning

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
Learning Dynamic Guidance for Depth Image Enhancement (PDF)
Shuhang Gu, Wangmeng Zuo, Shi Guo, Yunjin Chen, Chongyu Chen, Lei Zhang
A-Lamp: Adaptive Layout-Aware Multi-Patch Deep Convolutional Neural Network for Photo Aesthetic Assessment (PDF)
Shuang Ma, Jing Liu, Chang Wen Chen
Teaching Compositionality to CNNs
Austin Stone, Huayan Wang, Michael Stark, Yi Liu, D. Scott Phoenix, Dileep George
Using Ranking-CNN for Age Estimation
Shixing Chen, Caojin Zhang, Ming Dong, Jialiang Le, Mike Rao
Accurate Single Stage Detector Using Recurrent Rolling Convolution
Jimmy Ren, Xiaohao Chen, Jianbo Liu, Wenxiu Sun, Jiahao Pang, Qiong Yan, Yu-Wing Tai, Li Xu
A Compact DNN: Approaching GoogLeNet-Level Accuracy of Classification and Domain Adaptation
Chunpeng Wu, Wei Wen, Tariq Afzal, Yongmei Zhang, Yiran Chen, Hai   [PDF](helen) Li
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

## Object Recognition & Scene Understanding

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

## Video Analytics

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

# Object Recognition & Scene Understanding - Computer Vision & Language

## Spotlight 1-2A

#### Discriminative Bimodal Networks for Visual Localization and Detection With Natural Language Queries  [PDF](https://arxiv.org/abs/1704.03944)

Yuting Zhang, Luyao Yuan, Yijie Guo, Zhiyuan He, I-An Huang, Honglak Lee

_**Abstract**_: &emsp;_[Associating image regions with text queries has been recently explored as a new way to bridge visual and linguistic representations. A few pioneering approaches have been proposed based on recurrent neural language models trained generatively (e.g., generating captions), but achieving somewhat limited localization accuracy. To better address natural-language-based visual entity localization, we propose a discriminative approach. We formulate a discriminative bimodal neural network (DBNet), which can be trained by a classifier with extensive use of negative samples. Our training objective encourages better localization on single images, incorporates text phrases in a broad range, and properly pairs image regions with text phrases into positive and negative examples. Experiments on the Visual Genome dataset demonstrate the proposed DBNet significantly outperforms previous state-of-the-art methods both for localization on single images and for detection on multiple images. We we also establish an evaluation protocol for natural-language visual detection. ]_

_**Comment**_: &emsp;_< natural-language-based visual entity localization, bridge visual and linguistic representations>_

#### Automatic Understanding of Image and Video Advertisements  [PDF](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwi928mth9TUAhUBe7wKHUPKDLEQFggnMAE&url=http%3A%2F%2Fpeople.cs.pitt.edu%2F~kovashka%2Fhussain_zhang_kovashka_ads_cvpr2017.pdf&usg=AFQjCNGSbgpDjn-hjeOgRkvkY1A2EHowsQ)

Zaeem Hussain, Mingda Zhang, Xiaozhong Zhang, Keren Ye, Christopher Thomas, Zuha Agha, Nathan Ong, Adriana Kovashka

_**Abstract**_: &emsp;_[There is more to images than their objective physical content: for example, advertisements are created to persuade a viewer to take a certain action. We propose the novel problem of automatic advertisement understanding. To enable research on this problem, we create two datasets: an image dataset of 64,832 image ads, and a video dataset of 3,477 ads. Our data contains rich annotations encompassing the topic and sentiment of the ads, questions and answers describing what actions the viewer is prompted to take and the reasoning that the ad presents to persuade the viewer ("What should I do according to this ad, and why should I do it?"), and symbolic references ads make (e.g. a dove symbolizes peace). We also analyze the most common persuasive strategies ads use, and the capabilities that computer vision systems should have to understand these strategies. We present baseline classification results for several prediction tasks, including automatically answering questions about the messages of the ads.]_

_**Comment**_: &emsp;_< problem of automatic advertisement understanding; dataset>_

#### *Deep Sketch Hashing: Fast Free-Hand Sketch-Based Image Retrieval  [PDF](https://arxiv.org/abs/1703.05605)

Li Liu, Fumin Shen, Yuming Shen, Xianglong Liu, Ling Shao

_**Abstract**_: &emsp;_[ Free-hand sketch-based image retrieval (SBIR) is a specific cross-view retrieval task, in which queries are abstract and ambiguous sketches while the retrieval database is formed with natural images. Work in this area mainly focuses on extracting representative and shared features for sketches and natural images. However, these can neither cope well with the geometric distortion between sketches and images nor be feasible for large-scale SBIR due to the heavy continuous-valued distance computation. In this paper, we speed up SBIR by introducing a novel binary coding method, named \textbf{Deep Sketch Hashing} (DSH), where a semi-heterogeneous deep architecture is proposed and incorporated into an end-to-end binary coding framework. Specifically, three convolutional neural networks are utilized to encode free-hand sketches, natural images and, especially, the auxiliary sketch-tokens which are adopted as bridges to mitigate the sketch-image geometric distortion. The learned DSH codes can effectively capture the cross-view similarities as well as the intrinsic semantic correlations between different categories. To the best of our knowledge, DSH is the first hashing work specifically designed for category-level SBIR with an end-to-end deep architecture. The proposed DSH is comprehensively evaluated on two large-scale datasets of TU-Berlin Extension and Sketchy, and the experiments consistently show DSH's superior SBIR accuracies over several state-of-the-art methods, while achieving significantly reduced retrieval time and memory footprint. ]_

_**Comment**_: &emsp;_< sketch-image,Free-hand sketch-based image retrieval by binary coding using DL>_

#### *Discover and Learn New Objects From Documentaries  [PDF](http://personal.ie.cuhk.edu.hk/~ccloy/files/cvpr_2017_discover.pdf)

Kai Chen, Hang Song, Chen Change Loy, Dahua Lin

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< learning object detectors from documentary films in a weakly supervised manner, using a joint probabilistic framework>_

#### Spatial-Semantic Image Search by Visual Feature Synthesis  [PDF](http://web.cecs.pdx.edu/~fliu/papers/cvpr2017-search.pdf)

Long Mai, Hailin Jin, Zhe Lin, Chen Fang, Jonathan Brandt, Feng Liu

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< Spatial-Semantic Image Search, search with sematic & spatial info at the same time>_

#### *Fully-Adaptive Feature Sharing in Multi-Task Networks With Applications in Person Attribute Classification  [PDF](https://arxiv.org/abs/1611.05377)

Yongxi Lu, Abhishek Kumar, Shuangfei Zhai, Yu Cheng, Tara Javidi, Rogerio Feris

_**Abstract**_: &emsp;_[Multi-task learning aims to improve generalization performance of multiple prediction tasks by appropriately sharing relevant information across them. In the context of deep neural networks, this idea is often realized by hand-designed network architectures with layers that are shared across tasks and branches that encode task-specific features. However, the space of possible multi-task deep architectures is combinatorially large and often the final architecture is arrived at by manual exploration of this space subject to designer's bias, which can be both error-prone and tedious. In this work, we propose a principled approach for designing compact multi-task deep learning architectures. Our approach starts with a thin network and dynamically widens it in a greedy manner during training using a novel criterion that promotes grouping of similar tasks together. Our Extensive evaluation on person attributes classification tasks involving facial and clothing attributes suggests that the models produced by the proposed method are fast, compact and can closely match or exceed the state-of-the-art accuracy from strong baselines by much more expensive models. ]_

_**Comment**_: &emsp;_< Multi-task learning in DL with layers shared across tasks,adaptive widening of net>_

#### Semantic Compositional Networks for Visual Captioning  [PDF](https://arxiv.org/abs/1611.08002)

Zhe Gan, Chuang Gan, Xiaodong He, Yunchen Pu, Kenneth Tran, Jianfeng Gao, Lawrence Carin, Li Deng

_**Abstract**_: &emsp;_[A Semantic Compositional Network (SCN) is developed for image captioning, in which semantic concepts (i.e., tags) are detected from the image, and the probability of each tag is used to compose the parameters in a long short-term memory (LSTM) network. The SCN extends each weight matrix of the LSTM to an ensemble of tag-dependent weight matrices. The degree to which each member of the ensemble is used to generate an image caption is tied to the image-dependent probability of the corresponding tag. In addition to captioning images, we also extend the SCN to generate captions for video clips. We qualitatively analyze semantic composition in SCNs, and quantitatively evaluate the algorithm on three benchmark datasets: COCO, Flickr30k, and Youtube2Text. Experimental results show that the proposed method significantly outperforms prior state-of-the-art approaches, across multiple evaluation metrics. ]_

_**Comment**_: &emsp;_< visual captioning , LSTM, ensemble of tag-dependent weight matrices>_

#### ***Training Object Class Detectors With Click Supervision  [PDF](https://arxiv.org/abs/1704.06189)

Dim P. Papadopoulos, Jasper R. R. Uijlings, Frank Keller, Vittorio Ferrari

_**Abstract**_: &emsp;_[Training object class detectors typically requires a large set of images with objects annotated by bounding boxes. However, manually drawing bounding boxes is very time consuming. In this paper we greatly reduce annotation time by proposing center-click annotations: we ask annotators to click on the center of an imaginary bounding box which tightly encloses the object instance. We then incorporate these clicks into existing Multiple Instance Learning techniques for weakly supervised object localization, to jointly localize object bounding boxes over all training images. Extensive experiments on PASCAL VOC 2007 and MS COCO show that: (1) our scheme delivers high-quality detectors, performing substantially better than those produced by weakly supervised techniques, with a modest extra annotation effort; (2) these detectors in fact perform in a range close to those trained from manually drawn bounding boxes; (3) as the center-click task is very fast, our scheme reduces total annotation time by 9x to 18x. ]_

_**Comment**_: &emsp;_< not bounding boxes but center clicks, then incorporate these clicks into existing Multiple Instance Learning techniques >_

## Oral 1-2A

#### *Deep Reinforcement Learning-Based Image Captioning With Embedding Reward  [PDF](https://arxiv.org/abs/1704.03899)

Zhou Ren, Xiaoyu Wang, Ning Zhang, Xutao Lv, Li-Jia Li

_**Abstract**_: &emsp;_[Image captioning is a challenging problem owing to the complexity in understanding the image content and diverse ways of describing it in natural language. Recent advances in deep neural networks have substantially improved the performance of this task. Most state-of-the-art approaches follow an encoder-decoder framework, which generates captions using a sequential recurrent prediction model. However, in this paper, we introduce a novel decision-making framework for image captioning. We utilize a "policy network" and a "value network" to collaboratively generate captions. The policy network serves as a local guidance by providing the confidence of predicting the next word according to the current state. Additionally, the value network serves as a global and lookahead guidance by evaluating all possible extensions of the current state. In essence, it adjusts the goal of predicting the correct words towards the goal of generating captions similar to the ground truth captions. We train both networks using an actor-critic reinforcement learning model, with a novel reward defined by visual-semantic embedding. Extensive experiments and analyses on the Microsoft COCO dataset show that the proposed framework outperforms state-of-the-art approaches across different evaluation metrics. ]_

_**Comment**_: &emsp;_< image caption, utilize a "policy network"(local guidance) and a "value network"(global guidance) to collaboratively generate captions.>_

#### From Red Wine to Red Tomato: Composition With Context  [PDF](http://www.cs.cmu.edu/~imisra/data/composing_cvpr17.pdf)

Ishan Misra, Abhinav Gupta, Martial Hebert

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< context ;present a simple method that respects contextuality in order to compose classifiers of known visual concepts; it composes while respecting context >_

#### *Captioning Images With Diverse Objects  [PDF](https://arxiv.org/abs/1606.07770)

Subhashini Venugopalan, Lisa Anne Hendricks, Marcus Rohrbach, Raymond Mooney, Trevor Darrell, Kate Saenko

_**Abstract**_: &emsp;_[Recent captioning models are limited in their ability to scale and describe concepts unseen in paired image-text corpora. We propose the Novel Object Captioner (NOC), a deep visual semantic captioning model that can describe a large number of object categories not present in existing image-caption datasets. Our model takes advantage of external sources -- labeled images from object recognition datasets, and semantic knowledge extracted from unannotated text. We propose minimizing a joint objective which can learn from these diverse data sources and leverage distributional semantic embeddings, enabling the model to generalize and describe novel objects outside of image-caption datasets. We demonstrate that our model exploits semantic information to generate captions for hundreds of object categories in the ImageNet object recognition dataset that are not observed in MSCOCO image-caption training data, as well as many categories that are observed very rarely. Both automatic evaluations and human judgements show that our model considerably outperforms prior work in being able to describe many more categories of objects. ]_

_**Comment**_: &emsp;_< can describe a large number of object categories not existing, such as in ImageNet; takes advantage of labeled images from object recognition datasets, and semantic knowledge extracted from unannotated text; minimizing a joint objective>_

#### Self-Critical Sequence Training for Image Captioning  [PDF](https://arxiv.org/abs/1612.00563)

Steven J. Rennie, Etienne Marcheret, Youssef Mroueh, Jerret Ross, Vaibhava Goel

_**Abstract**_: &emsp;_[Recently it has been shown that policy-gradient methods for reinforcement learning can be utilized to train deep end-to-end systems directly on non-differentiable metrics for the task at hand. In this paper we consider the problem of optimizing image captioning systems using reinforcement learning, and show that by carefully optimizing our systems using the test metrics of the MSCOCO task, significant gains in performance can be realized. Our systems are built using a new optimization approach that we call self-critical sequence training (SCST). SCST is a form of the popular REINFORCE algorithm that, rather than estimating a "baseline" to normalize the rewards and reduce variance, utilizes the output of its own test-time inference algorithm to normalize the rewards it experiences. Using this approach, estimating the reward signal (as actor-critic methods must do) and estimating normalization (as REINFORCE algorithms typically do) is avoided, while at the same time harmonizing the model with respect to its test-time inference procedure. Empirically we find that directly optimizing the CIDEr metric with SCST and greedy decoding at test-time is highly effective. Our results on the MSCOCO evaluation sever establish a new state-of-the-art on the task, improving the best result in terms of CIDEr from 104.9 to 112.3. ]_

_**Comment**_: &emsp;_<  image captioning, using reinforcement learning, a new optimization approach that we call self-critical sequence training: utilizes the output of its own test-time inference algorithm to normalize the rewards it experiences>_

# Analyzing Humans 1

## Spotlight 1-2B

#### Crossing Nets: Combining GANs and VAEs With a Shared Latent Space for Hand Pose Estimation

Chengde Wan, Thomas Probst, Luc Van Gool, Angela Yao

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Predicting Behaviors of Basketball Players From First Person Videos

Shan Su, Jung Pyo Hong, Jianbo Shi, Hyun Soo Park

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### LCR-Net: Localization-Classification-Regression for Human Pose

Grégory Rogez, Philippe Weinzaepfel, Cordelia Schmid

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### *Learning Residual Images for Face Attribute Manipulation  [PDF](https://arxiv.org/abs/1612.05363)

Wei Shen, Rujie Liu

_**Abstract**_: &emsp;_[Face attributes are interesting due to their detailed description of human faces. Unlike prior researches working on attribute prediction, we address an inverse and more challenging problem called face attribute manipulation which aims at modifying a face image according to a given attribute value. Instead of manipulating the whole image, we propose to learn the corresponding residual image defined as the difference between images before and after the manipulation. In this way, the manipulation can be operated efficiently with modest pixel modification. The framework of our approach is based on the Generative Adversarial Network. It consists of two image transformation networks and a discriminative network. The transformation networks are responsible for the attribute manipulation and its dual operation and the discriminative network is used to distinguish the generated images from real images. We also apply dual learning to allow transformation networks to learn from each other. Experiments show that residual images can be effectively learned and used for attribute manipulations. The generated images remain most of the details in attribute-irrelevant areas. ]_

_**Comment**_: &emsp;_< modifying a face image according to a given attribute value, GAN>_

#### *Seeing What Is Not There: Learning Context to Determine Where Objects Are Missing  [PDF](https://arxiv.org/abs/1702.07971)

Jin Sun, David W. Jacobs

_**Abstract**_: &emsp;_[Most of computer vision focuses on what is in an image. We propose to train a standalone object-centric context representation to perform the opposite task: seeing what is not there. Given an image, our context model can predict where objects should exist, even when no object instances are present. Combined with object detection results, we can perform a novel vision task: finding where objects are missing in an image. Our model is based on a convolutional neural network structure. With a specially designed training strategy, the model learns to ignore objects and focus on context only. It is fully convolutional thus highly efficient. Experiments show the effectiveness of the proposed approach in one important accessibility task: finding city street regions where curb ramps are missing, which could help millions of people with mobility disabilities. ]_

_**Comment**_: &emsp;_< special traning strategy , ignore objects and focus on context only>_

#### Deep Learning on Lie Groups for Skeleton-Based Action Recognition

Zhiwu Huang, Chengde Wan, Thomas Probst, Luc Van Gool

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Harvesting Multiple Views for Marker-Less 3D Human Pose Annotations

Georgios Pavlakos, Xiaowei Zhou, Konstantinos G. Derpanis, Kostas Daniilidis

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Coarse-To-Fine Volumetric Prediction for Single-Image 3D Human Pose

Georgios Pavlakos, Xiaowei Zhou, Konstantinos G. Derpanis, Kostas Daniilidis

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

## Oral 1-2B

#### Weakly Supervised Action Learning With RNN Based Fine-To-Coarse Modeling  [PDF](https://arxiv.org/abs/1703.08132)

Alexander Richard, Hilde Kuehne, Juergen Gall

_**Abstract**_: &emsp;_[We present an approach for weakly supervised learning of human actions. Given a set of videos and an ordered list of the occurring actions, the goal is to infer start and end frames of the related action classes within the video and to train the respective action classifiers without any need for hand labeled frame boundaries. To address this task, we propose a combination of a discriminative representation of subactions, modeled by a recurrent neural network, and a coarse probabilistic model to allow for a temporal alignment and inference over long sequences. While this system alone already generates good results, we show that the performance can be further improved by approximating the number of subactions to the characteristics of the different action classes. To this end, we adapt the number of subaction classes by iterating realignment and reestimation during training. The proposed system is evaluated on two benchmark datasets, the Breakfast and the Hollywood extended dataset, showing a competitive performance on various weak learning tasks such as temporal action segmentation and action alignment. ]_

_**Comment**_: &emsp;_< >_

#### *Disentangled Representation Learning GAN for Pose-Invariant Face Recognition  [PDF](http://cvlab.cse.msu.edu/pdfs/Tran_Yin_Liu_CVPR2017.pdf)

Luan Tran, Xi Yin, Xiaoming Liu

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### ArtTrack: Articulated Multi-Person Tracking in the Wild  [PDF](https://arxiv.org/abs/1612.01465)

Eldar Insafutdinov, Mykhaylo Andriluka, Leonid Pishchulin, Siyu Tang, Evgeny Levinkov, Bjoern Andres, Bernt Schiele

_**Abstract**_: &emsp;_[In this paper we propose an approach for articulated tracking of multiple people in unconstrained videos. Our starting point is a model that resembles existing architectures for single-frame pose estimation but is substantially faster. We achieve this in two ways: (1) by simplifying and sparsifying the body-part relationship graph and leveraging recent methods for faster inference, and (2) by offloading a substantial share of computation onto a feed-forward convolutional architecture that is able to detect and associate body joints of the same person even in clutter. We use this model to generate proposals for body joint locations and formulate articulated tracking as spatio-temporal grouping of such proposals. This allows to jointly solve the association problem for all people in the scene by propagating evidence from strong detections through time and enforcing constraints that each proposal can be assigned to one person only. We report results on a public MPII Human Pose benchmark and on a new MPII Video Pose dataset of image sequences with multiple people. We demonstrate that our model achieves state-of-the-art results while using only a fraction of time and is able to leverage temporal information to improve state-of-the-art for crowded scenes. ]_

_**Comment**_: &emsp;_< >_

#### Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields (PDF, code)  [PDF](https://arxiv.org/pdf/1611.08050.pdf)  [PDF](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh


_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

# Image Motion & Tracking; Video Analysis

## Spotlight 1-2C

#### *Template Matching With Deformable Diversity Similarity（https://arxiv.org/abs/1612.02190）

Itamar Talmi, Roey Mechrez, Lihi Zelnik-Manor

_**Abstract**_: &emsp;_[We propose a novel measure for template matching named Deformable Diversity Similarity -- based on the diversity of feature matches between a target image window and the template. We rely on both local appearance and geometric information that jointly lead to a powerful approach for matching. Our key contribution is a similarity measure, that is robust to complex deformations, significant background clutter, and occlusions. Empirical evaluation on the most up-to-date benchmark shows that our method outperforms the current state-of-the-art in its detection accuracy while improving computational complexity. ]_

_**Comment**_: &emsp;_< >_

#### **Beyond Triplet Loss: A Deep Quadruplet Network for Person Re-Identification  [PDF](https://arxiv.org/abs/1704.01719)

Weihua Chen, Xiaotang Chen, Jianguo Zhang, Kaiqi Huang

_**Abstract**_: &emsp;_[Person re-identification (ReID) is an important task in wide area video surveillance which focuses on identifying people across different cameras. Recently, deep learning networks with a triplet loss become a common framework for person ReID. However, the triplet loss pays main attentions on obtaining correct orders on the training set. It still suffers from a weaker generalization capability from the training set to the testing set, thus resulting in inferior performance. In this paper, we design a quadruplet loss, which can lead to the model output with a larger inter-class variation and a smaller intra-class variation compared to the triplet loss. As a result, our model has a better generalization ability and can achieve a higher performance on the testing set. In particular, a quadruplet deep network using a margin-based online hard negative mining is proposed based on the quadruplet loss for the person ReID. In extensive experiments, the proposed network outperforms most of the state-of-the-art algorithms on representative datasets which clearly demonstrates the effectiveness of our proposed method. ]_

_**Comment**_: &emsp;_< a triplet loss to a quadruplet loss>_

#### Agent-Centric Risk Assessment: Accident Anticipation and Risky Region Localization  [PDF](https://arxiv.org/abs/1705.06560)

Kuo-Hao Zeng, Shih-Han Chou, Fu-Hsiang Chan, Juan Carlos Niebles, Min Sun

_**Abstract**_: &emsp;_[For survival, a living agent must have the ability to assess risk (1) by temporally anticipating accidents before they occur, and (2) by spatially localizing risky regions in the environment to move away from threats. In this paper, we take an agent-centric approach to study the accident anticipation and risky region localization tasks. We propose a novel soft-attention Recurrent Neural Network (RNN) which explicitly models both spatial and appearance-wise non-linear interaction between the agent triggering the event and another agent or static-region involved. In order to test our proposed method, we introduce the Epic Fail (EF) dataset consisting of 3000 viral videos capturing various accidents. In the experiments, we evaluate the risk assessment accuracy both in the temporal domain (accident anticipation) and spatial domain (risky region localization) on our EF dataset and the Street Accident (SA) dataset. Our method consistently outperforms other baselines on both datasets. ]_

_**Comment**_: &emsp;_< >_

#### Bidirectional Multirate Reconstruction for Temporal Modeling in Videos

Linchao Zhu, Zhongwen Xu, Yi Yang

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Action-Decision Networks for Visual Tracking With Deep Reinforcement Learning  [PDF](https://sites.google.com/view/cvpr2017-adnet)

Sangdoo Yun, Jongwon Choi, Youngjoon Yoo, Kimin Yun, Jin Young Choi

_**Abstract**_: &emsp;_[This paper proposes a novel tracker which is controlled by sequentially pursuing actions learned by deep reinforcement learning. In contrast to the existing trackers using deep networks, the proposed tracker is designed to achieve a light computation as well as satisfactory tracking accuracy in both location and scale. The deep network to control actions is pre-trained using various training sequences and fine-tuned during tracking for online adaptation to target and background changes. The pre-training is done by utilizing deep reinforcement learning as well as supervised learning. The use of reinforcement learning enables even partially labeled data to be successfully utilized for semi-supervised learning. Through evaluation of the OTB dataset, the proposed tracker is validated to achieve a competitive performance that is three times faster than state-of-the-art, deep network–based trackers. The fast version of the proposed method, which operates in real-time on GPU, outperforms the state-of-the-art real-time trackers.]_

_**Comment**_: &emsp;_< >_

#### TGIF-QA: Toward Spatio-Temporal Reasoning in Visual Question Answering  [PDF](https://arxiv.org/abs/1704.04497)

Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, Gunhee Kim

_**Abstract**_: &emsp;_[Vision and language understanding has emerged as a subject undergoing intense study in Artificial Intelligence. Among many tasks in this line of research, visual question answering (VQA) has been one of the most successful ones, where the goal is to learn a model that understands visual content at region-level details and finds their associations with pairs of questions and answers in the natural language form. Despite the rapid progress in the past few years, most existing work in VQA have focused primarily on images. In this paper, we focus on extending VQA to the video domain and contribute to the literature in three important ways. First, we propose three new tasks designed specifically for video VQA, which require spatio-temporal reasoning from videos to answer questions correctly. Next, we introduce a new large-scale dataset for video VQA named TGIF-QA that extends existing VQA work with our new tasks. Finally, we propose a dual-LSTM based approach with both spatial and temporal attention, and show its effectiveness over conventional VQA techniques through empirical evaluations. ]_

_**Comment**_: &emsp;_< >_

#### Making 360° Video Watchable in 2D: Learning Videography for Click Free Viewing  [PDF](https://arxiv.org/abs/1703.00495)

Yu-Chuan Su, Kristen Grauman

_**Abstract**_: &emsp;_[360 video requires human viewers to actively control "where" to look while watching the video. Although it provides a more immersive experience of the visual content, it also introduces additional burden for viewers; awkward interfaces to navigate the video lead to suboptimal viewing experiences. Virtual cinematography is an appealing direction to remedy these problems, but conventional methods are limited to virtual environments or rely on hand-crafted heuristics. We propose a new algorithm for virtual cinematography that automatically controls a virtual camera within a 360 video. Compared to the state of the art, our algorithm allows more general camera control, avoids redundant outputs, and extracts its output videos substantially more efficiently. Experimental results on over 7 hours of real "in the wild" video show that our generalized camera control is crucial for viewing 360 video, while the proposed efficient algorithm is essential for making the generalized control computationally tractable. ]_

_**Comment**_: &emsp;_< >_

#### Unsupervised Adaptive Re-Identification in Open World Dynamic Camera Networks  [PDF](https://arxiv.org/abs/1706.03112)

Rameswar Panda, Amran Bhuiyan, Vittorio Murino, Amit K. Roy-Chowdhury

_**Abstract**_: &emsp;_[Person re-identification is an open and challenging problem in computer vision. Existing approaches have concentrated on either designing the best feature representation or learning optimal matching metrics in a static setting where the number of cameras are fixed in a network. Most approaches have neglected the dynamic and open world nature of the re-identification problem, where a new camera may be temporarily inserted into an existing system to get additional information. To address such a novel and very practical problem, we propose an unsupervised adaptation scheme for re-identification models in a dynamic camera network. First, we formulate a domain perceptive re-identification method based on geodesic flow kernel that can effectively find the best source camera (already installed) to adapt with a newly introduced target camera, without requiring a very expensive training phase. Second, we introduce a transitive inference algorithm for re-identification that can exploit the information from best source camera to improve the accuracy across other camera pairs in a network of multiple cameras. Extensive experiments on four benchmark datasets demonstrate that the proposed approach significantly outperforms the state-of-the-art unsupervised learning based alternatives whilst being extremely efficient to compute.]_

_**Comment**_: &emsp;_<  based on geodesic flow kernel; cameras are not fixed>_


## Oral 1-2C

#### Context-Aware Correlation Filter Tracking  [PDF](https://goo.gl/5rDpff)

Matthias Mueller, Neil Smith, Bernard Ghanem

_**Abstract**_: &emsp;_[​Correlation filter (CF) based trackers have recently gained a lot of popularity due to their impressive performance on benchmark datasets, while maintaining high frame rates. A significant amount of recent research focuses on the incorporation of stronger features for a richer representation of the tracking target. However, this only helps to discriminate the target from background within a small neighborhood. In this paper, we present a framework that allows the explicit incorporation of global context within CF trackers. We reformulate the original optimization problem and provide a closed form solution for single and multi-dimensional features in the primal and dual domain. Extensive experiments demonstrate that this framework significantly improves the performance of many CF trackers with only a modest impact on frame rate.​]_

_**Comment**_: &emsp;_< tracking>_

 #### Deep 360 Pilot: Learning a Deep Agent for Piloting Through 360° Sports Videos

Hou-Ning Hu, Yen-Chen Lin, Ming-Yu Liu, Hsien-Tzu Cheng, Yung-Ju Chang, Min Sun

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Slow Flow: Exploiting High-Speed Cameras for Accurate and Diverse Optical Flow Reference Data

 Joel Janai, Fatma Güney, Jonas Wulff, Michael J. Black, Andreas Geiger

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### *CDC: Convolutional-De-Convolutional Networks for Precise Temporal Action Localization in Untrimmed Videos  [PDF](https://arxiv.org/abs/1703.01515)

Zheng Shou, Jonathan Chan, Alireza Zareian, Kazuyuki Miyazawa, Shih-Fu Chang

_**Abstract**_: &emsp;_[Temporal action localization is an important yet challenging problem. Given a long, untrimmed video consisting of multiple action instances and complex background contents, we need not only to recognize their action categories, but also to localize the start time and end time of each instance. Many state-of-the-art systems use segment-level classifiers to select and rank proposal segments of pre-determined boundaries. However, a desirable model should move beyond segment-level and make dense predictions at a fine granularity in time to determine precise temporal boundaries. To this end, we design a novel Convolutional-De-Convolutional (CDC) network that places CDC filters on top of 3D ConvNets, which have been shown to be effective for abstracting action semantics but reduce the temporal length of the input data. The proposed CDC filter performs the required temporal upsampling and spatial downsampling operations simultaneously to predict actions at the frame-level granularity. It is unique in jointly modeling action semantics in space-time and fine-grained temporal dynamics. We train the CDC network in an end-to-end manner efficiently. Our model not only achieves superior performance in detecting actions in every frame, but also significantly boosts the precision of localizing temporal boundaries. Finally, the CDC network demonstrates a very high efficiency with the ability to process 500 frames per second on a single GPU server. We will update the camera-ready version and publish the source codes online soon. ]_

_**Comment**_: &emsp;_< Convolutional-De-Convolutional (CDC) filters on top of 3D ConvNets,>_

# Poster 1-2

## 3D Computer Vision

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

## Analyzing Humans in Images

#### Finding Tiny Faces
Peiyun Hu, Deva Ramanan

_**Abstract**_:Though tremendous strides have been made in object recognition, one of the remaining open challenges is detecting small objects. We explore three aspects of the problem in the context of finding small faces: the role of scale invariance, image resolution, and contextual reasoning. While most recognition approaches aim to be scale-invariant, the cues for recognizing a 3px tall face are fundamentally different than those for recognizing a 300px tall face. We take a different approach and train separate detectors for different scales. To maintain efficiency, detectors are trained in a multi-task fashion: they make use of features extracted from multiple layers of single (deep) feature hierarchy. While training detectors for large objects is straightforward, the crucial challenge remains training detectors for small objects. We show that context is crucial, and define templates that make use of massively-large receptive fields (where 99% of the template extends beyond the object of interest). Finally, we explore the role of scale in pre-trained deep networks, providing ways to extrapolate networks tuned for limited scales to rather extreme ranges. We demonstrate state-of-the-art results on massively-benchmarked face datasets (FDDB and WIDER FACE). In particular, when compared to prior art on WIDER FACE, our results reduce error by a factor of 2 (our models produce an AP of 82% while prior art ranges from 29-64%).

_**Comment**_: &emsp;_< face detection under crowd scenes，image pyramid，feature fusion>_

#### Dynamic Facial Analysis: From Bayesian Filtering to Recurrent Neural Network
Jinwei Gu, Xiaodong Yang, Shalini De Mello, Jan Kautz
#### Deep Temporal Linear Encoding Networks
Ali Diba, Vivek Sharma, Luc Van Gool
#### Joint Registration and Representation Learning for Unconstrained Face Identification
Munawar Hayat, Salman H. Khan, Naoufel Werghi, Roland Goecke
#### 3D Human Pose Estimation From a Single Image via Distance Matrix Regression
Francesc Moreno-Noguer
#### One-Shot Metric Learning for Person Re-Identification
Slawomir BÄ…k, Peter Carr
#### Generalized Rank Pooling for Activity Recognition
Anoop Cherian, Basura Fernando, Mehrtash Harandi, Stephen Gould
#### Deep Representation Learning for Human Motion Prediction and Classification
Judith BÃ¼tepage, Michael J. Black, Danica Kragic, Hedvig KjellstrÃ¶m
#### Interspecies Knowledge Transfer for Facial Keypoint Detection
Maheen Rashid, Xiuye Gu, Yong Jae Lee
#### Recurrent Convolutional Neural Networks for Continuous Sign Language Recognition by Staged Optimization
Runpeng Cui, Hu Liu, Changshui Zhang

## Applications

Modeling Sub-Event Dynamics in First-Person Action Recognition
Hasan F. M. Zaki, Faisal Shafait, Ajmal Mian

## Computational Photography

Turning an Urban Scene Video Into a Cinemagraph
Hang Yan, Yebin Liu, Yasutaka Furukawa
Light Field Reconstruction Using Deep Convolutional Network on EPI
Gaochang Wu, Mandan Zhao, Liangyong Wang, Qionghai Dai, Tianyou Chai, Yebin Liu

## Image Motion & Tracking

FlowNet 2.0: Evolution of Optical Flow Estimation With Deep Networks
Eddy Ilg, Nikolaus Mayer, Tonmoy Saikia, Margret Keuper, Alexey Dosovitskiy, Thomas Brox

## Low- & Mid-Level Vision

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

## Machine Learning

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

## Object Recognition & Scene Understanding

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

## Video Analytics

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

# Machine Learning 2

## Spotlight 2-1A

#### Dual Attention Networks for Multimodal Reasoning and Matching(  [PDF](https://arxiv.org/abs/1611.00471))

Hyeonseob Nam, Jung-Woo Ha, Jeonghee Kim

_**Abstract**_: &emsp;_[We propose Dual Attention Networks (DANs) which jointly leverage visual and textual attention mechanisms to capture fine-grained interplay between vision and language. DANs attend to specific regions in images and words in text through multiple steps and gather essential information from both modalities. Based on this framework, we introduce two types of DANs for multimodal reasoning and matching, respectively. The reasoning model allows visual and textual attentions to steer each other during collaborative inference, which is useful for tasks such as Visual Question Answering (VQA). In addition, the matching model exploits the two attention mechanisms to estimate the similarity between images and sentences by focusing on their shared semantics. Our extensive experiments validate the effectiveness of DANs in combining vision and language, achieving the state-of-the-art performance on public benchmarks for VQA and image-text matching. ]_

_**Comment**_: &emsp;_< visual and textual attention ,>_

#### DESIRE: Distant Future Prediction in Dynamic Scenes With Interacting Agents  [PDF](https://arxiv.org/abs/1704.04394)

Namhoon Lee, Wongun Choi, Paul Vernaza, Christopher B. Choy, Philip H. S. Torr, Manmohan Chandraker

_**Abstract**_: &emsp;_[We introduce a Deep Stochastic IOC RNN Encoderdecoder framework, DESIRE, for the task of future predictions of multiple interacting agents in dynamic scenes. DESIRE effectively predicts future locations of objects in multiple scenes by 1) accounting for the multi-modal nature of the future prediction (i.e., given the same context, future may vary), 2) foreseeing the potential future outcomes and make a strategic prediction based on that, and 3) reasoning not only from the past motion history, but also from the scene context as well as the interactions among the agents. DESIRE achieves these in a single end-to-end trainable neural network model, while being computationally efficient. The model first obtains a diverse set of hypothetical future prediction samples employing a conditional variational autoencoder, which are ranked and refined by the following RNN scoring-regression module. Samples are scored by accounting for accumulated future rewards, which enables better long-term strategic decisions similar to IOC frameworks. An RNN scene context fusion module jointly captures past motion histories, the semantic scene context and interactions among multiple agents. A feedback mechanism iterates over the ranking and refinement to further boost the prediction accuracy. We evaluate our model on two publicly available datasets: KITTI and Stanford Drone Dataset. Our experiments show that the proposed model significantly improves the prediction accuracy compared to other baseline methods. ]_

_**Comment**_: &emsp;_< >_

#### *Interpretable Structure-Evolving LSTM  [PDF](https://arxiv.org/abs/1703.03055)

Xiaodan Liang, Liang Lin, Xiaohui Shen, Jiashi Feng, Shuicheng Yan, Eric P. Xing

_**Abstract**_: &emsp;_[This paper develops a general framework for learning interpretable data representation via Long Short-Term Memory (LSTM) recurrent neural networks over hierarchal graph structures. Instead of learning LSTM models over the pre-fixed structures, we propose to further learn the intermediate interpretable multi-level graph structures in a progressive and stochastic way from data during the LSTM network optimization. We thus call this model the structure-evolving LSTM. In particular, starting with an initial element-level graph representation where each node is a small data element, the structure-evolving LSTM gradually evolves the multi-level graph representations by stochastically merging the graph nodes with high compatibilities along the stacked LSTM layers. In each LSTM layer, we estimate the compatibility of two connected nodes from their corresponding LSTM gate outputs, which is used to generate a merging probability. The candidate graph structures are accordingly generated where the nodes are grouped into cliques with their merging probabilities. We then produce the new graph structure with a Metropolis-Hasting algorithm, which alleviates the risk of getting stuck in local optimums by stochastic sampling with an acceptance probability. Once a graph structure is accepted, a higher-level graph is then constructed by taking the partitioned cliques as its nodes. During the evolving process, representation becomes more abstracted in higher-levels where redundant information is filtered out, allowing more efficient propagation of long-range data dependencies. We evaluate the effectiveness of structure-evolving LSTM in the application of semantic object parsing and demonstrate its advantage over state-of-the-art LSTM models on standard benchmarks. ]_

_**Comment**_: &emsp;_<  structure-evolving LSTM, learning interpretable data representation via LSTM recurrent neural networks over hierarchal graph structures>_

#### ShapeOdds: Variational Bayesian Learning of Generative Shape Models  [PDF](https://www.researchgate.net/profile/Shireen_Elhabian/publication/314237204_ShapeOdds_Variational_Bayesian_Learning_of_Generative_Shape_Models/links/58beee88a6fdccff7b1f97c6/ShapeOdds-Variational-Bayesian-Learning-of-Generative-Shape-Models.pdf)

Shireen Elhabian, Ross Whitaker

_**Abstract**_: &emsp;_[Shape models provide a compact parameterization of a class of shapes, and have been shown to be important to a variety of vision problems, including object detection, tracking, and image segmentation. Learning generative shape models from grid-structured representations, aka silhouettes , is usually hindered by (1) data likelihoods with intractable marginals and posteriors, (2) high-dimensional shape spaces with limited training samples (and the associated risk of overfitting), and (3) estimation of hyperparam-eters relating to model complexity that often entails compu-tationally expensive grid searches. In this paper, we propose a Bayesian treatment that relies on direct probabilis-tic formulation for learning generative shape models in the silhouettes space. We propose a variational approach for learning a latent variable model in which we make use of, and extend, recent works on variational bounds of logistic-Gaussian integrals to circumvent intractable marginals and posteriors. Spatial coherency and sparsity priors are also incorporated to lend stability to the optimization problem by regularizing the solution space while avoiding overfitting in this high-dimensional, low-sample-size scenario. We deploy a type-II maximum likelihood estimate of the model hy-perparameters to avoid grid searches. We demonstrate that the proposed model generates realistic samples, generalizes to unseen examples, and is able to handle missing regions and/or background clutter, while comparing favorably with recent, neural-network-based approaches.]_

_**Comment**_: &emsp;_<  generative shape models in the silhouettes space>_

#### **Fast Video Classification via Adaptive Cascading of Deep Models  [PDF](https://arxiv.org/abs/1611.06453)

Haichen Shen, Seungyeop Han, Matthai Philipose, Arvind Krishnamurthy

_**Abstract**_: &emsp;_[Recent advances have enabled "oracle" classifiers that can classify across many classes and input distributions with high accuracy without retraining. However, these classifiers are relatively heavyweight, so that applying them to classify video is costly. We show that day-to-day video exhibits highly skewed class distributions over the short term, and that these distributions can be classified by much simpler models. We formulate the problem of detecting the short-term skews online and exploiting models based on it as a new sequential decision making problem dubbed the Online Bandit Problem, and present a new algorithm to solve it. When applied to recognizing faces in TV shows and movies, we realize end-to-end classification speedups of 2.5-8.5x/2.8-12.7x (on GPU/CPU) relative to a state-of-the-art convolutional neural network, at competitive accuracy. ]_

_**Comment**_: &emsp;_< skewed class distributions, as a new seqential disicison making, lightweight>_

#### *Deep Metric Learning via Facility Location  [PDF](https://arxiv.org/abs/1612.012130

Hyun Oh Song, Stefanie Jegelka, Vivek Rathod, Kevin Murphy

_**Abstract**_: &emsp;_[Learning the representation and the similarity metric in an end-to-end fashion with deep networks have demonstrated outstanding results for clustering and retrieval. However, these recent approaches still suffer from the performance degradation stemming from the local metric training procedure which is unaware of the global structure of the embedding space. We propose a global metric learning scheme for optimizing the deep metric embedding with the learnable clustering function and the clustering metric (NMI) in a novel structured prediction framework. Our experiments on CUB200-2011, Cars196, and Stanford online products datasets show state of the art performance both on the clustering and retrieval tasks measured in the NMI and Recall@K evaluation metrics. ]_

_**Comment**_: &emsp;_< global metric learning using DL>_

#### Semi-Supervised Deep Learning for Monocular Depth Map Prediction  [PDF](https://arxiv.org/abs/1702.02706)

Yevhen Kuznietsov, JÃ¶rg StÃ¼ckler, Bastian Leibe

_**Abstract**_: &emsp;_[Supervised deep learning often suffers from the lack of sufficient training data. Specifically in the context of monocular depth map prediction, it is barely possible to determine dense ground truth depth images in realistic dynamic outdoor environments. When using LiDAR sensors, for instance, noise is present in the distance measurements, the calibration between sensors cannot be perfect, and the measurements are typically much sparser than the camera images. In this paper, we propose a novel approach to depth map prediction from monocular images that learns in a semi-supervised way. While we use sparse ground-truth depth for supervised learning, we also enforce our deep network to produce photoconsistent dense depth maps in a stereo setup using a direct image alignment loss. In experiments we demonstrate superior performance in depth map prediction from single images compared to the state-of-the-art methods. ]_

_**Comment**_: &emsp;_< monocular depth map prediction, supervised + semi-supervised>_

#### *Weakly Supervised Semantic Segmentation Using Web-Crawled Videos  [PDF](https://arxiv.org/abs/1701.00352)

Seunghoon Hong, Donghun Yeo, Suha Kwak, Honglak Lee, Bohyung Han

_**Abstract**_: &emsp;_[We propose a novel algorithm for weakly supervised semantic segmentation based on image-level class labels only. In weakly supervised setting, it is commonly observed that trained model overly focuses on discriminative parts rather than the entire object area. Our goal is to overcome this limitation with no additional human intervention by retrieving videos relevant to target class labels from web repository, and generating segmentation labels from the retrieved videos to simulate strong supervision for semantic segmentation. During this process, we take advantage of image classification with discriminative localization technique to reject false alarms in retrieved videos and identify relevant spatio-temporal volumes within retrieved videos. Although the entire procedure does not require any additional supervision, the segmentation annotations obtained from videos are sufficiently strong to learn a model for semantic segmentation. The proposed algorithm substantially outperforms existing methods based on the same level of supervision and is even as competitive as the approaches relying on extra annotations. ]_

_**Comment**_: &emsp;_<  based on image-level class labels only, Guo's project>_

## Oral 2-1A

#### *Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach  [PDF](https://arxiv.org/abs/1609.03683)

Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, Lizhen Qu

_**Abstract**_: &emsp;_[We present a theoretically grounded approach to train deep neural networks, including recurrent networks, subject to class-dependent label noise. We propose two procedures for loss correction that are agnostic to both application domain and network architecture. They simply amount to at most a matrix inversion and multiplication, provided that we know the probability of each class being corrupted into another. We further show how one can estimate these probabilities, adapting a recent technique for noise estimation to the multi-class setting, and thus providing an end-to-end framework. Extensive experiments on MNIST, IMDB, CIFAR-10, CIFAR-100 and a large scale dataset of clothing images employing a diversity of architectures --- stacking dense, convolutional, pooling, dropout, batch normalization, word embedding, LSTM and residual layers --- demonstrate the noise robustness of our proposals. Incidentally, we also prove that, when ReLU is the only non-linearity, the loss curvature is immune to class-dependent label noise. ]_

_**Comment**_: &emsp;_< robust to label noise, >_

#### *Learning From Simulated and Unsupervised Images Through Adversarial Training  [PDF](https://arxiv.org/abs/1612.07828)
Ashish Shrivastava, Tomas Pfister, Oncel Tuzel, Joshua Susskind, Wenda Wang, Russell Webb

_**Abstract**_: &emsp;_[With recent progress in graphics, it has become more tractable to train models on synthetic images, potentially avoiding the need for expensive annotations. However, learning from synthetic images may not achieve the desired performance due to a gap between synthetic and real image distributions. To reduce this gap, we propose Simulated+Unsupervised (S+U) learning, where the task is to learn a model to improve the realism of a simulator's output using unlabeled real data, while preserving the annotation information from the simulator. We develop a method for S+U learning that uses an adversarial network similar to Generative Adversarial Networks (GANs), but with synthetic images as inputs instead of random vectors. We make several key modifications to the standard GAN algorithm to preserve annotations, avoid artifacts and stabilize training: (i) a 'self-regularization' term, (ii) a local adversarial loss, and (iii) updating the discriminator using a history of refined images. We show that this enables generation of highly realistic images, which we demonstrate both qualitatively and with a user study. We quantitatively evaluate the generated images by training models for gaze estimation and hand pose estimation. We show a significant improvement over using synthetic images, and achieve state-of-the-art results on the MPIIGaze dataset without any labeled real data. ]_

_**Comment**_: &emsp;_< train on synthetic images( exits a gap), Simulated+Unsupervised (S+U) learning; how about physical model's image>_

#### *Inverse Compositional Spatial Transformer Networks  [PDF](https://arxiv.org/abs/1612.03897)

Chen-Hsuan Lin, Simon Lucey

_**Abstract**_: &emsp;_[In this paper, we establish a theoretical connection between the classical Lucas & Kanade (LK) algorithm and the emerging topic of Spatial Transformer Networks (STNs). STNs are of interest to the vision and learning communities due to their natural ability to combine alignment and classification within the same theoretical framework. Inspired by the Inverse Compositional (IC) variant of the LK algorithm, we present Inverse Compositional Spatial Transformer Networks (IC-STNs). We demonstrate that IC-STNs can achieve better performance than conventional STNs with less model capacity; in particular, we show superior performance in pure image alignment tasks as well as joint alignment/classification problems on real-world problems. ]_

_**Comment**_: &emsp;_< LK algorithm & STN, image alignment tasks>_

#### ***Densely Connected Convolutional Networks  [PDF](https://arxiv.org/abs/1608.06993)  [PDF](https://github.com/liuzhuang13/DenseNet)

Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger

_**Abstract**_: &emsp;_[Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less memory and computation to achieve high performance. Code and models are available at this https URL . ]_

_**Comment**_: &emsp;_< shorter connections between 'input' &'output' layers, connects each layer to every other layer>_

# Computational Photography

## Spotlight 2-1B
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

## Oral 2-1B

Unrolling the Shutter: CNN to Correct Motion Distortions
Vijay Rengarajan, Yogesh Balaji, A. N. Rajagopalan
Light Field Blind Motion Deblurring
Pratul P. Srinivasan, Ren Ng, Ravi Ramamoorthi
Computational Imaging on the Electric Grid
Mark Sheinin, Yoav Y. Schechner, Kiriakos N. Kutulakos
Deep Outdoor Illumination Estimation
Yannick Hold-Geoffroy, Kalyan Sunkavalli, Sunil Hadap, Emiliano Gambaretto, Jean-FranÃ§ois Lalonde

# 3D Vision 2

## Spotlight 2-1C

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

## Oral 2-1C

A Point Set Generation Network for 3D Object Reconstruction From a Single Image
Haoqiang Fan, Hao Su, Leonidas J. Guibas
3D Point Cloud Registration for Localization Using a Deep Neural Network Auto-Encoder
Gil Elbaz, Tamar Avraham, Anath Fischer
Flight Dynamics-Based Recovery of a UAV Trajectory Using Ground Cameras
Artem Rozantsev, Sudipta N. Sinha, Debadeepta Dey, Pascal Fua
DSAC - Differentiable RANSAC for Camera Localization
Eric Brachmann, Alexander Krull, Sebastian Nowozin, Jamie Shotton, Frank Michel, Stefan Gumhold, Carsten Rother

# Poster 2-1

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
A Unified Approach of Multi-Scale Deep and Hand-Crafted Features for Defocus Estimation (PDF, project)
Jinsun Park, Yu-Wing Tai, Donghyeon Cho, In So Kweon
StyleBank: An Explicit Representation for Neural Image Style Transfer
Dongdong Chen, Lu Yuan, Jing Liao, Nenghai Yu, Gang Hua
Specular Highlight Removal in Facial Images
Chen Li, Stephen Lin, Kun Zhou, Katsushi Ikeuchi
Image Super-Resolution via Deep Recursive Residual Network
Ying Tai, Jian Yang, Xiaoming Liu
Deep Image Harmonization
Yi-Hsuan Tsai, Xiaohui Shen, Zhe Lin, Kalyan Sunkavalli, Xin Lu, Ming-Hsuan Yang
Learning Deep CNN Denoiser Prior for Image Restoration (PDF, code)
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

# Object Recognition & Scene Understanding 1

## Spotlight 2-2A

#### Graph-Structured Representations for Visual Question Answering  [PDF](https://arxiv.org/abs/1609.05600)

Damien Teney, Lingqiao Liu, Anton van den Hengel

_**Abstract**_: &emsp;_[This paper proposes to improve visual question answering (VQA) with structured representations of both scene contents and questions. A key challenge in VQA is to require joint reasoning over the visual and text domains. The predominant CNN/LSTM-based approach to VQA is limited by monolithic vector representations that largely ignore structure in the scene and in the form of the question. CNN feature vectors cannot effectively capture situations as simple as multiple object instances, and LSTMs process questions as series of words, which does not reflect the true complexity of language structure. We instead propose to build graphs over the scene objects and over the question words, and we describe a deep neural network that exploits the structure in these representations. This shows significant benefit over the sequential processing of LSTMs. The overall efficacy of our approach is demonstrated by significant improvements over the state-of-the-art, from 71.2% to 74.4% in accuracy on the "abstract scenes" multiple-choice benchmark, and from 34.7% to 39.1% in accuracy over pairs of "balanced" scenes, i.e. images with fine-grained differences and opposite yes/no answers to a same question. ]_

_**Comment**_: &emsp;_< structured representations of both scene contents and questions>_

#### Knowing When to Look: Adaptive Attention via a Visual Sentinel for Image Captioning  [PDF](https://arxiv.org/abs/1612.01887)

Jiasen Lu, Caiming Xiong, Devi Parikh, Richard Socher

_**Abstract**_: &emsp;_[Attention-based neural encoder-decoder frameworks have been widely adopted for image captioning. Most methods force visual attention to be active for every generated word. However, the decoder likely requires little to no visual information from the image to predict non-visual words such as "the" and "of". Other words that may seem visual can often be predicted reliably just from the language model e.g., "sign" after "behind a red stop" or "phone" following "talking on a cell". In this paper, we propose a novel adaptive attention model with a visual sentinel. At each time step, our model decides whether to attend to the image (and if so, to which regions) or to the visual sentinel. The model decides whether to attend to the image and where, in order to extract meaningful information for sequential word generation. We test our method on the COCO image captioning 2015 challenge dataset and Flickr30K. Our approach sets the new state-of-the-art by a significant margin. ]_

_**Comment**_: &emsp;_< >_

#### *Learned Contextual Feature Reweighting for Image Geo-Localization  [PDF](https://hyojin.web.unc.edu/files/2017/06/CVPR2017_0780.pdf)

Hyo Jin Kim, Enrique Dunn, Jan-Michael Frahm

_**Abstract**_: &emsp;_[We address the problem of large scale image geolocalization where the location of an image is estimated by identifying geo-tagged reference images depicting the same place. We propose a novel model for learning image representations that integrates context-aware feature reweighting in order to effectively focus on regions that positively contribute to geo-localization. In particular, we introduce a Contextual Reweighting Network (CRN) that predicts the importance of each region in the feature map based on the image context. Our model is learned end-to-end for the image geo-localization task, and requires no annotation other than image geo-tags for training. In experimental results, the proposed approach significantly outperforms the previous state-of-the-art on the standard geo-localization benchmark datasets.We also demonstrate that our CRN discovers task-relevant contexts without any additional supervision.]_

_**Comment**_: &emsp;_< geolocalization by geo-tagged reference images, predicts the importance of each region in the feature map based on the image context>_

#### End-To-End Concept Word Detection for Video Captioning, Retrieval, and Question Answering  [PDF](https://128.84.21.199/abs/1610.02947v2)

Youngjae Yu, Hyungjin Ko, Jongwook Choi, Gunhee Kim

_**Abstract**_: &emsp;_[We propose a high-level concept word detector that can be integrated with any video-to-language models. It takes a video as input and generates a list of concept words as useful semantic priors for language generation models. The proposed word detector has two important properties. First, it does not require any external knowledge sources for training. Second, the proposed word detector is trainable in an end-to-end manner jointly with any video-to-language models. To maximize the values of detected words, we also develop a semantic attention mechanism that selectively focuses on the detected concept words and fuse them with the word encoding and decoding in the language model. In order to demonstrate that the proposed approach indeed improves the performance of multiple video-to-language tasks, we participate in four tasks of LSMDC 2016. Our approach achieves the best accuracies in three of them, including fill-in-the-blank, multiple-choice test, and movie retrieval. We also attain comparable performance for the other task, movie description.]_

_**Comment**_: &emsp;_< concept word detector with any video-to-language tasks>_

#### Deep Cross-Modal Hashing  [PDF](https://arxiv.org/abs/1602.02255)

Qing-Yuan Jiang, Wu-Jun Li

_**Abstract**_: &emsp;_[Due to its low storage cost and fast query speed, cross-modal hashing (CMH) has been widely used for similarity search in multimedia retrieval applications. However, almost all existing CMH methods are based on hand-crafted features which might not be optimally compatible with the hash-code learning procedure. As a result, existing CMH methods with handcrafted features may not achieve satisfactory performance. In this paper, we propose a novel cross-modal hashing method, called deep crossmodal hashing (DCMH), by integrating feature learning and hash-code learning into the same framework. DCMH is an end-to-end learning framework with deep neural networks, one for each modality, to perform feature learning from scratch. Experiments on two real datasets with text-image modalities show that DCMH can outperform other baselines to achieve the state-of-the-art performance in cross-modal retrieval applications. ]_

_**Comment**_: &emsp;_< cross-modal hashing for similarity search, integrating feature learning and hash-code learning into the same framework>_

#### Unambiguous Text Localization and Retrieval for Cluttered Scenes  [PDF](http://www-ee.ccny.cuny.edu/wwwn/yltian/Publications/CVPR17-2326.pdf)

Xuejian Rong, Chucai Yi, Yingli Tian

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< Text instance localization>_

#### Bayesian Supervised Hashing

Zihao Hu, Junxuan Chen, Hongtao Lu, Tongzhen Zhang

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### ***Speed/Accuracy Trade-Offs for Modern Convolutional Object Detectors  [PDF](https://arxiv.org/abs/1611.10012)

Jonathan Huang, Vivek Rathod, Chen Sun, Menglong Zhu, Anoop Korattikara, Alireza Fathi, Ian Fischer, Zbigniew Wojna, Yang Song, Sergio Guadarrama, Kevin Murphy

_**Abstract**_: &emsp;_[The goal of this paper is to serve as a guide for selecting a detection architecture that achieves the right speed/memory/accuracy balance for a given application and platform. To this end, we investigate various ways to trade accuracy for speed and memory usage in modern convolutional object detection systems. A number of successful systems have been proposed in recent years, but apples-to-apples comparisons are difficult due to different base feature extractors (e.g., VGG, Residual Networks), different default image resolutions, as well as different hardware and software platforms. We present a unified implementation of the Faster R-CNN [Ren et al., 2015], R-FCN [Dai et al., 2016] and SSD [Liu et al., 2015] systems, which we view as "meta-architectures" and trace out the speed/accuracy trade-off curve created by using alternative feature extractors and varying other critical parameters such as image size within each of these meta-architectures. On one extreme end of this spectrum where speed and memory are critical, we present a detector that achieves real time speeds and can be deployed on a mobile device. On the opposite end in which accuracy is critical, we present a detector that achieves state-of-the-art performance measured on the COCO detection task.]_

_**Comment**_: &emsp;_< a gudide for speed/memory/accuracy balance, comparsion>_

## Oral 2-2A

#### **Detecting Visual Relationships With Deep Relational Networks  [PDF](https://arxiv.org/abs/1704.03114)

Bo Dai, Yuqi Zhang, Dahua Lin

_**Abstract**_: &emsp;_[Relationships among objects play a crucial role in image understanding. Despite the great success of deep learning techniques in recognizing individual objects, reasoning about the relationships among objects remains a challenging task. Previous methods often treat this as a classification problem, considering each type of relationship (e.g. "ride") or each distinct visual phrase (e.g. "person-ride-horse") as a category. Such approaches are faced with significant difficulties caused by the high diversity of visual appearance for each kind of relationships or the large number of distinct visual phrases. We propose an integrated framework to tackle this problem. At the heart of this framework is the Deep Relational Network, a novel formulation designed specifically for exploiting the statistical dependencies between objects and their relationships. On two large datasets, the proposed method achieves substantial improvement over state-of-the-art.]_

_**Comment**_: &emsp;_<  Relationships among objects,Deep Relational Network>_

#### *Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes  [PDF](https://arxiv.org/abs/1611.08323)

Tobias Pohlen, Alexander Hermans, Markus Mathias, Bastian Leibe

_**Abstract**_: &emsp;_[Semantic image segmentation is an essential component of modern autonomous driving systems, as an accurate understanding of the surrounding scene is crucial to navigation and action planning. Current state-of-the-art approaches in semantic image segmentation rely on pre-trained networks that were initially developed for classifying images as a whole. While these networks exhibit outstanding recognition performance (i.e., what is visible?), they lack localization accuracy (i.e., where precisely is something located?). Therefore, additional processing steps have to be performed in order to obtain pixel-accurate segmentation masks at the full image resolution. To alleviate this problem we propose a novel ResNet-like architecture that exhibits strong localization and recognition performance. We combine multi-scale context with pixel-level accuracy by using two processing streams within our network: One stream carries information at the full image resolution, enabling precise adherence to segment boundaries. The other stream undergoes a sequence of pooling operations to obtain robust features for recognition. The two streams are coupled at the full image resolution using residuals. Without additional processing steps and without pre-training, our approach achieves an intersection-over-union score of 71.8% on the Cityscapes dataset. ]_

_**Comment**_: &emsp;_<  semantic image segmentation,using two processing stream(one for full image, one for robust features>_

#### ***Network Dissection: Quantifying Interpretability of Deep Visual Representations  [PDF](https://arxiv.org/abs/1704.05796)

David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, Antonio Torralba

_**Abstract**_: &emsp;_[We propose a general framework called Network Dissection for quantifying the interpretability of latent representations of CNNs by evaluating the alignment between individual hidden units and a set of semantic concepts. Given any CNN model, the proposed method draws on a broad data set of visual concepts to score the semantics of hidden units at each intermediate convolutional layer. The units with semantics are given labels across a range of objects, parts, scenes, textures, materials, and colors. We use the proposed method to test the hypothesis that interpretability of units is equivalent to random linear combinations of units, then we apply our method to compare the latent representations of various networks when trained to solve different supervised and self-supervised training tasks. We further analyze the effect of training iterations, compare networks trained with different initializations, examine the impact of network depth and width, and measure the effect of dropout and batch normalization on the interpretability of deep visual representations. We demonstrate that the proposed method can shed light on characteristics of CNN models and training methods that go beyond measurements of their discriminative power. ]_

_**Comment**_: &emsp;_<  score the semantics of hidden units for different objects using different CNN, The units with semantics are given labels, interpretability of units is equivalent to random linear combinations of units, >_

#### **AGA: Attribute-Guided Augmentation  [PDF](https://arxiv.org/abs/1612.02559)

Mandar Dixit, Roland Kwitt, Marc Niethammer, Nuno Vasconcelos

_**Abstract**_: &emsp;_[We consider the problem of data augmentation, i.e., generating artificial samples to extend a given corpus of training data. Specifically, we propose attributed-guided augmentation (AGA) which learns a mapping that allows to synthesize data such that an attribute of a synthesized sample is at a desired value or strength. This is particularly interesting in situations where little data with no attribute annotation is available for learning, but we have access to a large external corpus of heavily annotated samples. While prior works primarily augment in the space of images, we propose to perform augmentation in feature space instead. We implement our approach as a deep encoder-decoder architecture that learns the synthesis function in an end-to-end manner. We demonstrate the utility of our approach on the problems of (1) one-shot object recognition in a transfer-learning setting where we have no prior knowledge of the new classes, as well as (2) object-based one-shot scene recognition. As external data, we leverage 3D depth and pose information from the SUN RGB-D dataset. Our experiments show that attribute-guided augmentation of high-level CNN features considerably improves one-shot recognition performance on both problems. ]_

_**Comment**_: &emsp;_< attributed-guided data augmentation in feature space, a deep encoder-decoder architecture; transfer learning, one-shot recognition >_

# Analyzing Humans 2

## Spotlight 2-2B

#### A Hierarchical Approach for Generating Descriptive Image Paragraphs  [PDF](https://arxiv.org/abs/1611.06607)

Jonathan Krause, Justin Johnson, Ranjay Krishna, Li Fei-Fei

_**Abstract**_: &emsp;_[Recent progress on image captioning has made it possible to generate novel sentences describing images in natural language, but compressing an image into a single sentence can describe visual content in only coarse detail. While one new captioning approach, dense captioning, can potentially describe images in finer levels of detail by captioning many regions within an image, it in turn is unable to produce a coherent story for an image. In this paper we overcome these limitations by generating entire paragraphs for describing images, which can tell detailed, unified stories. We develop a model that decomposes both images and paragraphs into their constituent parts, detecting semantic regions in images and using a hierarchical recurrent neural network to reason about language. Linguistic analysis confirms the complexity of the paragraph generation task, and thorough experiments on a new dataset of image and paragraph pairs demonstrate the effectiveness of our approach. ]_

_**Comment**_: &emsp;_<  generating entire paragraphs for describing images>_

#### Person Re-Identification in the Wild  [PDF](https://arxiv.org/abs/1604.02531)

Liang Zheng, Hengheng Zhang, Shaoyan Sun, Manmohan Chandraker, Yi Yang, Qi Tian

_**Abstract**_: &emsp;_[We present a novel large-scale dataset and comprehensive baselines for end-to-end pedestrian detection and person recognition in raw video frames. Our baselines address three issues: the performance of various combinations of detectors and recognizers, mechanisms for pedestrian detection to help improve overall re-identification accuracy and assessing the effectiveness of different detectors for re-identification. We make three distinct contributions. First, a new dataset, PRW, is introduced to evaluate Person Re-identification in the Wild, using videos acquired through six synchronized cameras. It contains 932 identities and 11,816 frames in which pedestrians are annotated with their bounding box positions and identities. Extensive benchmarking results are presented on this dataset. Second, we show that pedestrian detection aids re-identification through two simple yet effective improvements: a discriminatively trained ID-discriminative Embedding (IDE) in the person subspace using convolutional neural network (CNN) features and a Confidence Weighted Similarity (CWS) metric that incorporates detection scores into similarity measurement. Third, we derive insights in evaluating detector performance for the particular scenario of accurate person re-identification. ]_

_**Comment**_: &emsp;_< end-to-end pedestrian detection and person recognition in raw video frames, >_

#### Scalable Person Re-Identification on Supervised Smoothed Manifold  [PDF](https://arxiv.org/abs/1703.08359)

Song Bai, Xiang Bai, Qi Tian

_**Abstract**_: &emsp;_[Most existing person re-identification algorithms either extract robust visual features or learn discriminative metrics for person images. However, the underlying manifold which those images reside on is rarely investigated. That raises a problem that the learned metric is not smooth with respect to the local geometry structure of the data manifold. In this paper, we study person re-identification with manifold-based affinity learning, which did not receive enough attention from this area. An unconventional manifold-preserving algorithm is proposed, which can 1) make the best use of supervision from training data, whose label information is given as pairwise constraints; 2) scale up to large repositories with low on-line time complexity; and 3) be plunged into most existing algorithms, serving as a generic postprocessing procedure to further boost the identification accuracies. Extensive experimental results on five popular person re-identification benchmarks consistently demonstrate the effectiveness of our method. Especially, on the largest CUHK03 and Market-1501, our method outperforms the state-of-the-art alternatives by a large margin with high efficiency, which is more appropriate for practical applications. ]_

_**Comment**_: &emsp;_< maniford>_

#### Binge Watching: Scaling Affordance Learning From Sitcoms

Xiaolong Wang, Rohit Girdhar, Abhinav Gupta

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Joint Detection and Identification Feature Learning for Person Search  [PDF](https://pdfs.semanticscholar.org/cec9/17ae255439e17b9a345556f1e091b6b9485a.pdf?_ga=2.153350832.985571883.1498350464-2104590067.1498350464)

Tong Xiao, Shuang Li, Bochao Wang, Liang Lin, Xiaogang Wang

_**Abstract**_: &emsp;_[Existing person re-identification benchmarks and methods mainly focus on matching cropped pedestrian images between queries and candidates. However, it is different from real-world scenarios where the annotations of pedestrian bounding boxes are unavailable and the target person needs to be searched from a gallery of whole scene images. To close the gap, we propose a new deep learning framework for person search. Instead of breaking it down into two separate tasks—pedestrian detection and person re-identification, we jointly handle both aspects in a single convolutional neural network. An Online Instance Matching (OIM) loss function is proposed to train the network effectively , which is scalable to datasets with numerous identities. To validate our approach, we collect and annotate a large-scale benchmark dataset for person search. It contains 18, 184 images, 8, 432 identities, and 96, 143 pedestrian bounding boxes. Experiments show that our framework outperforms other separate approaches, and the proposed OIM loss function converges much faster and better than the conventional Softmax loss.]_

_**Comment**_: &emsp;_< jointly handle pedestrian detection and reidentification in a single CNN, OIM losss function>_

#### Synthesizing Normalized Faces From Facial Identity Features  [PDF](https://arxiv.org/abs/1701.04851)

Forrester Cole, David Belanger, Dilip Krishnan, Aaron Sarna, Inbar Mosseri, William T. Freeman

_**Abstract**_: &emsp;_[We present a method for synthesizing a frontal, neutral-expression image of a person's face given an input face photograph. This is achieved by learning to generate facial landmarks and textures from features extracted from a facial-recognition network. Unlike previous approaches, our encoding feature vector is largely invariant to lighting, pose, and facial expression. Exploiting this invariance, we train our decoder network using only frontal, neutral-expression photographs. Since these photographs are well aligned, we can decompose them into a sparse set of landmark points and aligned texture maps. The decoder then predicts landmarks and textures independently and combines them using a differentiable image warping operation. The resulting images can be used for a number of applications, such as analyzing facial attributes, exposure and white balance adjustment, or creating a 3-D avatar. ]_

_**Comment**_: &emsp;_< generate a frontal, neutral-expression image,  predicts landmarks and textures independently>_

#### Consistent-Aware Deep Learning for Person Re-Identification in a Camera Network

Ji Lin, Liangliang Ren, Jiwen Lu, Jianjiang Feng, Jie Zhou

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Level Playing Field for Million Scale Face Recognition  [PDF](https://arxiv.org/abs/1705.00393)

Aaron Nech, Ira Kemelmacher-Shlizerman

_**Abstract**_: &emsp;_[Face recognition has the perception of a solved problem, however when tested at the million-scale exhibits dramatic variation in accuracies across the different algorithms. Are the algorithms very different? Is access to good/big training data their secret weapon? Where should face recognition improve? To address those questions, we created a benchmark, MF2, that requires all algorithms to be trained on same data, and tested at the million scale. MF2 is a public large-scale set with 672K identities and 4.7M photos created with the goal to level playing field for large scale face recognition. We contrast our results with findings from the other two large-scale benchmarks MegaFace Challenge and MS-Celebs-1M where groups were allowed to train on any private/public/big/small set. Some key discoveries: 1) algorithms, trained on MF2, were able to achieve state of the art and comparable results to algorithms trained on massive private sets, 2) some outperformed themselves once trained on MF2, 3) invariance to aging suffers from low accuracies as in MegaFace, identifying the need for larger age variations possibly within identities or adjustment of algorithms in future testings. ]_

_**Comment**_: &emsp;_< a new face dataset>_

## Oral 2-2B

#### Re-Sign: Re-Aligned End-To-End Sequence Modelling With Deep Recurrent CNN-HMMs  [PDF](https://www.researchgate.net/profile/Oscar_Koller/publication/315892054_Re-Sign_Re-Aligned_End-to-End_Sequence_Modelling_with_Deep_Recurrent_CNN-HMMs/links/58ec87800f7e9b6b274bb137/Re-Sign-Re-Aligned-End-to-End-Sequence-Modelling-with-Deep-Recurrent-CNN-HMMs.pdf)

Oscar Koller, Sepehr Zargaran, Hermann Ney

_**Abstract**_: &emsp;_[This work presents an iterative realignment approach applicable to visual sequence labelling tasks such as gesture recognition, activity recognition and continuous sign language recognition. Previous methods dealing with video data usually rely on given frame labels to train their clas-sifiers. Looking at recent data sets, these labels often tend to be noisy which is commonly overseen. We propose an algorithm that treats the provided training labels as weak labels and refines the label-to-image alignment on-the-fly in a weakly supervised fashion. Given a series of frames and sequence-level labels, a deep recurrent CNN-BLSTM network is trained end-to-end. Embedded into an HMM, the resulting deep model corrects the frame labels and continuously improves its performance in several realignments. We evaluate on two challenging publicly available sign recognition benchmark data sets featuring over 1000 classes. We outperform the state-of-the-art by up to 10% absolute and 30% relative. ]_

_**Comment**_: &emsp;_< visual sequence labelling tasks such as gesture recognition, activity recognition and continuous sign language recognition; using weak labels>_

#### Social Scene Understanding: End-To-End Multi-Person Action Localization and Collective Activity Recognition  [PDF](https://arxiv.org/abs/1611.09078)

Timur Bagautdinov, Alexandre Alahi, FranÃ§ois Fleuret, Pascal Fua, Silvio Savarese

_**Abstract**_: &emsp;_[We present a unified framework for understanding human social behaviors in raw image sequences. Our model jointly detects multiple individuals, infers their social actions, and estimates the collective actions with a single feed-forward pass through a neural network. We propose a single architecture that does not rely on external detection algorithms but rather is trained end-to-end to generate dense proposal maps that are refined via a novel inference scheme. The temporal consistency is handled via a person-level matching Recurrent Neural Network. The complete model takes as input a sequence of frames and outputs detections along with the estimates of individual actions and collective activities. We demonstrate state-of-the-art performance of our algorithm on multiple publicly available benchmarks. ]_

_**Comment**_: &emsp;_< understanding human social behaviors in raw image sequences>_

#### Detangling People: Individuating Multiple Close People and Their Body Parts via Region Assembly  [PDF](https://arxiv.org/abs/1604.03880)

Hao Jiang, Kristen Grauman

_**Abstract**_: &emsp;_[Today's person detection methods work best when people are in common upright poses and appear reasonably well spaced out in the image. However, in many real images, that's not what people do. People often appear quite close to each other, e.g., with limbs linked or heads touching, and their poses are often not pedestrian-like. We propose an approach to detangle people in multi-person images. We formulate the task as a region assembly problem. Starting from a large set of overlapping regions from body part semantic segmentation and generic object proposals, our optimization approach reassembles those pieces together into multiple person instances. It enforces that the composed body part regions of each person instance obey constraints on relative sizes, mutual spatial relationships, foreground coverage, and exclusive label assignments when overlapping. Since optimal region assembly is a challenging combinatorial problem, we present a Lagrangian relaxation method to accelerate the lower bound estimation, thereby enabling a fast branch and bound solution for the global optimum. As output, our method produces a pixel-level map indicating both 1) the body part labels (arm, leg, torso, and head), and 2) which parts belong to which individual person. Our results on three challenging datasets show our method is robust to clutter, occlusion, and complex poses. It outperforms a variety of competing methods, including existing detector CRF methods and region CNN approaches. In addition, we demonstrate its impact on a proxemics recognition task, which demands a precise representation of "whose body part is where" in crowded images. ]_

_**Comment**_: &emsp;_< detangle people in multi-person images,>_

#### Lip Reading Sentences in the Wild  [PDF](https://arxiv.org/abs/1611.05358)

Joon Son Chung, Andrew Senior, Oriol Vinyals, Andrew Zisserman

_**Abstract**_: &emsp;_[The goal of this work is to recognise phrases and sentences being spoken by a talking face, with or without the audio. Unlike previous works that have focussed on recognising a limited number of words or phrases, we tackle lip reading as an open-world problem - unconstrained natural language sentences, and in the wild videos.
Our key contributions are: (1) a 'Watch, Listen, Attend and Spell' (WLAS) network that learns to transcribe videos of mouth motion to characters; (2) a curriculum learning strategy to accelerate training and to reduce overfitting; (3) a 'Lip Reading Sentences' (LRS) dataset for visual speech recognition, consisting of over 100,000 natural sentences from British television.
The WLAS model trained on the LRS dataset surpasses the performance of all previous work on standard lip reading benchmark datasets, often by a significant margin. This lip reading performance beats a professional lip reader on videos from BBC television, and we also demonstrate that visual information helps to improve speech recognition performance even when the audio is available. ]_

_**Comment**_: &emsp;_< lip reading>_

# Applications

## Spotlight 2-2C

#### *Deep Matching Prior Network: Toward Tighter Multi-Oriented Text Detection  [PDF](https://arxiv.org/abs/1703.014250）

Yuliang Liu, Lianwen Jin

_**Abstract**_: &emsp;_[Detecting incidental scene text is a challenging task because of multi-orientation, perspective distortion, and variation of text size, color and scale. Retrospective research has only focused on using rectangular bounding box or horizontal sliding window to localize text, which may result in redundant background noise, unnecessary overlap or even information loss. To address these issues, we propose a new Convolutional Neural Networks (CNNs) based method, named Deep Matching Prior Network (DMPNet), to detect text with tighter quadrangle. First, we use quadrilateral sliding windows in several specific intermediate convolutional layers to roughly recall the text with higher overlapping area and then a shared Monte-Carlo method is proposed for fast and accurate computing of the polygonal areas. After that, we designed a sequential protocol for relative regression which can exactly predict text with compact quadrangle. Moreover, a auxiliary smooth Ln loss is also proposed for further regressing the position of text, which has better overall performance than L2 loss and smooth L1 loss in terms of robustness and stability. The effectiveness of our approach is evaluated on a public word-level, multi-oriented scene text database, ICDAR 2015 Robust Reading Competition Challenge 4 "Incidental scene text localization". The performance of our method is evaluated by using F-measure and found to be 70.64%, outperforming the existing state-of-the-art method with F-measure 63.76%. ]_

_**Comment**_: &emsp;_< text detection, CNN based Deep Matching Prior Net, with quadrangle slding windows>_

#### ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases  [PDF](https://arxiv.org/abs/1705.02315)

Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald M. Summers

_**Abstract**_: &emsp;_[The chest X-ray is one of the most commonly accessible radiological examinations for screening and diagnosis of many lung diseases. A tremendous number of X-ray imaging studies accompanied by radiological reports are accumulated and stored in many modern hospitals' Picture Archiving and Communication Systems (PACS). On the other side, it is still an open question how this type of hospital-size knowledge database containing invaluable imaging informatics (i.e., loosely labeled) can be used to facilitate the data-hungry deep learning paradigms in building truly large-scale high precision computer-aided diagnosis (CAD) systems. In this paper, we present a new chest X-ray database, namely "ChestX-ray8", which comprises 108,948 frontal-view X-ray images of 32,717 unique patients with the text-mined eight disease image labels (where each image can have multi-labels), from the associated radiological reports using natural language processing. Importantly, we demonstrate that these commonly occurring thoracic diseases can be detected and even spatially-located via a unified weakly-supervised multi-label image classification and disease localization framework, which is validated using our proposed dataset. Although the initial quantitative results are promising as reported, deep convolutional neural network based "reading chest X-rays" (i.e., recognizing and locating the common disease patterns trained with only image-level labels) remains a strenuous task for fully-automated high precision CAD systems. ]_

_**Comment**_: &emsp;_< new X-ray dataset>_

#### *Attentional Push: A Deep Convolutional Network for Augmenting Image Salience With Shared Attention Modeling in Social Scenes

Siavash Gorji, James J. Clark

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### *Detecting Oriented Text in Natural Images by Linking Segments  [PDF](https://arxiv.org/abs/1703.06520)

Baoguang Shi, Xiang Bai, Serge Belongie

_**Abstract**_: &emsp;_[Most state-of-the-art text detection methods are specific to horizontal Latin text and are not fast enough for real-time applications. We introduce Segment Linking (SegLink), an oriented text detection method. The main idea is to decompose text into two locally detectable elements, namely segments and links. A segment is an oriented box covering a part of a word or text line; A link connects two adjacent segments, indicating that they belong to the same word or text line. Both elements are detected densely at multiple scales by an end-to-end trained, fully-convolutional neural network. Final detections are produced by combining segments connected by links. Compared with previous methods, SegLink improves along the dimensions of accuracy, speed, and ease of training. It achieves an f-measure of 75.0% on the standard ICDAR 2015 Incidental (Challenge 4) benchmark, outperforming the previous best by a large margin. It runs at over 20 FPS on 512x512 images. Moreover, without modification, SegLink is able to detect long lines of non-Latin text, such as Chinese. ]_

_**Comment**_: &emsp;_< oriented text; segment(oriented box coving words & linking(connects two adjacent segment>_

#### Learning Video Object Segmentation From Static Images  [PDF](https://arxiv.org/abs/1612.02646)

Federico Perazzi, Anna Khoreva, Rodrigo Benenson, Bernt Schiele, Alexander Sorkine-Hornung

_**Abstract**_: &emsp;_[Inspired by recent advances of deep learning in instance segmentation and object tracking, we introduce video object segmentation problem as a concept of guided instance segmentation. Our model proceeds on a per-frame basis, guided by the output of the previous frame towards the object of interest in the next frame. We demonstrate that highly accurate object segmentation in videos can be enabled by using a convnet trained with static images only. The key ingredient of our approach is a combination of offline and online learning strategies, where the former serves to produce a refined mask from the previous frame estimate and the latter allows to capture the appearance of the specific object instance. Our method can handle different types of input annotations: bounding boxes and segments, as well as incorporate multiple annotated frames, making the system suitable for diverse applications. We obtain competitive results on three different datasets, independently from the type of input annotation. ]_

_**Comment**_: &emsp;_< >_

#### Seeing Invisible Poses: Estimating 3D Body Pose From Egocentric Video  [PDF](https://arxiv.org/abs/1603.07763)

Hao Jiang, Kristen Grauman

_**Abstract**_: &emsp;_[Understanding the camera wearer's activity is central to egocentric vision, yet one key facet of that activity is inherently invisible to the camera--the wearer's body pose. Prior work focuses on estimating the pose of hands and arms when they come into view, but this 1) gives an incomplete view of the full body posture, and 2) prevents any pose estimate at all in many frames, since the hands are only visible in a fraction of daily life activities. We propose to infer the "invisible pose" of a person behind the egocentric camera. Given a single video, our efficient learning-based approach returns the full body 3D joint positions for each frame. Our method exploits cues from the dynamic motion signatures of the surrounding scene--which changes predictably as a function of body pose--as well as static scene structures that reveal the viewpoint (e.g., sitting vs. standing). We further introduce a novel energy minimization scheme to infer the pose sequence. It uses soft predictions of the poses per time instant together with a non-parametric model of human pose dynamics over longer windows. Our method outperforms an array of possible alternatives, including deep learning approaches for direct pose regression from images. ]_

_**Comment**_: &emsp;_< estimate camera wearer's body pose>_

#### Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space  [PDF](https://arxiv.org/abs/1612.00005)

Anh Nguyen, Jeff Clune, Yoshua Bengio, Alexey Dosovitskiy, Jason Yosinski

_**Abstract**_: &emsp;_[Generating high-resolution, photo-realistic images has been a long-standing goal in machine learning. Recently, Nguyen et al. (2016) showed one interesting way to synthesize novel images by performing gradient ascent in the latent space of a generator network to maximize the activations of one or multiple neurons in a separate classifier network. In this paper we extend this method by introducing an additional prior on the latent code, improving both sample quality and sample diversity, leading to a state-of-the-art generative model that produces high quality images at higher resolutions (227x227) than previous generative models, and does so for all 1000 ImageNet categories. In addition, we provide a unified probabilistic interpretation of related activation maximization methods and call the general class of models "Plug and Play Generative Networks". PPGNs are composed of 1) a generator network G that is capable of drawing a wide range of image types and 2) a replaceable "condition" network C that tells the generator what to draw. We demonstrate the generation of images conditioned on a class (when C is an ImageNet or MIT Places classification network) and also conditioned on a caption (when C is an image captioning network). Our method also improves the state of the art of Multifaceted Feature Visualization, which generates the set of synthetic inputs that activate a neuron in order to better understand how deep neural networks operate. Finally, we show that our model performs reasonably well at the task of image inpainting. While image models are used in this paper, the approach is modality-agnostic and can be applied to many types of data. ]_

_**Comment**_: &emsp;_< >_

#### *A Joint Speaker-Listener-Reinforcer Model for Referring Expressions  [PDF](https://arxiv.org/abs/1612.09542)

Licheng Yu, Hao Tan, Mohit Bansal, Tamara L. Berg

_**Abstract**_: &emsp;_[Referring expressions are natural language constructions used to identify particular objects within a scene. In this paper, we propose a unified framework for the tasks of referring expression comprehension and generation. Our model is composed of three modules: speaker, listener, and reinforcer. The speaker generates referring expressions, the listener comprehends referring expressions, and the reinforcer introduces a reward function to guide sampling of more discriminative expressions. The listener-speaker modules are trained jointly in an end-to-end learning framework, allowing the modules to be aware of one another during learning while also benefiting from the discriminative reinforcer's feedback. We demonstrate that this unified framework and training achieves state-of-the-art results for both comprehension and generation on three referring expression datasets. Project and demo page: this https URL]_

_**Comment**_: &emsp;_< referring expression, Speaker-Listener-Reinforcer network>_

## Oral 2-2C

#### End-To-End Learning of Driving Models From Large-Scale Video Datasets  [PDF](https://arxiv.org/abs/1612.010790

Huazhe Xu, Yang Gao, Fisher Yu, Trevor Darrell

_**Abstract**_: &emsp;_[Robust perception-action models should be learned from training data with diverse visual appearances and realistic behaviors, yet current approaches to deep visuomotor policy learning have been generally limited to in-situ models learned from a single vehicle or a simulation environment. We advocate learning a generic vehicle motion model from large scale crowd-sourced video data, and develop an end-to-end trainable architecture for learning to predict a distribution over future vehicle egomotion from instantaneous monocular camera observations and previous vehicle state. Our model incorporates a novel FCN-LSTM architecture, which can be learned from large-scale crowd-sourced vehicle action data, and leverages available scene segmentation side tasks to improve performance under a privileged learning paradigm. ]_

_**Comment**_: &emsp;_< driving models>_

#### Deep Future Gaze: Gaze Anticipation on Egocentric Videos Using Adversarial Networks

Mengmi Zhang, Keng Teck Ma, Joo Hwee Lim, Qi Zhao, Jiashi Feng

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### *MDNet: A Semantically and Visually Interpretable Medical Image Diagnosis Network

Zizhao Zhang, Yuanpu Xie, Fuyong Xing, Mason McGough, Lin Yang

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

# Poster 2-2

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

# Machine Learning 3

## Spotlight 3-1A

#### ***Local Binary Convolutional Neural Networks  [PDF](https://arxiv.org/abs/1608.06049)

Felix Juefei-Xu, Vishnu Naresh Boddeti, Marios Savvides

_**Abstract**_: &emsp;_[We propose local binary convolution (LBC), an efficient alternative to convolutional layers in standard convolutional neural networks (CNN). The design principles of LBC are motivated by local binary patterns (LBP). The LBC layer comprises of a set of fixed sparse pre-defined binary convolutional filters that are not updated during the training process, a non-linear activation function and a set of learnable linear weights. The linear weights combine the activated filter responses to approximate the corresponding activated filter responses of a standard convolutional layer. The LBC layer affords significant parameter savings, 9x to 169x in the number of learnable parameters compared to a standard convolutional layer. Furthermore, due to lower model complexity and sparse and binary nature of the weights also results in up to 9x to 169x savings in model size compared to a standard convolutional layer. We demonstrate both theoretically and experimentally that our local binary convolution layer is a good approximation of a standard convolutional layer. Empirically, CNNs with LBC layers, called local binary convolutional neural networks (LBCNN), reach state-of-the-art performance on a range of visual datasets (MNIST, SVHN, CIFAR-10, and a subset of ImageNet) while enjoying significant computational savings. ]_

_**Comment**_: &emsp;_< LBCNN, approximation to standard CNN>_

#### *Deep Self-Taught Learning for Weakly Supervised Object Localization  [PDF](https://arxiv.org/abs/1704.05188)

Zequn Jie, Yunchao Wei, Xiaojie Jin, Jiashi Feng, Wei Liu

_**Abstract**_: &emsp;_[Most existing weakly supervised localization (WSL) approaches learn detectors by finding positive bounding boxes based on features learned with image-level supervision. However, those features do not contain spatial location related information and usually provide poor-quality positive samples for training a detector. To overcome this issue, we propose a deep self-taught learning approach, which makes the detector learn the object-level features reliable for acquiring tight positive samples and afterwards re-train itself based on them. Consequently, the detector progressively improves its detection ability and localizes more informative positive samples. To implement such self-taught learning, we propose a seed sample acquisition method via image-to-object transferring and dense subgraph discovery to find reliable positive samples for initializing the detector. An online supportive sample harvesting scheme is further proposed to dynamically select the most confident tight positive samples and train the detector in a mutual boosting way. To prevent the detector from being trapped in poor optima due to overfitting, we propose a new relative improvement of predicted CNN scores for guiding the self-taught learning process. Extensive experiments on PASCAL 2007 and 2012 show that our approach outperforms the state-of-the-arts, strongly validating its effectiveness. ]_

_**Comment**_: &emsp;_< selft-taught learning, imporve progressively>_

#### Multi-Modal Mean-Fields via Cardinality-Based Clamping  [PDF](https://arxiv.org/abs/1611.07941)

Pierre Baqué, FranÃ§ois Fleuret, Pascal Fua

_**Abstract**_: &emsp;_[Mean Field inference is central to statistical physics. It has attracted much interest in the Computer Vision community to efficiently solve problems expressible in terms of large Conditional Random Fields. However, since it models the posterior probability distribution as a product of marginal probabilities, it may fail to properly account for important dependencies between variables. We therefore replace the fully factorized distribution of Mean Field by a weighted mixture of such distributions, that similarly minimizes the KL-Divergence to the true posterior. By introducing two new ideas, namely, conditioning on groups of variables instead of single ones and using a parameter of the conditional random field potentials, that we identify to the temperature in the sense of statistical physics to select such groups, we can perform this minimization efficiently. Our extension of the clamping method proposed in previous works allows us to both produce a more descriptive approximation of the true posterior and, inspired by the diverse MAP paradigms, fit a mixture of Mean Field approximations. We demonstrate that this positively impacts real-world algorithms that initially relied on mean fields. ]_

_**Comment**_: &emsp;_< >_

#### Probabilistic Temporal Subspace Clustering

Behnam Gholami, Vladimir Pavlovic

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### *Provable Self-Representation Based Outlier Detection in a Union of Subspaces  [PDF](https://arxiv.org/abs/1704.03925)

Chong You, Daniel P. Robinson, René Vidal

_**Abstract**_: &emsp;_[Many computer vision tasks involve processing large amounts of data contaminated by outliers, which need to be detected and rejected. While outlier detection methods based on robust statistics have existed for decades, only recently have methods based on sparse and low-rank representation been developed along with guarantees of correct outlier detection when the inliers lie in one or more low-dimensional subspaces. This paper proposes a new outlier detection method that combines tools from sparse representation with random walks on a graph. By exploiting the property that data points can be expressed as sparse linear combinations of each other, we obtain an asymmetric affinity matrix among data points, which we use to construct a weighted directed graph. By defining a suitable Markov Chain from this graph, we establish a connection between inliers/outliers and essential/inessential states of the Markov chain, which allows us to detect outliers by using random walks. We provide a theoretical analysis that justifies the correctness of our method under geometric and connectivity assumptions. Experimental results on image databases demonstrate its superiority with respect to state-of-the-art sparse and low-rank outlier detection methods. ]_

_**Comment**_: &emsp;_< data contaminated by outliers, outlier detection>_

#### Latent Multi-View Subspace Clustering

Changqing Zhang, Qinghua Hu, Huazhu Fu, Pengfei Zhu, Xiaochun Cao

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### *Learning to Extract Semantic Structure From Documents Using Multimodal Fully Convolutional Neural Networks  [PDF](https://arxiv.org/abs/1706.02337)

Xiao Yang, Ersin Yumer, Paul Asente, Mike Kraley, Daniel Kifer, C. Lee Giles

_**Abstract**_: &emsp;_[We present an end-to-end, multimodal, fully convolutional network for extracting semantic structures from document images. We consider document semantic structure extraction as a pixel-wise segmentation task, and propose a unified model that classifies pixels based not only on their visual appearance, as in the traditional page segmentation task, but also on the content of underlying text. Moreover, we propose an efficient synthetic document generation process that we use to generate pretraining data for our network. Once the network is trained on a large set of synthetic documents, we fine-tune the network on unlabeled real documents using a semi-supervised approach. We systematically study the optimum network architecture and show that both our multimodal approach and the synthetic data pretraining significantly boost the performance. ]_

_**Comment**_: &emsp;_< extracting semantic structures from document images; synthetic document generation for pertraining data, fine-tune using semi-supervised,>_

#### *Age Progression/Regression by Conditional Adversarial Autoencoder  [PDF](https://arxiv.org/abs/1702.08423)

Zhifei Zhang, Yang Song, Hairong Qi

_**Abstract**_: &emsp;_["If I provide you a face image of mine (without telling you the actual age when I took the picture) and a large amount of face images that I crawled (containing labeled faces of different ages but not necessarily paired), can you show me what I would look like when I am 80 or what I was like when I was 5?" The answer is probably a "No." Most existing face aging works attempt to learn the transformation between age groups and thus would require the paired samples as well as the labeled query image. In this paper, we look at the problem from a generative modeling perspective such that no paired samples is required. In addition, given an unlabeled image, the generative model can directly produce the image with desired age attribute. We propose a conditional adversarial autoencoder (CAAE) that learns a face manifold, traversing on which smooth age progression and regression can be realized simultaneously. In CAAE, the face is first mapped to a latent vector through a convolutional encoder, and then the vector is projected to the face manifold conditional on age through a deconvolutional generator. The latent vector preserves personalized face features (i.e., personality) and the age condition controls progression vs. regression. Two adversarial networks are imposed on the encoder and generator, respectively, forcing to generate more photo-realistic faces. Experimental results demonstrate the appealing performance and flexibility of the proposed framework by comparing with the state-of-the-art and ground truth. ]_

_**Comment**_: &emsp;_< generate face images of different age >_

## Oral 3-1A

#### Compact Matrix Factorization With Dependent Subspaces

Viktor Larsson, Carl Olsson
_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### FFTLasso: Large-Scale LASSO in the Fourier Domain  [PDF](https://www.researchgate.net/profile/Adel_Bibi/publication/315765699_FFTLasso_Large-Scale_LASSO_in_the_Fourier_Domain/links/58e325cbaca2722505d16b85/FFTLasso-Large-Scale-LASSO-in-the-Fourier-Domain.pdf)

Adel Bibi, Hani Itani, Bernard Ghanem

_**Abstract**_: &emsp;_[In this paper, we revisit the LASSO sparse representation problem, which has been studied and used in a variety of different areas, ranging from signal processing and information theory to computer vision and machine learning. In the vision community, it found its way into many important applications, including face recognition, tracking, super resolution, image denoising, to name a few. Despite advances in efficient sparse algorithms, solving large-scale LASSO problems remains a challenge. To circumvent this difficulty, people tend to downsample and subsample the problem (e.g. via dimensionality reduction) to maintain a manageable sized LASSO, which usually comes at the cost of losing solution accuracy. This paper proposes a novel circulant reformulation of the LASSO that lifts the problem to a higher dimension, where ADMM can be efficiently applied to its dual form. Because of this lifting, all optimization variables are updated using only basic element-wise operations, the most computationally expensive of which is a 1D FFT. In this way, there is no need for a linear system solver nor matrix-vector multiplication. Since all operations in our FFTLasso method are element-wise, the sub-problems are completely independent and can be trivially parallelized (e.g. on a GPU). The attractive computational properties of FFTLasso are verified by extensive experiments on synthetic and real data and on the face recognition task. They demonstrate that FFTLasso scales much more effectively than a state-of-the-art solver.]_

_**Comment**_: &emsp;_< large-scale LASSO problems, lifted to a higer dim and use ADMM ,only basic element-wise operations, >_

#### On the Global Geometry of Sphere-Constrained Sparse Blind Deconvolution

Yuqian Zhang, Yenson Lau, Han-wen Kuo, Sky Cheung, Abhay Pasupathy, John Wright

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### *Global Optimality in Neural Network Training

Benjamin D. Haeffele, René Vidal

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

# Object Recognition & Scene Understanding 2

## Spotlight 3-1B

#### What Is and What Is Not a Salient Object? Learning Salient Object Detector by Ensembling Linear Exemplar Regressors

Changqun Xia, Jia Li, Xiaowu Chen, Anlin Zheng, Yu Zhang

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Deep Variation-Structured Reinforcement Learning for Visual Relationship and Attribute Detection  [PDF](https://arxiv.org/abs/1703.03054)

Xiaodan Liang, Lisa Lee, Eric P. Xing

_**Abstract**_: &emsp;_[Despite progress in visual perception tasks such as image classification and detection, computers still struggle to understand the interdependency of objects in the scene as a whole, e.g., relations between objects or their attributes. Existing methods often ignore global context cues capturing the interactions among different object instances, and can only recognize a handful of types by exhaustively training individual detectors for all possible relationships. To capture such global interdependency, we propose a deep Variation-structured Reinforcement Learning (VRL) framework to sequentially discover object relationships and attributes in the whole image. First, a directed semantic action graph is built using language priors to provide a rich and compact representation of semantic correlations between object categories, predicates, and attributes. Next, we use a variation-structured traversal over the action graph to construct a small, adaptive action set for each step based on the current state and historical actions. In particular, an ambiguity-aware object mining scheme is used to resolve semantic ambiguity among object categories that the object detector fails to distinguish. We then make sequential predictions using a deep RL framework, incorporating global context cues and semantic embeddings of previously extracted phrases in the state vector. Our experiments on the Visual Relationship Detection (VRD) dataset and the large-scale Visual Genome dataset validate the superiority of VRL, which can achieve significantly better detection results on datasets involving thousands of relationship and attribute types. We also demonstrate that VRL is able to predict unseen types embedded in our action graph by learning correlations on shared graph nodes. ]_

_**Comment**_: &emsp;_< global interdependency( object relationships and attributes)  >_

#### *Modeling Relationships in Referential Expressions With Compositional Modular Networks  [PDF](https://arxiv.org/abs/1611.09978)

Ronghang Hu, Marcus Rohrbach, Jacob Andreas, Trevor Darrell, Kate Saenko

_**Abstract**_: &emsp;_[People often refer to entities in an image in terms of their relationships with other entities. For example, "the black cat sitting under the table" refers to both a "black cat" entity and its relationship with another "table" entity. Understanding these relationships is essential for interpreting and grounding such natural language expressions. Most prior work focuses on either grounding entire referential expressions holistically to one region, or localizing relationships based on a fixed set of categories. In this paper we instead present a modular deep architecture capable of analyzing referential expressions into their component parts, identifying entities and relationships mentioned in the input expression and grounding them all in the scene. We call this approach Compositional Modular Networks (CMNs): a novel architecture that learns linguistic analysis and visual inference end-to-end. Our approach is built around two types of neural modules that inspect local regions and pairwise interactions between regions. We evaluate CMNs on multiple referential expression datasets, outperforming state-of-the-art approaches on all tasks. ]_

_**Comment**_: &emsp;_< learns linguistic analysis and visual inference end-to-end( a modular network),referential expressions, e.g.the black cat sitting under the table>_

#### Counting Everyday Objects in Everyday Scenes  [PDF](https://arxiv.org/abs/1604.03505)

Prithvijit Chattopadhyay, Ramakrishna Vedantam, Ramprasaath R. Selvaraju, Dhruv Batra, Devi Parikh

_**Abstract**_: &emsp;_[We are interested in counting the number of instances of object classes in natural, everyday images. Previous counting approaches tackle the problem in restricted domains such as counting pedestrians in surveillance videos. Counts can also be estimated from outputs of other vision tasks like object detection. In this work, we build dedicated models for counting designed to tackle the large variance in counts, appearances, and scales of objects found in natural scenes. Our approach is inspired by the phenomenon of subitizing - the ability of humans to make quick assessments of counts given a perceptual signal, for small count values. Given a natural scene, we employ a divide and conquer strategy while incorporating context across the scene to adapt the subitizing idea to counting. Our approach offers consistent improvements over numerous baseline approaches for counting on the PASCAL VOC 2007 and COCO datasets. Subsequently, we study how counting can be used to improve object detection. We then show a proof of concept application of our counting methods to the task of Visual Question Answering, by studying the `how many?' questions in the VQA and COCO-QA datasets. ]_

_**Comment**_: &emsp;_<  dedicated models for counting>_

#### *Fully Convolutional Instance-Aware Semantic Segmentation  [PDF](https://arxiv.org/abs/1611.07709)

Yi Li, Haozhi Qi, Jifeng Dai, Xiangyang Ji, Yichen Wei

_**Abstract**_: &emsp;_[We present the first fully convolutional end-to-end solution for instance-aware semantic segmentation task. It inherits all the merits of FCNs for semantic segmentation and instance mask proposal. It performs instance mask prediction and classification jointly. The underlying convolutional representation is fully shared between the two sub-tasks, as well as between all regions of interest. The proposed network is highly integrated and achieves state-of-the-art performance in both accuracy and efficiency. It wins the COCO 2016 segmentation competition by a large margin. Code would be released at ]_

_**Comment**_: &emsp;_< instance-aware semantic segmentation task>_

#### ****Semantic Autoencoder for Zero-Shot Learning  [PDF](https://arxiv.org/abs/1704.08345)

Elyor Kodirov, Tao Xiang, Shaogang Gong

_**Abstract**_: &emsp;_[Existing zero-shot learning (ZSL) models typically learn a projection function from a feature space to a semantic embedding space (e.g.~attribute space). However, such a projection function is only concerned with predicting the training seen class semantic representation (e.g.~attribute prediction) or classification. When applied to test data, which in the context of ZSL contains different (unseen) classes without training data, a ZSL model typically suffers from the project domain shift problem. In this work, we present a novel solution to ZSL based on learning a Semantic AutoEncoder (SAE). Taking the encoder-decoder paradigm, an encoder aims to project a visual feature vector into the semantic space as in the existing ZSL models. However, the decoder exerts an additional constraint, that is, the projection/code must be able to reconstruct the original visual feature. We show that with this additional reconstruction constraint, the learned projection function from the seen classes is able to generalise better to the new unseen classes. Importantly, the encoder and decoder are linear and symmetric which enable us to develop an extremely efficient learning algorithm. Extensive experiments on six benchmark datasets demonstrate that the proposed SAE outperforms significantly the existing ZSL models with the additional benefit of lower computational cost. Furthermore, when the SAE is applied to supervised clustering problem, it also beats the state-of-the-art. ]_

_**Comment**_: &emsp;_<  zero-shot learning; exiting ZSL learn a projection, causing project domain shift problem; with decoder's constraint that projection must be able to reconstruct the original visual feature, thus generalise better to new unseen classes; classification with rejection>_

#### CityPersons: A Diverse Dataset for Pedestrian Detection  [PDF](https://arxiv.org/abs/1702.05693)

Shanshan Zhang, Rodrigo Benenson, Bernt Schiele

_**Abstract**_: &emsp;_[Convnets have enabled significant progress in pedestrian detection recently, but there are still open questions regarding suitable architectures and training data. We revisit CNN design and point out key adaptations, enabling plain FasterRCNN to obtain state-of-the-art results on the Caltech dataset. To achieve further improvement from more and better data, we introduce CityPersons, a new set of person annotations on top of the Cityscapes dataset. The diversity of CityPersons allows us for the first time to train one single CNN model that generalizes well over multiple benchmarks. Moreover, with additional training with CityPersons, we obtain top results using FasterRCNN on Caltech, improving especially for more difficult cases   [PDF](heavy occlusion and small scale) and providing higher localization quality. ]_

_**Comment**_: &emsp;_< pedestrain detection>_

#### GuessWhat?! Visual Object Discovery Through Multi-Modal Dialogue  [PDF](https://arxiv.org/abs/1611.08481)

Harm de Vries, Florian Strub, Sarath Chandar, Olivier Pietquin, Hugo Larochelle, Aaron Courville

_**Abstract**_: &emsp;_[We introduce GuessWhat?!, a two-player guessing game as a testbed for research on the interplay of computer vision and dialogue systems. The goal of the game is to locate an unknown object in a rich image scene by asking a sequence of questions. Higher-level image understanding, like spatial reasoning and language grounding, is required to solve the proposed task. Our key contribution is the collection of a large-scale dataset consisting of 150K human-played games with a total of 800K visual question-answer pairs on 66K images. We explain our design decisions in collecting the dataset and introduce the oracle and questioner tasks that are associated with the two players of the game. We prototyped deep learning models to establish initial baselines of the introduced tasks. ]_

_**Comment**_: &emsp;_<  two-player guessing game >_

## Oral 3-1B

#### *Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-Grained Image Recognition
Jianlong Fu, Heliang Zheng, Tao Mei

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Annotating Object Instances With a Polygon-RNN  [PDF](https://arxiv.org/abs/1704.05548)

LluÃ­s CastrejÃ³n, Kaustav Kundu, Raquel Urtasun, Sanja Fidler

_**Abstract**_: &emsp;_[We propose an approach for semi-automatic annotation of object instances. While most current methods treat object segmentation as a pixel-labeling problem, we here cast it as a polygon prediction task, mimicking how most current datasets have been annotated. In particular, our approach takes as input an image crop and sequentially produces vertices of the polygon outlining the object. This allows a human annotator to interfere at any time and correct a vertex if needed, producing as accurate segmentation as desired by the annotator. We show that our approach speeds up the annotation process by a factor of 4.7 across all classes in Cityscapes, while achieving 78.4% agreement in IoU with original ground-truth, matching the typical agreement between human annotators. For cars, our speed-up factor is 7.3 for an agreement of 82.2%. We further show generalization capabilities of our approach to unseen datasets. ]_

_**Comment**_: &emsp;_< semi-automatic annotation of object instances, treat object seg as a polygon prediction task instead of pixel-labeling prob.>_

#### *Connecting Look and Feel: Associating the Visual and Tactile Properties of Physical Materials  [PDF](https://arxiv.org/abs/1704.03822)

Wenzhen Yuan, Shaoxiong Wang, Siyuan Dong, Edward Adelson

_**Abstract**_: &emsp;_[For machines to interact with the physical world, they must understand the physical properties of objects and materials they encounter. We use fabrics as an example of a deformable material with a rich set of mechanical properties. A thin flexible fabric, when draped, tends to look different from a heavy stiff fabric. It also feels different when touched. Using a collection of 118 fabric sample, we captured color and depth images of draped fabrics along with tactile data from a high resolution touch sensor. We then sought to associate the information from vision and touch by jointly training CNNs across the three modalities. Through the CNN, each input, regardless of the modality, generates an embedding vector that records the fabric's physical property. By comparing the embeddings, our system is able to look at a fabric image and predict how it will feel, and vice versa. We also show that a system jointly trained on vision and touch data can outperform a similar system trained only on visual data when tested purely with visual inputs. ]_

_**Comment**_: &emsp;_< look and feel, predict mutually, jointly training CNNs(color, depth, tactile data)>_

#### *Deep Learning Human Mind for Automated Visual Classification  [PDF](https://arxiv.org/abs/1609.00344)

Concetto Spampinato, Simone Palazzo, Isaak Kavasidis, Daniela Giordano, Nasim Souly, Mubarak Shah

_**Abstract**_: &emsp;_[What if we could effectively read the mind and transfer human visual capabilities to computer vision methods? In this paper, we aim at addressing this question by developing the first visual object classifier driven by human brain signals. In particular, we employ EEG data evoked by visual object stimuli combined with Recurrent Neural Networks (RNN) to learn a discriminative brain activity manifold of visual categories. Afterwards, we train a Convolutional Neural Network (CNN)-based regressor to project images onto the learned manifold, thus effectively allowing machines to employ human brain-based features for automated visual classification. We use a 32-channel EEG to record brain activity of seven subjects while looking at images of 40 ImageNet object classes. The proposed RNN based approach for discriminating object classes using brain signals reaches an average accuracy of about 40%, which outperforms existing methods attempting to learn EEG visual object representations. As for automated object categorization, our human brain-driven approach obtains competitive performance, comparable to those achieved by powerful CNN models, both on ImageNet and CalTech 101, thus demonstrating its classification and generalization capabilities. This gives us a real hope that, indeed, human mind can be read and transferred to machines. ]_

_**Comment**_: &emsp;_< visual object classifier driven by human brain signals, human brain-based features>_

# Poster 3-1

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
Efficient Optimization for Hierarchically-structured Interacting Segments   [PDF](hINTS)
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
One-Shot Video Object Segmentation (PDF, project, code, code)
Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-TaixÃ©, Daniel Cremers, Luc Van Gool
Fast Person Re-Identification via Cross-Camera Semantic Binary Transformation
Jiaxin Chen, Yunhong Wang, Jie Qin, Li Liu, Ling Shao
SPFTN: A Self-Paced Fine-Tuning Network for Segmenting Objects in Weakly Labelled Videos
Dingwen Zhang, Le Yang, Deyu Meng, Dong Xu, Junwei Han

# Machine Learning 4

## Spotlight 4-1A

#### *Hidden Layers in Perceptual Learning
Gad Cohen, Daphna Weinshall

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### *Few-Shot Object Recognition From Machine-Labeled Web Images  [PDF](https://arxiv.org/abs/1612.06152)

Zhongwen Xu, Linchao Zhu, Yi Yang

_**Abstract**_: &emsp;_[With the tremendous advances of Convolutional Neural Networks (ConvNets) on object recognition, we can now obtain reliable enough machine-labeled annotations easily by predictions from off-the-shelf ConvNets. In this work, we present an abstraction memory based framework for few-shot learning, building upon machine-labeled image annotations. Our method takes some large-scale machine-annotated datasets (e.g., OpenImages) as an external memory bank. In the external memory bank, the information is stored in the memory slots with the form of key-value, where image feature is regarded as key and label embedding serves as value. When queried by the few-shot examples, our model selects visually similar data from the external memory bank, and writes the useful information obtained from related external data into another memory bank, i.e., abstraction memory. Long Short-Term Memory (LSTM) controllers and attention mechanisms are utilized to guarantee the data written to the abstraction memory is correlated to the query example. The abstraction memory concentrates information from the external memory bank, so that it makes the few-shot recognition effective. In the experiments, we firstly confirm that our model can learn to conduct few-shot object recognition on clean human-labeled data from ImageNet dataset. Then, we demonstrate that with our model, machine-labeled image annotations are very effective and abundant resources to perform object recognition on novel categories. Experimental results show that our proposed model with machine-labeled annotations achieves great performance, only with a gap of 1% between of the one with human-labeled annotations. ]_

_**Comment**_: &emsp;_<  few-shot learning, machine-labeled>_

#### Hallucinating Very Low-Resolution Unaligned and Noisy Face Images by Transformative Discriminative Autoencoders

Xin Yu, Fatih Porikli

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Are You Smarter Than a Sixth Grader? Textbook Question Answering for Multimodal Machine Comprehension

Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, Hannaneh Hajishirzi

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### *Deep Hashing Network for Unsupervised Domain Adaptation  [PDF](https://arxiv.org/abs/1706.07522)

Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, Sethuraman Panchanathan

_**Abstract**_: &emsp;_[In recent years, deep neural networks have emerged as a dominant machine learning tool for a wide variety of application domains. However, training a deep neural network requires a large amount of labeled data, which is an expensive process in terms of time, labor and human expertise. Domain adaptation or transfer learning algorithms address this challenge by leveraging labeled data in a different, but related source domain, to develop a model for the target domain. Further, the explosive growth of digital data has posed a fundamental challenge concerning its storage and retrieval. Due to its storage and retrieval efficiency, recent years have witnessed a wide application of hashing in a variety of computer vision applications. In this paper, we first introduce a new dataset, Office-Home, to evaluate domain adaptation algorithms. The dataset contains images of a variety of everyday objects from multiple domains. We then propose a novel deep learning framework that can exploit labeled source data and unlabeled target data to learn informative hash codes, to accurately classify unseen target data. To the best of our knowledge, this is the first research effort to exploit the feature learning capabilities of deep neural networks to learn representative hash codes to address the domain adaptation problem. Our extensive empirical studies on multiple transfer tasks corroborate the usefulness of the framework in learning efficient hash codes which outperform existing competitive baselines for unsupervised domain adaptation. ]_

_**Comment**_: &emsp;_< Domain adaptation or transfer learning algorithms that leveraging labeled data in a different but related source domain. office-home: domain adaptation dataset. learn representative hash codes >_

#### **Generalized Deep Image to Image Regression  [PDF](https://arxiv.org/abs/1612.03268)

Venkataraman Santhanam, Vlad I. Morariu, Larry S. Davis

_**Abstract**_: &emsp;_[We present a Deep Convolutional Neural Network architecture which serves as a generic image-to-image regressor that can be trained end-to-end without any further machinery. Our proposed architecture: the Recursively Branched Deconvolutional Network (RBDN) develops a cheap multi-context image representation very early on using an efficient recursive branching scheme with extensive parameter sharing and learnable upsampling. This multi-context representation is subjected to a highly non-linear locality preserving transformation by the remainder of our network comprising of a series of convolutions/deconvolutions without any spatial downsampling. The RBDN architecture is fully convolutional and can handle variable sized images during inference. We provide qualitative/quantitative results on 3 diverse tasks: relighting, denoising and colorization and show that our proposed RBDN architecture obtains comparable results to the state-of-the-art on each of these tasks when used off-the-shelf without any post processing or task-specific architectural modifications. ]_

_**Comment**_: &emsp;_< CNN as generic im-to-im regressor, multi-context representation, recursive branching scheme, no spatial downsampling>_

#### ***Deep Learning With Low Precision by Half-Wave Gaussian Quantization  [PDF](https://arxiv.org/abs/1702.00953)

Zhaowei Cai, Xiaodong He, Jian Sun, Nuno Vasconcelos

_**Abstract**_: &emsp;_[The problem of quantizing the activations of a deep neural network is considered. An examination of the popular binary quantization approach shows that this consists of approximating a classical non-linearity, the hyperbolic tangent, by two functions: a piecewise constant sign function, which is used in feedforward network computations, and a piecewise linear hard tanh function, used in the backpropagation step during network learning. The problem of approximating the ReLU non-linearity, widely used in the recent deep learning literature, is then considered. An half-wave Gaussian quantizer   [PDF](hWGQ) is proposed for forward approximation and shown to have efficient implementation, by exploiting the statistics of of network activations and batch normalization operations commonly used in the literature. To overcome the problem of gradient mismatch, due to the use of different forward and backward approximations, several piece-wise backward approximators are then investigated. The implementation of the resulting quantized network, denoted as HWGQ-Net, is shown to achieve much closer performance to full precision networks, such as AlexNet, ResNet, GoogLeNet and VGG-Net, than previously available low-precision networks, with 1-bit binary weights and 2-bit quantized activations. ]_

_**Comment**_: &emsp;_< quantized network, quantizing the activations of a deep neural network, binary quantization approach(tanh), HWGQ(Relu),>_

#### Creativity: Generating Diverse Questions Using Variational Autoencoders  [PDF](https://arxiv.org/abs/1704.03493)

Unnat Jain, Ziyu Zhang, Alexander G. Schwing

_**Abstract**_: &emsp;_[Generating diverse questions for given images is an important task for computational education, entertainment and AI assistants. Different from many conventional prediction techniques is the need for algorithms to generate a diverse set of plausible questions, which we refer to as "creativity". In this paper we propose a creative algorithm for visual question generation which combines the advantages of variational autoencoders with long short-term memory networks. We demonstrate that our framework is able to generate a large set of varying questions given a single input image. ]_

_**Comment**_: &emsp;_< Generating diverse questions for given images, combine variational AE with LSTM>_

## Oral 4-1A

#### *Geometric Deep Learning on Graphs and Manifolds Using Mixture Model CNNs  [PDF](https://arxiv.org/abs/1611.08402)

Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele RodolÃ , Jan Svoboda, Michael M. Bronstein

_**Abstract**_: &emsp;_[Deep learning has achieved a remarkable performance breakthrough in several fields, most notably in speech recognition, natural language processing, and computer vision. In particular, convolutional neural network (CNN) architectures currently produce state-of-the-art performance on a variety of image analysis tasks such as object detection and recognition. Most of deep learning research has so far focused on dealing with 1D, 2D, or 3D Euclidean-structured data such as acoustic signals, images, or videos. Recently, there has been an increasing interest in geometric deep learning, attempting to generalize deep learning methods to non-Euclidean structured data such as graphs and manifolds, with a variety of applications from the domains of network analysis, computational social science, or computer graphics. In this paper, we propose a unified framework allowing to generalize CNN architectures to non-Euclidean domains (graphs and manifolds) and learn local, stationary, and compositional task-specific features. We show that various non-Euclidean CNN methods previously proposed in the literature can be considered as particular instances of our framework. We test the proposed method on standard tasks from the realms of image-, graph- and 3D shape analysis and show that it consistently outperforms previous approaches. ]_

_**Comment**_: &emsp;_< using CNN to deal with non-Euclidean structured data such as graphs and manifolds>_

#### Full Resolution Image Compression With Recurrent Neural Networks  [PDF](https://arxiv.org/abs/1608.05148)

George Toderici, Damien Vincent, Nick Johnston, Sung Jin Hwang, David Minnen, Joel Shor, Michele Covell

_**Abstract**_: &emsp;_[This paper presents a set of full-resolution lossy image compression methods based on neural networks. Each of the architectures we describe can provide variable compression rates during deployment without requiring retraining of the network: each network need only be trained once. All of our architectures consist of a recurrent neural network (RNN)-based encoder and decoder, a binarizer, and a neural network for entropy coding. We compare RNN types (LSTM, associative LSTM) and introduce a new hybrid of GRU and ResNet. We also study "one-shot" versus additive reconstruction architectures and introduce a new scaled-additive framework. We compare to previous work, showing improvements of 4.3%-8.8% AUC (area under the rate-distortion curve), depending on the perceptual metric used. As far as we know, this is the first neural network architecture that is able to outperform JPEG at image compression across most bitrates on the rate-distortion curve on the Kodak dataset images, with and without the aid of entropy coding. ]_

_**Comment**_: &emsp;_< im compression outperform jpeg using RNN>_

#### **Neural Face Editing With Intrinsic Image Disentangling  [PDF](https://arxiv.org/abs/1704.04131)

Zhixin Shu, Ersin Yumer, Sunil Hadap, Kalyan Sunkavalli, Eli Shechtman, Dimitris Samaras

_**Abstract**_: &emsp;_[Traditional face editing methods often require a number of sophisticated and task specific algorithms to be applied one after the other --- a process that is tedious, fragile, and computationally intensive. In this paper, we propose an end-to-end generative adversarial network that infers a face-specific disentangled representation of intrinsic face properties, including shape (i.e. normals), albedo, and lighting, and an alpha matte. We show that this network can be trained on "in-the-wild" images by incorporating an in-network physically-based image formation module and appropriate loss functions. Our disentangling latent representation allows for semantically relevant edits, where one aspect of facial appearance can be manipulated while keeping orthogonal properties fixed, and we demonstrate its use for a number of facial editing applications. ]_

_**Comment**_: &emsp;_< GAN, disentangling latent representation allows for semantically relevant edits, divide different properties>_

#### **Ubernet: Training a Universal Convolutional Neural Network for Low-, Mid-, and High-Level Vision Using Diverse Datasets and Limited Memory  [PDF](https://arxiv.org/abs/1609.02132)

Iasonas Kokkinos

_**Abstract**_: &emsp;_[In this work we introduce a convolutional neural network (CNN) that jointly handles low-, mid-, and high-level vision tasks in a unified architecture that is trained end-to-end. Such a universal network can act like a `swiss knife' for vision tasks; we call this architecture an UberNet to indicate its overarching nature. We address two main technical challenges that emerge when broadening up the range of tasks handled by a single CNN: (i) training a deep architecture while relying on diverse training sets and (ii) training many (potentially unlimited) tasks with a limited memory budget. Properly addressing these two problems allows us to train accurate predictors for a host of tasks, without compromising accuracy. Through these advances we train in an end-to-end manner a CNN that simultaneously addresses (a) boundary detection (b) normal estimation (c) saliency estimation (d) semantic segmentation (e) human part segmentation (f) semantic boundary detection, (g) region proposal generation and object detection. We obtain competitive performance while jointly addressing all of these tasks in 0.7 seconds per frame on a single GPU. A demonstration of this system can be found at this http URL]_

_**Comment**_: &emsp;_< >_


# Analyzing Humans with 3D Vision

## Spotlight 4-1B

#### 3D Face Morphable Models In-The-Wild

James Booth, Epameinondas Antonakos, Stylianos Ploumpis, George Trigeorgis, Yannis Panagakis, Stefanos Zafeiriou

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### KillingFusion: Non-Rigid 3D Reconstruction Without Correspondences

Miroslava Slavcheva, Maximilian Baust, Daniel Cremers, Slobodan Ilic

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Detailed, Accurate, Human Shape Estimation From Clothed 3D Scan Sequences

Chao Zhang, Sergi Pujades, Michael J. Black, Gerard Pons-Moll

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### POSEidon: Face-From-Depth for Driver Pose Estimation

Guido Borghi, Marco Venturelli, Roberto Vezzani, Rita Cucchiara

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Human Shape From Silhouettes Using Generative HKS Descriptors and Cross-Modal Neural Networks

Endri Dibra, Himanshu Jain, Cengiz Ã–ztireli, Remo Ziegler, Markus Gross

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Parametric T-Spline Face Morphable Model for Detailed Fitting in Shape Subspace

Weilong Peng, Zhiyong Feng, Chao Xu, Yong Su

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### 3D Menagerie: Modeling the 3D Shape and Pose of Animals

Silvia Zuffi, Angjoo Kanazawa, David W. Jacobs, Michael J. Black

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### ***iCaRL: Incremental Classifier and Representation Learning  [PDF](https://arxiv.org/abs/1611.07725)

Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert

_**Abstract**_: &emsp;_[A major open problem on the road to artificial intelligence is the development of incrementally learning systems that learn about more and more concepts over time from a stream of data. In this work, we introduce a new training strategy, iCaRL, that allows learning in such a class-incremental way: only the training data for a small number of classes has to be present at the same time and new classes can be added progressively. iCaRL learns strong classifiers and a data representation simultaneously. This distinguishes it from earlier works that were fundamentally limited to fixed data representations and therefore incompatible with deep learning architectures. We show by experiments on CIFAR-100 and ImageNet ILSVRC 2012 data that iCaRL can learn many classes incrementally over a long period of time where other strategies quickly fail. ]_

_**Comment**_: &emsp;_< a class-incremental way,new classes can be added progressively.>_


# Oral 4-1B

#### Recurrent 3D Pose Sequence Machines

Mude Lin, Liang Lin, Xiaodan Liang, Keze Wang, Hui Cheng

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### **Learning Detailed Face Reconstruction From a Single Image  [PDF](https://arxiv.org/abs/1611.05053)

Elad Richardson, Matan Sela, Roy Or-El, Ron Kimmel

_**Abstract**_: &emsp;_[Reconstructing the detailed geometric structure of a face from a given image is a key to many computer vision and graphics applications, such as motion capture and reenactment. The reconstruction task is challenging as human faces vary extensively when considering expressions, poses, textures, and intrinsic geometries. While many approaches tackle this complexity by using additional data to reconstruct the face of a single subject, extracting facial surface from a single image remains a difficult problem. As a result, single-image based methods can usually provide only a rough estimate of the facial geometry. In contrast, we propose to leverage the power of convolutional neural networks to produce a highly detailed face reconstruction from a single image. For this purpose, we introduce an end-to-end CNN framework which derives the shape in a coarse-to-fine fashion. The proposed architecture is composed of two main blocks, a network that recovers the coarse facial geometry (CoarseNet), followed by a CNN that refines the facial features of that geometry (FineNet). The proposed networks are connected by a novel layer which renders a depth image given a mesh in 3D. Unlike object recognition and detection problems, there are no suitable datasets for training CNNs to perform face geometry reconstruction. Therefore, our training regime begins with a supervised phase, based on synthetic images, followed by an unsupervised phase that uses only unconstrained facial images. The accuracy and robustness of the proposed model is demonstrated by both qualitative and quantitative evaluation tests. ]_

_**Comment**_: &emsp;_< an end-to-end CNN framework which derives the shape in a coarse-to-fine fashion.based on synthetic images>_

#### Thin-Slicing Network: A Deep Structured Model for Pose Estimation in Videos

Jie Song, Limin Wang, Luc Van Gool, Otmar Hilliges

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Dynamic FAUST: Registering Human Bodies in Motion

Federica Bogo, Javier Romero, Gerard Pons-Moll, Michael J. Black

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_


# Poster 4-1

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
Robert Walecki, Ognjen (Oggi) Rudovic, Vladimir Pavlovic, BjÃ¶ern Schuller, Maja Pantic
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

# Object Recognition & Scene Understanding 3

## Spotlight 4-2A

#### *Gaze Embeddings for Zero-Shot Image Classification  [PDF](https://arxiv.org/abs/1611.09309)

Nour Karessli, Zeynep Akata, Bernt Schiele, Andreas Bulling

_**Abstract**_: &emsp;_[Zero-shot image classification using auxiliary information, such as attributes describing discriminative object properties, requires time-consuming annotation by domain experts. We instead propose a method that relies on human gaze as auxiliary information, exploiting that even non-expert users have a natural ability to judge class membership. We present a data collection paradigm that involves a discrimination task to increase the information content obtained from gaze data. Our method extracts discriminative descriptors from the data and learns a compatibility function between image and gaze using three novel gaze embeddings: Gaze Histograms (GH), Gaze Features with Grid (GFG) and Gaze Features with Sequence (GFS). We introduce two new gaze-annotated datasets for fine-grained image classification and show that human gaze data is indeed class discriminative, provides a competitive alternative to expert-annotated attributes, and outperforms other baselines for zero-shot image classification. ]_

_**Comment**_: &emsp;_< human gaze as auxiliary information, how it collected?>_

#### What's in a Question: Using Visual Questions as a Form of Supervision  [PDF](https://arxiv.org/abs/1704.03895)

Siddha Ganju, Olga Russakovsky, Abhinav Gupta

_**Abstract**_: &emsp;_[Collecting fully annotated image datasets is challenging and expensive. Many types of weak supervision have been explored: weak manual annotations, web search results, temporal continuity, ambient sound and others. We focus on one particular unexplored mode: visual questions that are asked about images. The key observation that inspires our work is that the question itself provides useful information about the image (even without the answer being available). For instance, the question "what is the breed of the dog?" informs the AI that the animal in the scene is a dog and that there is only one dog present. We make three contributions: (1) providing an extensive qualitative and quantitative analysis of the information contained in human visual questions, (2) proposing two simple but surprisingly effective modifications to the standard visual question answering models that allow them to make use of weak supervision in the form of unanswered questions associated with images and (3) demonstrating that a simple data augmentation strategy inspired by our insights results in a 7.1% improvement on the standard VQA benchmark. ]_

_**Comment**_: &emsp;_< weak manual annotations: unanswered visual questins>_

#### *Attend to You: Personalized Image Captioning With Context Sequence Memory Networks  [PDF](https://arxiv.org/abs/1704.06485)

Cesc Chunseong Park, Byeongchang Kim, Gunhee Kim

_**Abstract**_: &emsp;_[We address personalization issues of image captioning, which have not been discussed yet in previous research. For a query image, we aim to generate a descriptive sentence, accounting for prior knowledge such as the user's active vocabularies in previous documents. As applications of personalized image captioning, we tackle two post automation tasks: hashtag prediction and post generation, on our newly collected Instagram dataset, consisting of 1.1M posts from 6.3K users. We propose a novel captioning model named Context Sequence Memory Network (CSMN). Its unique updates over previous memory network models include (i) exploiting memory as a repository for multiple types of context information, (ii) appending previously generated words into memory to capture long-term information without suffering from the vanishing gradient problem, and (iii) adopting CNN memory structure to jointly represent nearby ordered memory slots for better context understanding. With quantitative evaluation and user studies via Amazon Mechanical Turk, we show the effectiveness of the three novel features of CSMN and its performance enhancement for personalized image captioning over state-of-the-art captioning models. ]_

_**Comment**_: &emsp;_< Personalized Image Captioning, adopting CNN memory structure?>_

#### Adversarially Tuned Scene Generation  [PDF](https://arxiv.org/abs/1701.00405)

VSR Veeravasarapu, Constantin Rothkopf, Ramesh Visvanathan

_**Abstract**_: &emsp;_[Generalization performance of trained computer vision systems that use computer graphics (CG) generated data is not yet effective due to the concept of 'domain-shift' between virtual and real data. Although simulated data augmented with a few real world samples has been shown to mitigate domain shift and improve transferability of trained models, guiding or bootstrapping the virtual data generation with the distributions learnt from target real world domain is desired, especially in the fields where annotating even few real images is laborious (such as semantic labeling, and intrinsic images etc.). In order to address this problem in an unsupervised manner, our work combines recent advances in CG (which aims to generate stochastic scene layouts coupled with large collections of 3D object models) and generative adversarial training (which aims train generative models by measuring discrepancy between generated and real data in terms of their separability in the space of a deep discriminatively-trained classifier). Our method uses iterative estimation of the posterior density of prior distributions for a generative graphical model. This is done within a rejection sampling framework. Initially, we assume uniform distributions as priors on the parameters of a scene described by a generative graphical model. As iterations proceed the prior distributions get updated to distributions that are closer to the (unknown) distributions of target data. We demonstrate the utility of adversarially tuned scene generation on two real-world benchmark datasets (CityScapes and CamVid) for traffic scene semantic labeling with a deep convolutional net (DeepLab). We realized performance improvements by 2.28 and 3.14 points (using the IoU metric) between the DeepLab models trained on simulated sets prepared from the scene generation models before and after tuning to CityScapes and CamVid respectively. ]_

_**Comment**_: &emsp;_< domain-shift' between virtual and real data, >_

#### ***Residual Attention Network for Image Classification  [PDF](https://arxiv.org/abs/1704.06904)

Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang

_**Abstract**_: &emsp;_[In this work, we propose "Residual Attention Network", a convolutional neural network using attention mechanism which can incorporate with state-of-art feed forward network architecture in an end-to-end training fashion. Our Residual Attention Network is built by stacking Attention Modules which generate attention-aware features. The attention-aware features from different modules change adaptively as layers going deeper. Inside each Attention Module, bottom-up top-down feedforward structure is used to unfold the feedforward and feedback attention process into a single feedforward process. Importantly, we propose attention residual learning to train very deep Residual Attention Networks which can be easily scaled up to hundreds of layers. Extensive analyses are conducted on CIFAR-10 and CIFAR-100 datasets to verify the effectiveness of every module mentioned above. Our Residual Attention Network achieves state-of-the-art object recognition performance on three benchmark datasets including CIFAR-10 (3.90% error), CIFAR-100 (20.45% error) and ImageNet (4.8% single model and single crop, top-5 error). Note that, our method achieves 0.6% top-1 accuracy improvement with 46% trunk depth and 69% forward FLOPs comparing to ResNet-200. The experiment also demonstrates that our network is robust against noisy labels. ]_

_**Comment**_: &emsp;_< based on CNN, by stacking Attention Modules which generate attention-aware features>_

#### **Not All Pixels Are Equal: Difficulty-Aware Semantic Segmentation via Deep Layer Cascade  [PDF](https://arxiv.org/abs/1704.01344)

Xiaoxiao Li, Ziwei Liu, Ping Luo, Chen Change Loy, Xiaoou Tang

_**Abstract**_: &emsp;_[We propose a novel deep layer cascade (LC) method to improve the accuracy and speed of semantic segmentation. Unlike the conventional model cascade (MC) that is composed of multiple independent models, LC treats a single deep model as a cascade of several sub-models. Earlier sub-models are trained to handle easy and confident regions, and they progressively feed-forward harder regions to the next sub-model for processing. Convolutions are only calculated on these regions to reduce computations. The proposed method possesses several advantages. First, LC classifies most of the easy regions in the shallow stage and makes deeper stage focuses on a few hard regions. Such an adaptive and 'difficulty-aware' learning improves segmentation performance. Second, LC accelerates both training and testing of deep network thanks to early decisions in the shallow stage. Third, in comparison to MC, LC is an end-to-end trainable framework, allowing joint learning of all sub-models. We evaluate our method on PASCAL VOC and Cityscapes datasets, achieving state-of-the-art performance and fast speed. ]_

_**Comment**_: &emsp;_< reats a single deep model as a cascade of several sub-models, LC classifies most of the easy regions in the shallow stage>_

#### Learning Non-Maximum Suppression  [PDF](https://arxiv.org/abs/1705.02950)

Jan Hosang, Rodrigo Benenson, Bernt Schiele

_**Abstract**_: &emsp;_[Object detectors have hugely profited from moving towards an end-to-end learning paradigm: proposals, features, and the classifier becoming one neural network improved results two-fold on general object detection. One indispensable component is non-maximum suppression (NMS), a post-processing algorithm responsible for merging all detections that belong to the same object. The de facto standard NMS algorithm is still fully hand-crafted, suspiciously simple, and -- being based on greedy clustering with a fixed distance threshold -- forces a trade-off between recall and precision. We propose a new network architecture designed to perform NMS, using only boxes and their score. We report experiments for person detection on PETS and for general object categories on the COCO dataset. Our approach shows promise providing improved localization and occlusion handling. ]_

_**Comment**_: &emsp;_< object detection>_

#### The Amazing Mysteries of the Gutter: Drawing Inferences Between Panels in Comic Book Narratives  [PDF](https://arxiv.org/abs/1611.05118)

Mohit Iyyer, Varun Manjunatha, Anupam Guha, Yogarshi Vyas, Jordan Boyd-Graber, Hal Daumé III, Larry S. Davis

_**Abstract**_: &emsp;_[Visual narrative is often a combination of explicit information and judicious omissions, relying on the viewer to supply missing details. In comics, most movements in time and space are hidden in the "gutters" between panels. To follow the story, readers logically connect panels together by inferring unseen actions through a process called "closure". While computers can now describe what is explicitly depicted in natural images, in this paper we examine whether they can understand the closure-driven narratives conveyed by stylized artwork and dialogue in comic book panels. We construct a dataset, COMICS, that consists of over 1.2 million panels (120 GB) paired with automatic textbox transcriptions. An in-depth analysis of COMICS demonstrates that neither text nor image alone can tell a comic book story, so a computer must understand both modalities to keep up with the plot. We introduce three cloze-style tasks that ask models to predict narrative and character-centric aspects of a panel given n preceding panels as context. Various deep neural architectures underperform human baselines on these tasks, suggesting that COMICS contains fundamental challenges for both vision and language.]_

_**Comment**_: &emsp;_< comic book panels, a new dataset COMICS with text and image>_


## Oral 4-2A

#### *Object Region Mining With Adversarial Erasing: A Simple Classification to Semantic Segmentation Approach  [PDF](https://arxiv.org/abs/1703.08448)

Yunchao Wei, Jiashi Feng, Xiaodan Liang, Ming-Ming Cheng, Yao Zhao, Shuicheng Yan

_**Abstract**_: &emsp;_[ We investigate a principle way to progressively mine discriminative object regions using classification networks to address the weakly-supervised semantic segmentation problems. Classification networks are only responsive to small and sparse discriminative regions from the object of interest, which deviates from the requirement of the segmentation task that needs to localize dense, interior and integral regions for pixel-wise inference. To mitigate this gap, we propose a new adversarial erasing approach for localizing and expanding object regions progressively. Starting with a single small object region, our proposed approach drives the classification network to sequentially discover new and complement object regions by erasing the current mined regions in an adversarial manner. These localized regions eventually constitute a dense and complete object region for learning semantic segmentation. To further enhance the quality of the discovered regions by adversarial erasing, an online prohibitive segmentation learning approach is developed to collaborate with adversarial erasing by providing auxiliary segmentation supervision modulated by the more reliable classification scores. Despite its apparent simplicity, the proposed approach achieves 55.0% and 55.7% mean Intersection-over-Union (mIoU) scores on PASCAL VOC 2012 val and test sets, which are the new state-of-the-arts. ]_

_**Comment**_: &emsp;_< propose a new adversarial erasing approach using classification network, Starting with a single small object region, our proposed approach drives the classification network to sequentially discover new and complement object regions by erasing the current mined regions in an adversarial manner>_

#### *Fine-Grained Recognition as HSnet Search for Informative Image Parts  [PDF](http://web.engr.oregonstate.edu/~sinisa/research/publications/cvpr17_lstmsearch.pdf)

Michael Lam, Behrooz Mahasseni, Sinisa Todorovic

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< fine-grained im classfication, huristic func& successor func unified via a LSTM. How CNN  is used ?>_

#### *G2DeNet: Global Gaussian Distribution Embedding Network and Its Application to Visual Recognition

Qilong Wang, Peihua Li, Lei Zhang

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### *YOLO9000: Better, Faster, Stronger  [PDF](https://arxiv.org/abs/1612.08242)

Joseph Redmon, Ali Farhadi

_**Abstract**_: &emsp;_[We introduce YOLO9000, a state-of-the-art, real-time object detection system that can detect over 9000 object categories. First we propose various improvements to the YOLO detection method, both novel and drawn from prior work. The improved model, YOLOv2, is state-of-the-art on standard detection tasks like PASCAL VOC and COCO. At 67 FPS, YOLOv2 gets 76.8 mAP on VOC 2007. At 40 FPS, YOLOv2 gets 78.6 mAP, outperforming state-of-the-art methods like Faster RCNN with ResNet and SSD while still running significantly faster. Finally we propose a method to jointly train on object detection and classification. Using this method we train YOLO9000 simultaneously on the COCO detection dataset and the ImageNet classification dataset. Our joint training allows YOLO9000 to predict detections for object classes that don't have labelled detection data. We validate our approach on the ImageNet detection task. YOLO9000 gets 19.7 mAP on the ImageNet detection validation set despite only having detection data for 44 of the 200 classes. On the 156 classes not in COCO, YOLO9000 gets 16.0 mAP. But YOLO can detect more than just 200 classes; it predicts detections for more than 9000 different object categories. And it still runs in real-time. ]_

_**Comment**_: &emsp;_< object detection system that can detect over 9000 object categories, a method to jointly train on object detection and classification, how is it jointly trained?>_


# Machine Learning for 3D Vision

## Spotlight 4-2B

#### Multi-View 3D Object Detection Network for Autonomous Driving

Xiaozhi Chen, Huimin Ma, Ji Wan, Bo Li, Tian Xia

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### UltraStereo: Efficient Learning-Based Matching for Active Stereo Systems

Sean Ryan Fanello, Julien Valentin, Christoph Rhemann, Adarsh Kowdle, Vladimir Tankovich, Philip Davidson, Shahram Izadi

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Shape Completion Using 3D-Encoder-Predictor CNNs and Shape Synthesis  [PDF](https://arxiv.org/abs/1612.00101)

Angela Dai, Charles Ruizhongtai Qi, Matthias NieÃŸner

_**Abstract**_: &emsp;_[We introduce a data-driven approach to complete partial 3D shapes through a combination of volumetric deep neural networks and 3D shape synthesis. From a partially-scanned input shape, our method first infers a low-resolution -- but complete -- output. To this end, we introduce a 3D-Encoder-Predictor Network (3D-EPN) which is composed of 3D convolutional layers. The network is trained to predict and fill in missing data, and operates on an implicit surface representation that encodes both known and unknown space. This allows us to predict global structure in unknown areas at high accuracy. We then correlate these intermediary results with 3D geometry from a shape database at test time. In a final pass, we propose a patch-based 3D shape synthesis method that imposes the 3D geometry from these retrieved shapes as constraints on the coarsely-completed mesh. This synthesis process enables us to reconstruct fine-scale detail and generate high-resolution output while respecting the global mesh structure obtained by the 3D-EPN. Although our 3D-EPN outperforms state-of-the-art completion method, the main contribution in our work lies in the combination of a data-driven shape predictor and analytic 3D shape synthesis. In our results, we show extensive evaluations on a newly-introduced shape completion benchmark for both real-world and synthetic data. ]_

_**Comment**_: &emsp;_< >_

#### Geometric Loss Functions for Camera Pose Regression With Deep Learning

Alex Kendall, Roberto Cipolla

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### CNN-SLAM: Real-Time Dense Monocular SLAM With Learned Depth Prediction

Keisuke Tateno, Federico Tombari, Iro Laina, Nassir Navab

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### *Learning From Noisy Large-Scale Datasets With Minimal Supervision  [PDF](https://arxiv.org/abs/1701.01619)

Andreas Veit, Neil Alldrin, Gal Chechik, Ivan Krasin, Abhinav Gupta, Serge Belongie

_**Abstract**_: &emsp;_[We present an approach to effectively use millions of images with noisy annotations in conjunction with a small subset of cleanly-annotated images to learn powerful image representations. One common approach to combine clean and noisy data is to first pre-train a network using the large noisy dataset and then fine-tune with the clean dataset. We show this approach does not fully leverage the information contained in the clean set. Thus, we demonstrate how to use the clean annotations to reduce the noise in the large dataset before fine-tuning the network using both the clean set and the full set with reduced noise. The approach comprises a multi-task network that jointly learns to clean noisy annotations and to accurately classify images. We evaluate our approach on the recently released Open Images dataset, containing ~9 million images, multiple annotations per image and over 6000 unique classes. For the small clean set of annotations we use a quarter of the validation set with ~40k images. Our results demonstrate that the proposed approach clearly outperforms direct fine-tuning across all major categories of classes in the Open Image dataset. Further, our approach is particularly effective for a large number of classes with wide range of noise in annotations (20-80% false positive annotations). ]_

_**Comment**_: &emsp;_< in conjunction with a small subset of cleanly-annotated images, >_

#### *SyncSpecCNN: Synchronized Spectral CNN for 3D Shape Segmentation  [PDF](https://arxiv.org/abs/1612.00606)

Li Yi, Hao Su, Xingwen Guo, Leonidas J. Guibas

_**Abstract**_: &emsp;_[In this paper, we study the problem of semantic annotation on 3D models that are represented as shape graphs. A functional view is taken to represent localized information on graphs, so that annotations such as part segment or keypoint are nothing but 0-1 indicator vertex functions. Compared with images that are 2D grids, shape graphs are irregular and non-isomorphic data structures. To enable the prediction of vertex functions on them by convolutional neural networks, we resort to spectral CNN method that enables weight sharing by parameterizing kernels in the spectral domain spanned by graph laplacian eigenbases. Under this setting, our network, named SyncSpecCNN, strive to overcome two key challenges: how to share coefficients and conduct multi-scale analysis in different parts of the graph for a single shape, and how to share information across related but different shapes that may be represented by very different graphs. Towards these goals, we introduce a spectral parameterization of dilated convolutional kernels and a spectral transformer network. Experimentally we tested our SyncSpecCNN on various tasks, including 3D shape part segmentation and 3D keypoint prediction. State-of-the-art performance has been achieved on all benchmark datasets. ]_

_**Comment**_: &emsp;_< use  CNN to deal with other data>_

#### *Non-Local Deep Features for Salient Object Detection

Zhiming Luo, Akshaya Mishra, Andrew Achkar, Justin Eichel, Shaozi Li, Pierre-Marc Jodoin

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_


## Oral 4-2B

#### Unsupervised Monocular Depth Estimation With Left-Right Consistency

Clément Godard, Oisin Mac Aodha, Gabriel J. Brostow

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### Unsupervised Learning of Depth and Ego-Motion From Video

Tinghui Zhou, Matthew Brown, Noah Snavely, David G. Lowe

_**Abstract**_: &emsp;_[]_

_**Comment**_: &emsp;_< >_

#### OctNet: Learning Deep 3D Representations at High Resolutions  [PDF](https://arxiv.org/abs/1611.05009)

Gernot Riegler, Ali Osman Ulusoy, Andreas Geiger

_**Abstract**_: &emsp;_[We present OctNet, a representation for deep learning with sparse 3D data. In contrast to existing models, our representation enables 3D convolutional networks which are both deep and high resolution. Towards this goal, we exploit the sparsity in the input data to hierarchically partition the space using a set of unbalanced octrees where each leaf node stores a pooled feature representation. This allows to focus memory allocation and computation to the relevant dense regions and enables deeper networks without compromising resolution. We demonstrate the utility of our OctNet representation by analyzing the impact of resolution on several 3D tasks including 3D object classification, orientation estimation and point cloud labeling. ]_

_**Comment**_: &emsp;_< what is 3D CNN?>_

#### 3D Shape Segmentation With Projective Convolutional Networks  [PDF](https://arxiv.org/abs/1612.02808)

Evangelos Kalogerakis, Melinos Averkiou, Subhransu Maji, Siddhartha Chaudhuri

_**Abstract**_: &emsp;_[This paper introduces a deep architecture for segmenting 3D objects into their labeled semantic parts. Our architecture combines image-based Fully Convolutional Networks (FCNs) and surface-based Conditional Random Fields (CRFs) to yield coherent segmentations of 3D shapes. The image-based FCNs are used for efficient view-based reasoning about 3D object parts. Through a special projection layer, FCN outputs are effectively aggregated across multiple views and scales, then are projected onto the 3D object surfaces. Finally, a surface-based CRF combines the projected outputs with geometric consistency cues to yield coherent segmentations. The whole architecture (multi-view FCNs and CRF) is trained end-to-end. Our approach significantly outperforms the existing state-of-the-art methods in the currently largest segmentation benchmark (ShapeNet). Finally, we demonstrate promising segmentation results on noisy 3D shapes acquired from consumer-grade depth cameras. ]_

_**Comment**_: &emsp;_< >_


# Poster 4-2

## 3D Computer Vision
#### SGM-Nets: Semi-Global Matching With Neural Networks
Akihito Seki, Marc Pollefeys
#### Stereo-Based 3D Reconstruction of Dynamic Fluid Surfaces by Global Optimization
Yiming Qian, Minglun Gong, Yee-Hong Yang
#### Fine-To-Coarse Global Registration of RGB-D Scans
Maciej Halber, Thomas Funkhouser
#### Analyzing Computer Vision Data - The Good, the Bad and the Ugly
Oliver Zendel, Katrin Honauer, Markus Murschitz, Martin Humenberger, Gustavo FernÃ¡ndez DomÃ­nguez
#### Product Manifold Filter: Non-Rigid Shape Correspondence via Kernel Density Estimation in the Product Space
Matthias Vestner, Roee Litman, Emanuele RodolÃ , Alex Bronstein, Daniel Cremers
#### Unsupervised Vanishing Point Detection and Camera Calibration From a Single Manhattan Image With Radial Distortion
Michel Antunes, JoÃ£o P. Barreto, Djamila Aouada, BjÃ¶rn Ottersten
#### Toroidal Constraints for Two-Point Localization Under High Outlier Ratios
Federico Camposeco, Torsten Sattler, Andrea Cohen, Andreas Geiger, Marc Pollefeys
#### 4D Light Field Superpixel and Segmentation
Hao Zhu, Qi Zhang, Qing Wang
#### Exploiting Symmetry and/or Manhattan Properties for 3D Object Structure Estimation From Single and Multiple Images
Yuan Gao, Alan L. Yuille
## Analyzing Humans in Images
#### Binary Coding for Partial Action Analysis With Limited Observation Ratios
Jie Qin, Li Liu, Ling Shao, Bingbing Ni, Chen Chen, Fumin Shen, Yunhong Wang
#### SphereFace: Deep Hypersphere Embedding for Face Recognition
Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, Le Song
#### IRINA: Iris Recognition (Even) in Inaccurately Segmented Data
Hugo ProenÃ§a, JoÃ£o C. Neves
#### Look Into Person: Self-Supervised Structure-Sensitive Learning and a New Benchmark for Human Parsing
Ke Gong, Xiaodan Liang, Dongyu Zhang, Xiaohui Shen, Liang Lin
#### Action Unit Detection With Region Adaptation, Multi-Labeling Learning and Optimal Temporal Fusing
Wei Li, Farnaz Abtahi, Zhigang Zhu
#### See the Forest for the Trees: Joint Spatial and Temporal Recurrent Neural Networks for Video-Based Person Re-Identification
Zhen Zhou, Yan Huang, Wei Wang, Liang Wang, Tieniu Tan
#### Joint Intensity and Spatial Metric Learning for Robust Gait Recognition
Yasushi Makihara, Atsuyuki Suzuki, Daigo Muramatsu, Xiang Li, Yasushi Yagi
#### Pose-Aware Person Recognition
Vijay Kumar, Anoop Namboodiri, Manohar Paluri, C. V. Jawahar
#### Not Afraid of the Dark: NIR-VIS Face Recognition via Cross-Spectral Hallucination and Low-Rank Embedding
José Lezama, Qiang Qiu, Guillermo Sapiro
## Applications
#### Jointly Learning Energy Expenditures and Activities Using Egocentric Multimodal Signals
Katsuyuki Nakamura, Serena Yeung, Alexandre Alahi, Li Fei-Fei
#### Binarized Mode Seeking for Scalable Visual Pattern Discovery
Wei Zhang, Xiaochun Cao, Rui Wang, Yuanfang Guo, Zhineng Chen
#### Scribbler: Controlling Deep Image Synthesis With Sketch and Color
Patsorn Sangkloy, Jingwan Lu, Chen Fang, Fisher Yu, James Hays
## Biomedical Image/Video Analysis
#### Multi-Way Multi-Level Kernel Modeling for Neuroimaging Classification
Lifang He, Chun-Ta Lu, Hao Ding, Shen Wang, Linlin Shen, Philip S. Yu, Ann B. Ragin
#### WSISA: Making Survival Prediction From Whole Slide Histopathological Images
Xinliang Zhu, Jiawen Yao, Feiyun Zhu, Junzhou Huang
## Computational Photography
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
Optical Flow Requires Multiple Strategies (but Only One Network)
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
