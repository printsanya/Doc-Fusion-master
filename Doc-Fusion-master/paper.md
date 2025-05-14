# Review Paper

## Abstract
Convolutional Neural Networks (CNNs) have become a cornerstone of deep learning, particularly in image processing tasks such as image classification, object detection, and semantic segmentation, owing to their ability to effectively learn spatial features through convolutional layers. However, CNNs face challenges related to computational efficiency, adaptability to varying data complexities, and the risk of overfitting, especially with limited datasets. Neural Architecture Search (NAS) offers a potential solution by optimizing network architectures but is often resource-intensive. To address these limitations, research efforts are directed towards developing more efficient and adaptable CNN models. One such approach involves dynamically expanding network architectures during training, inspired by neurogenesis, to overcome over-parameterization and enhance adaptability. Studies explore Self-Expanding Convolutional Neural Networks (SECNNs) that dynamically determine the optimal model size based on the task and leverage techniques like deeper convolutional layers, batch normalization, and dropout regularization to improve performance and generalization, especially in small-scale image classification tasks. Furthermore, research investigates the use of feedback alignment (FF) trained CNNs as an alternative to backpropagation, exploring their potential in neuromorphic hardware and unsupervised learning. These advancements aim to create adaptable, efficient CNN models for diverse vision-related tasks and improve our understanding of neuronal information processing in both biological and artificial systems.


## Introduction
Convolutional Neural Networks (CNNs) have become a cornerstone of deep learning, particularly in processing grid-like data such as images, revolutionizing fields like image classification, object detection, and semantic segmentation. Their efficacy arises from the ability to learn spatial features through convolutional layers, which employ filters to capture local patterns, and pooling layers, which reduce dimensionality and focus on salient aspects. However, traditional CNNs with fixed architectures face challenges in computational efficiency and adaptability across varying data complexities. Moreover, achieving high accuracy can be difficult with small image resolutions (such as those in CIFAR-10) due to limited information and the risk of overfitting, necessitating effective regularization techniques and careful architectural design.

To address these limitations, several approaches have emerged. Neural Architecture Search (NAS) aims to optimize network architectures for specific tasks, but its resource-intensive nature, requiring the training of multiple candidate models, poses a significant drawback. Self-Expanding Neural Networks (SENNs) offer an alternative by dynamically adding neurons and layers during training, inspired by neurogenesis, to overcome over-parameterization. 

This work explores enhanced CNN architectures and the application of self-expanding principles to CNNs. One approach involves integrating deeper convolutional blocks with batch normalization and dropout to improve feature extraction and mitigate overfitting, particularly for small-scale image classification tasks. Complementary research focuses on developing a Self Expanding Convolutional Neural Network (SECNN) that dynamically adjusts its size based on the task at hand, leveraging a natural expansion score to guide model growth without requiring retraining. These approaches represent significant steps toward creating more adaptable and efficient CNN models for diverse vision-related tasks.


## Litrature Review
Deep learning techniques have demonstrated significant potential in various applications. Cheng et al. [1] explored deep learning for visual defect detection, specifically addressing the challenges posed by noisy and imbalanced data. Another study by Cheng et al. [2] focused on estimating energy and time usage in 3D printing using a multimodal neural network.

Graph Neural Networks have emerged as powerful tools for processing graph-structured data. Veličković et al. [3] introduced Graph Attention Networks, a novel architecture leveraging attention mechanisms to weigh the importance of different neighbor nodes. Hamilton et al. [4] investigated inductive representation learning on large graphs, enabling generalization to unseen nodes and graphs.

Fuzzy rough sets have been integrated with machine learning techniques to enhance decision-making. Xing et al. [5] proposed a weighted fuzzy rough sets-based tri-training approach, applying it to medical diagnosis. Gao et al. [6] explored parameterized maximum-entropy-based three-way approximate attribute reduction.

In the realm of speech and emotion recognition, Livingstone and Russo [7] introduced the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), a valuable resource for training and evaluating models. Heigold et al. [8] presented an end-to-end text-dependent speaker verification system.

Recurrent Neural Networks, particularly Long Short-Term Memory (LSTM) networks, have been widely used for sequence modeling. Hochreiter and Schmidhuber [9] introduced the LSTM architecture to address the vanishing gradient problem in traditional recurrent networks. Gers et al. [10] further enhanced LSTM networks by incorporating a "forget gate," enabling the network to selectively forget irrelevant information.

Convolutional Neural Networks (CNNs) have achieved remarkable success in image classification. Ciresan et al. [11] proposed multi-column deep neural networks for image classification. Ciresan et al. [12] also explored the use of deep neural networks for mitosis detection in breast cancer histology images. Further research by Ciresan et al. [13] focused on developing flexible, high-performance CNNs for image classification. Ciresan et al. [14] investigated convolutional neural network committees for handwritten character classification. Egmont-Petersen et al. [15] provided a review of image processing techniques using neural networks. Farabet et al. [16] explored hardware-accelerated CNNs for synthetic vision systems.

Techniques like Restricted Boltzmann Machines (RBMs) and dropout have been used to improve the training and generalization of neural networks. Hinton [17] provided a practical guide to training RBMs. Hinton et al. [18] proposed a method to improve neural networks by preventing co-adaptation of feature detectors through dropout.

CNNs have also been extended to handle 3D data and video. Ji et al. [19] developed 3D convolutional neural networks for human action recognition. Karpathy et al. [20] explored large-scale video classification with CNNs. Krizhevsky et al. [21] achieved significant success in ImageNet classification using deep CNNs.

Early work by LeCun et al. [22] demonstrated the application of backpropagation to handwritten zip code recognition. LeCun et al. [23] further explored gradient-based learning for document recognition. Nebauer [24] evaluated CNNs for visual recognition. Simard et al. [25] presented best practices for applying CNNs to visual document analysis. Srivastava [26] investigated the use of dropout to improve neural networks. Szarvas et al. [27] explored pedestrian detection using CNNs. Szegedy et al. [28] investigated deep neural networks for object detection. Tivive and Bouzerdoum [29] introduced a new class of CNNs called SiConNets and applied them to face detection. Zeiler and Fergus [30] proposed stochastic pooling for regularization of deep CNNs. Zeiler and Fergus [31] also focused on visualizing and understanding CNNs.

Scattering transforms offer an alternative approach to feature extraction, providing invariance to certain transformations. Andén and Mallat [32] introduced the deep scattering spectrum. Bruna and Mallat [33] developed invariant scattering convolution networks. Estrach [35] discussed scattering representations for recognition. Mallat [39] explored group invariant scattering and further provided insights into understanding deep CNNs [40]. Kaiser [36] offers a guide to wavelets.

Support Vector Machines (SVMs) represent a different class of machine learning algorithms. Cortes and Vapnik [34] introduced support-vector networks. Boser et al. [37] explored handwritten digit recognition with a back-propagation network. LeCun et al. [38] provided an overview of deep learning.


## Methodology
The methodologies investigated focus on the development and analysis of Convolutional Neural Networks (CNNs). A key aspect involves creating dynamically expanding CNN architectures, utilizing an expansion criterion based on a natural expansion score to trigger model expansion. The research also explores the fundamental concepts of CNNs, highlighting their ability to exploit knowledge of specific input types, leading to simpler network architectures suitable for image analysis tasks. Further analysis delves into the properties of CNNs, introducing the scattering transform as a simplified model to understand CNN operations. This approach examines feature transformation built upon wavelet transforms, which separate variations at different scales. Additionally, the research includes the development of a novel technique for labeling positive and negative datasets, followed by a detailed implementation of the FF algorithm. Overall, the methodologies encompass both practical development of dynamic CNN architectures and theoretical analysis of their underlying principles.


## Convolutional layer

Convolutional layers are vital to CNNs, utilizing learnable kernels that are small in spatial dimensionality but extend through the input depth. These kernels convolve across the input's spatial dimensions, producing 2D activation maps. The network learns kernels that activate upon detecting specific features at spatial positions. Each kernel has a corresponding activation map, and these maps stack to form the convolutional layer's output volume. This layer reduces model complexity by connecting neurons to only a small region of the input volume, known as the receptive field. Optimizations occur through hyperparameters like depth, stride, and zero-padding.

## Convolutional layer

Convolutional layers are vital to CNNs, utilizing learnable kernels that are small in spatial dimensionality but extend through the input depth. These kernels convolve across the input's spatial dimensions, producing 2D activation maps. The network learns kernels that activate upon detecting specific features at spatial positions. Each kernel has a corresponding activation map, and these maps stack to form the convolutional layer's output volume. This layer reduces model complexity by connecting neurons to only a small region of the input volume, known as the receptive field. Optimizations occur through hyperparameters like depth, stride, and zero-padding.

## CNN architecture

CNNs are designed to process image inputs, featuring neurons organized in three dimensions: height, width, and depth. The depth represents the third dimension of an activation volume, not the number of layers. Unlike standard ANNs, neurons in a CNN layer connect to a small region of the preceding layer. For instance, a 64x64x3 input volume is condensed into a 1x1xn output layer, where n represents the number of classes.

## CNN architecture

CNNs are designed to process image inputs, featuring neurons organized in three dimensions: height, width, and depth. The depth represents the third dimension of an activation volume, not the number of layers. Unlike standard ANNs, neurons in a CNN layer connect to a small region of the preceding layer. For instance, a 64x64x3 input volume is condensed into a 1x1xn output layer, where n represents the number of classes.

## Convolutional Neural Networks

CNNs use convolutional filters and non-linearities to process inputs, forming a hierarchical architecture. Each layer is computed using a linear operator and a non-linearity, with the linear operator often being a convolution. The optimization of CNNs is non-convex, and weights are typically learned through stochastic gradient descent using backpropagation.

## General Convolutional Neural Network Architectures

The scattering transform offers a simplified view of CNNs, but it suffers from high variance and information loss due to single-channel convolutions. Analyzing general CNN architectures requires allowing for channel combinations, achieved by adapting filter weights similar to deep learning models and using contractions along adaptive groups of local symmetries.


## Results
The experimental results, derived from multiple trials on the CIFAR-10 dataset, demonstrate the performance of the proposed model. Across five trials with identical hyperparameters, the model achieved a mean highest validation accuracy of 84.1%. Specifically, validation accuracy reached 70% with a mean parameter count of 11,360, while reaching 80% required a mean of 33,655.2 parameters. The number of parameters required to achieve the highest validation accuracy varied across trials, with a mean of 62,698.4 parameters. Furthermore, the capacity of Feedforward trained Convolutional Neural Networks (CNNs) to implement Class Activation Maps, a method used in explainable AI, was also demonstrated. Optimal hyperparameter configurations were identified through a search process using validation data.


## Conclusion
Convolutional Neural Networks (CNNs) are established as a powerful tool for image analysis, exploiting specific input knowledge to simplify network architecture. Several studies focus on enhancing CNN architectures for image classification tasks, particularly using the CIFAR-10 dataset. These enhancements include deeper convolutional layers for complex feature extraction, batch normalization for training stabilization, and dropout layers to mitigate overfitting. Results from these enhanced architectures, coupled with techniques of balancing depth, feature extraction capabilities, and regularization techniques, showcase a test accuracy of approximately 84% representing a significant improvement over standard CNNs, and highlights the potential for improved robustness and generalization across diverse image categories. One study introduces a Self Expanding Convolutional Neural Network, a dynamically expanding architecture that uses the natural expansion score to optimize model growth, offering a computationally efficient solution to dynamically determine an optimal architecture for vision tasks without retraining. The work on Feedforward trained CNNs shows potential to address real-world computer vision problems, with the possibility of class activation maps to provide insight into neuronal information processing. Future research should focus on extending these architectures to more complex datasets like CIFAR-100 and ImageNet, exploring transfer learning techniques for larger, higher-resolution datasets, and adapting models for domain-specific tasks. Furthermore, investigating the application of Feedforward trained CNNs in neuromorphic hardware and unsupervised learning contexts warrants further attention.


## References
[1] Cheng, Qisen, and Shuhui Qu, and Janghwan Lee. "Deep Learning Based Visual Defect Detection in Noisy and Imbalanced Data." SID Symposium Digest of Technical Papers, vol. 53, no. 1, pp. 971-974, 2022.
[2] Cheng, Qisen, and Chang Zhang, and Xiang Shen. "Estimation of Energy and Time Usage in 3D Printing With Multimodal Neural Network." 2022 4th International Conference on Frontiers Technology of Information and Computer (ICFTIC), pp. 900-903, 2022.
[3] Veličković, Petar, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. "Graph attention networks." arXiv preprint arXiv:1710.10903 (2017).
[4] Hamilton, Will, Zhitao Ying, and Jure Leskovec. "Inductive representation learning on large graphs." Advances in neural information processing systems 30 (2017).
[5] Xing, Jinming, Can Gao, and Jie Zhou. "Weighted fuzzy rough sets-based tri-training and its application to medical diagnosis." Applied Soft Computing 124 (2022): 109025.
[6] Gao, Can, Jie Zhou, Jinming Xing, and Xiaodong Yue. "Parameterized maximum-entropy-based three-way approximate attribute reduction." International Journal of Approximate Reasoning 151 (2022): 85-100.
[7] Livingstone, S. R., and F. A. Russo, "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)," PloS one, vol. 13, no. 5, p. e0196391, 2018.
[8] Heigold, G., I. L. Moreno, S. Bengio, and N. Shazeer, "End-to-End Text-Dependent Speaker Verification," in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2016, pp. 5115–5119.
[9] Hochreiter, S. and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.
[10] Gers, F. A., J. Schmidhuber, and F. Cummins, "Learning to Forget: Continual Prediction with LSTM," Neural Computation, vol. 12, no. 10, pp. 2451–2471, 2000.
[11] Ciresan, D., Meier, U., Schmidhuber, J.: "Multi-column deep neural networks for image classification". In: Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. pp. 3642–3649. IEEE (2012)
[12] Cireşan, D.C., Giusti, A., Gambardella, L.M., Schmidhuber, J.: "Mitosis detection in breast cancer histology images with deep neural networks". In: Medical Image Computing and Computer-Assisted Intervention–MICCAI 2013, pp. 411–418. Springer (2013)
[13] Ciresan, D.C., Meier, U., Masci, J., Maria Gambardella, L., Schmidhuber, J.: "Flexible, high performance convolutional neural networks for image classification". In: IJCAI Proceedings-International Joint Conference on Artificial Intelligence. vol. 22, p. 1237 (2011)
[14] Cireşan, D.C., Meier, U., Gambardella, L.M., Schmidhuber, J.: "Convolutional neural network committees for handwritten character classification". In: Document Analysis and Recognition (ICDAR), 2011 International Conference on. pp. 1135–1139. IEEE (2011)
[15] Egmont-Petersen, M., de Ridder, D., Handels, H.: "Image processing with neural networks a review". Pattern recognition 35(10), 2279–2301 (2002)
[16] Farabet, C., Martini, B., Akselrod, P., Talay, S., LeCun, Y., Culurciello, E.: "Hardware accelerated convolutional neural networks for synthetic vision systems". In: Circuits and Systems (ISCAS), Proceedings of 2010 IEEE International Symposium on. pp. 257–260. IEEE (2010)
[17] Hinton, G.: "A practical guide to training restricted boltzmann machines". Momentum 9(1), 926 (2010)
[18] Hinton, G.E., Srivastava, N., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.R.: "Improving neural networks by preventing co-adaptation of feature detectors". arXiv preprint arXiv:1207.0580 (2012)
[19] Ji, S., Xu, W., Yang, M., Yu, K.: "3d convolutional neural networks for human action recognition". Pattern Analysis and Machine Intelligence, IEEE Transactions on 35(1), 221–231 (2013)
[20] Karpathy, A., Toderici, G., Shetty, S., Leung, T., Sukthankar, R., Fei-Fei, L.: "Large-scale video classification with convolutional neural networks". In: Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on. pp. 1725–1732. IEEE (2014)
[21] Krizhevsky, A., Sutskever, I., Hinton, G.E.: "Imagenet classification with deep convolutional neural networks". In: Advances in neural information processing systems. pp. 1097–1105 (2012)
[22] LeCun, Y., Boser, B., Denker, J.S., Henderson, D., Howard, R.E., Hubbard, W., Jackel, L.D.: "Backpropagation applied to handwritten zip code recognition". Neural computation 1(4), 541–551 (1989)
[23] LeCun, Y., Bottou, L., Bengio, Y., Haffner, P.: "Gradient-based learning applied to document recognition". Proceedings of the IEEE 86(11), 2278–2324 (1998)
[24] Nebauer, C.: "Evaluation of convolutional neural networks for visual recognition". Neural Networks, IEEE Transactions on 9(4), 685–696 (1998)
[25] Simard, P.Y., Steinkraus, D., Platt, J.C.: "Best practices for convolutional neural networks applied to visual document analysis". In: null. p. 958. IEEE (2003)
[26] Srivastava, N.: "Improving neural networks with dropout". Ph.D. thesis, University of Toronto (2013)
[27] Szarvas, M., Yoshizawa, A., Yamamoto, M., Ogata, J.: "Pedestrian detection with convolutional neural networks". In: Intelligent Vehicles Symposium, 2005. Proceedings. IEEE. pp. 224–229. IEEE (2005)
[28] Szegedy, C., Toshev, A., Erhan, D.: "Deep neural networks for object detection". In: Advances in Neural Information Processing Systems. pp. 2553–2561 (2013)
[29] Tivive, F.H.C., Bouzerdoum, A.: "A new class of convolutional neural networks (siconnets) and their application of face detection". In: Neural Networks, 2003. Proceedings of the International Joint Conference on. vol. 3, pp. 2157–2162. IEEE (2003)
[30] Zeiler, M.D., Fergus, R.: "Stochastic pooling for regularization of deep convolutional neural networks". arXiv preprint arXiv:1301.3557 (2013)
[31] Zeiler, M.D., Fergus, R.: "Visualizing and understanding convolutional networks". In: Computer Vision–ECCV 2014, pp. 818–833. Springer (2014)
[32] Andén, Joakim, and Stéphane Mallat. "Deep scattering spectrum." Signal Processing, IEEE Transactions on 62.16 (2014): 4114-4128.
[33] Bruna, Joan, and Stéphane Mallat. "Invariant scattering convolution networks." Pattern Analysis and Machine Intelligence, IEEE Transactions on 35.8 (2013): 1872-1886.
[34] Cortes, Corinna, and Vladimir Vapnik. "Support-vector networks." Machine learning 20.3 (1995): 273-297.
[35] Estrach, Joan Bruna. "Scattering representations for recognition."
[36] Gerald, Kaiser. A friendly guide to wavelets, 1994.
[37] Boser, Le Cun, John S Denker, D Henderson, Richard E Howard, W Hubbard, and Lawrence D Jackel. "Handwritten digit recognition with a back-propagation network." In Advances in neural information processing systems. Citeseer, 1990.
[38] LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. "Deep learning." Nature 521.7553 (2015): 436-444.
[39] Mallat, Stéphane. "Group invariant scattering." Communications on Pure and Applied Mathematics 65.10 (2012): 1331-1398.
[40] Mallat, Stéphane. "Understanding deep convolutional networks." arXiv preprint arXiv:1601.04920 (2016).


