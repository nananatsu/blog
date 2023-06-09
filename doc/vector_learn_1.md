## 向量搜索学习笔记

 通过统计方法或机器学习，能够将现实世界的对象、概念转为向量表示， 再通过向量间的距离来判断它们的相似度。
 

Transform近年最流行的一个模型，基于注意力机制，分为编码器、解码器两个部分，每个部分可以独立使用：
 - 仅编码器模型，需要理解输入的任务，模型代表：ALBERT、BERT、DistilBERT、ELECTRA、RoBERTa
 - 仅解码器模型，用于生成任务，模型代表：CTRL、GPT、GPT-2、TransformerXL
 - 编码器-解码器模型，用于需要输入的生成任务，模型代表：BART、mbart、Marian、T5

通过使用一些编码器模型我们能够将文本、图片、音频转换为向量表示：
- 文本转换：BERT、RoBERT、DistilBERT、ALBERT、Debert
- 图片转换：ViT、Swin Transfomer、BeIT、ViTMAE
- 音频转换：Wav2Vec2、HuBERT
向量在计算机中分为两类：
- 密集向量，每个维度都包含信息，难以压缩，如[1.0,0.2,0.4,...,1.2,4.5,3.4]
- 稀疏向量，大部分维度0，少部分维度非0，如： [1.0,0,0,...,0,0,3.4]
  - 可以用三个分量来表示稀疏向量，减少存储所需空间
    -  向量大小
    -  非0元素的索引
    -  非0元素的值
向量距离在数学中称为范数(norm)，是实数/复数向量空间到非负实数的函数，表现得像与原点间的距离。

给定$X$是复数子域$F$上的向量空间，$X$上的范数p需要满足以下条件：
  1. 三角不等式： $p(x+y) ≥ p(x) + p(y)$， $x,y \in X$；
  2. 绝对齐次性： $ p(sx) = \vert s \vert p(x)$，$x \in X$，$s$是一个标量；
  3. 正定性：对所有$x \in X$当且仅当$x=0$时$p(x) =0$

当仅满足条件1、2时称为半范数。

通常用$\Vert x \Vert$来表示范数,在欧几里得空间中时也会使用$\vert x \vert$来表示范数。

常见范数：
- 绝对值范数：$\Vert x \Vert = \vert x \vert $
- 欧几里得范数：$\Vert x \Vert _2 = \sqrt{\mathop{∑}\limits_{i=1}^{n} x_i^2} $
- 曼哈顿范数：$\Vert x \Vert _1 = \mathop{∑}\limits_{i=1}^{n} \vert x_i \vert $
- 最大范数：$\Vert x \Vert _∞ = \mathop{max}\limits_{i} ( \vert x_i \vert ) $
- p范数：对于向量$x=(x_1,x_2,...,x_n)$，$p ≥ 1$，p范数为$ \Vert x \Vert _p = (\mathop{∑}\limits_{i=1}^{n}  \vert  x_i - y_i \vert ^p  ) ^ {1/p}$
  - p为1时是曼哈顿范数
  - p为2时是欧几里得范数
  - p接近$∞$时p范数接近最大范数
  - p在 $0 < x <1 $间是结果不是范数，其结果不满足三角不等式

 常见相似度/距离计算方式：
  ```python
  In []: %pip install scipy
         from scipy.spatial import distance
         from scipy.stats import entropy
         import numpy as np
  ```
 - 欧几里得距离（Euclidean Distance），也被称为欧氏距离、$L_2$/$L^2$范数/距离，连接两点的直线距离，最常用的距离度量，随维度较增加适用性下降。

    在一维上点x与y的距离是两点坐标值差的绝对值：

    $D(x,y) = \vert x - y \vert = \sqrt{(x-y)^2}$
    
    在二维平面上，建立笛卡尔坐标,使x坐标$(x_1,x_2)$，y坐标$(y_1,y_2)$，则x与y的距离为：

    $D(x,y) =  \sqrt{(x_1-y_1)^2 + (x_2 - y_2)^2} $

    一般的对于笛卡尔坐标给定n维欧式空间，x与y间的距离为：
    
    $D(x,y) = \sqrt{\mathop{∑}\limits_{i=1}^{n} (x_i - y_i)^2}$

    简化为向量范数（欧几里得距离便是欧几里得向量空间上的定义的范数）表示：

    $D(x,y) = \Vert x -y \Vert $
    
    ```python
    In []: distance.euclidean([1, 0, 0], [0, 1, 0])
    Out []: 1.4142135623730951
    ```
- 余弦相似性（Cosine Similarity），两个向量夹角的余弦，只考虑了向量的方向，没有考虑向量大小，向量间的差异没有充分考虑。

  依据向量内积公式： $ x⋅y = \Vert x \Vert \Vert y \Vert cos θ $ 可知x,y的余弦相似性为：

   $D(x,y) = cos(θ) = \frac{ x \cdot y} { \Vert x  \Vert  \Vert y  \Vert} $

   向量内积也可以表示为向量各分量乘积之和：$ x⋅y = \mathop{∑}\limits_{i=1}^{n}  x_i × y_i $
   
   向量的范数为自身内积的平方根：$ \Vert x \Vert = \sqrt{ x \cdot x } = \sqrt{\mathop{∑}\limits_{i=1}^{n}  x_i ^2 }$

   可得出向量余弦相似性可表示为：

  $ D(x,y)= \frac{\mathop{∑}\limits_{i=1}^{n}  x_i × y_i} {\sqrt{\mathop{∑}\limits_{i=1}^{n}  x_i ^2}×\sqrt{\mathop{∑}\limits_{i=1}^{n}  y_i ^2}} $
    ```python
    In []: distance.cosine([1, 0, 0], [0, 1, 0]) 
    Out []: 1.0
    ```
- 曼哈顿距离（Manhattan Distance），也被称为出租车距离、、$L_1$/$L^1$范数/距离，两点间的网格距离，基于曼哈顿几何，相较欧几里得几何使用了新的距离函数/度量，两个点的距离是它们笛卡尔坐标差的绝对值之和，在高维数据中不够直观，因为不是直线距离，给出的距离比欧几里得距离大。
  
   $ D(x,y) = \mathop{∑}\limits_{i=1}^{n}  \vert x_i - y_i  \vert $ 

  相应范数形式为：

  $ D(x,y) = \Vert x -y \Vert_1$

    ```python
    In []: distance.cityblock([1, 0, 0], [0, 1, 0]) 
    Out []: 2
    ```

 - 切比雪夫距离（Chebyshev Distance），也被称为最大度量、$L_∞$度量，两个向量各维度的数据差值的最大值，即沿着坐标轴计算的最大距离，适用场景特殊，一般不推荐使用。
  
   $D(x,y) = \mathop{max}\limits_{i} ( \vert x_i - y_i \vert )  = \mathop{lim} \limits_{k \to ∞ } ( \mathop{∑}\limits_{i=1}^{n}  \vert  x_i - y_i \vert ^p  ) ^ {1/p} $

    ```python
    In []: distance.chebyshev([1, 0, 0], [0, 1, 0])
    Out []: 1
    ```

  - 闵可夫斯基距离（Minkowski，闵氏距离），是欧几里得距离或曼哈顿距离的推广，$p$为1时为曼哈顿距离、$p$为2时为欧几里得距离、$p$为$∞$时为切比雪夫距离。
     
       $D(x,y) = (\mathop{∑}\limits_{i=1}^{n}  \vert x_i - y_i  \vert ^ p ) ^ {1/p} $
    
    ```python
    In []: distance.minkowski([1, 0, 0], [0, 1, 0], 1)
    Out []: 2.0

    In []: distance.minkowski([1, 0, 0], [0, 1, 0], 2) 
    Out []: 1.4142135623730951

    In []: distance.minkowski([1, 0, 0], [0, 1, 0], 3)
    Out []: 1.2599210498948732
    ```

 - 雅卡尔指数（Jaccard Index），样本交集大小除以并集大小，从1减去雅卡尔指数得到雅卡尔距离，受数据集大小影响很大。
  
    $D(x,y) = 1 - \frac{ \vert  x \bigcap y \vert }{ \vert  x \bigcup y \vert }  = 1 -  \frac{  x \cdot y }{ \Vert x \Vert ^2 + \Vert y \Vert ^2  -   x \cdot y }  = 1- \frac{ \mathop{∑}\limits_{i=1}^{n} x_i \times y_i }{   \mathop{∑}\limits_{i=1}^{n} x_i^2 + \mathop{∑}\limits_{i=1}^{n} y_i^2 -  \mathop{∑}\limits_{i=1}^{n} x_i \times y_i   }$ 

    ```python
    In []: distance.jaccard([1, 0, 0], [0, 1, 0])
    Out []: 1.0
    ```

  - 戴斯系数（Sørensen-Dice Index），与雅卡尔指数十分相似，可以视为两个集合的重叠率，相应差异函数1-戴斯系数没有三角不等式，不适合作为距离度量，戴斯系数在异构数据集中保持敏感，能够降低异常值的权重。
  
   $D(x,y) = \frac{ 2 \vert  x \bigcap y \vert }{ \vert  x   \vert +  \vert  y \vert }   =  \frac{ 2  \vert x \cdot y \vert }{ \Vert x \Vert ^2 + \Vert y \Vert ^2  }  =\frac{ 2 \mathop{∑}\limits_{i=1}^{n} x_i \times y_i }{   \mathop{∑}\limits_{i=1}^{n} x_i^2 + \mathop{∑}\limits_{i=1}^{n} y_i^2 } $

  ```python
    In []: distance.dice([1, 0, 0], [0, 1, 0])
    Out []: 1.0
  ```

  - 汉明距离（Hamming Distance），两个向量间不同值的数量，当向量长度不等时很难使用并且不会考虑实际值，常用与网络传输中的纠错/检测。

  ```python
    In []: distance.hamming([1, 0, 0], [0, 1, 0])
    Out []: 0.6666666666666666
  ```
 
  - 半正矢距离（Haversine Distance），给定经纬度球面上两点间的距离，实际上很少有这种情况，更多的是计算椭圆面上的距离（vincenty距离）。

  $D(xy) = 2r \: arcsin(  \sqrt{ sin^2(\frac{ \varphi _2 - \varphi _1}{2}) + cos \varphi _1⋅cos \varphi _2⋅sin^2(\frac{λ_2 - λ_1}{2} ) } ) $

  ```python
    In []: from math import radians, cos, sin, asin, sqrt
          def haversine(lon1, lat1, lon2, lat2):
              lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
              dlon = lon2 - lon1 
              dlat = lat2 - lat1 
              a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
              c = 2 * asin(sqrt(a)) 
              r = 6371 
              return c * r

          haversine(1,0,0,1)
    Out []: 157.24938127194397
  ```

  - 马哈拉诺比斯距离（Mahalanobis Distance，马氏距离），用于计算点与分布间的距离。给定分布D，均值$μ = (μ_1,μ_2,...,μ_p)$，协方差矩阵S，则点$x = (x_1.x_2,...,x_n)$与分布D的距离为：
     
      $D(x,D) = ( (x-μ ) ^T S ^ {-1} (x-μ) )^{1/2}$
    
    两个服从分布D的随机点x、y间的马氏距离为：
      
      $D(x,y;D) = ( (x-y ) ^T S ^ {-1} (x-y) )^{1/2}$
    
     当协方差矩阵是单位矩阵时，可以将其等效为欧氏距离（$ σ_i$为$x_i$的标准差）：
        
       $D(x,y) = ( \frac{ \mathop{∑}\limits_{i=1}^{n} (x_i - y_i)^2}{ σ_i^2 })^{1/2}$


     ```python
    In []: iv = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
           distance.mahalanobis([1, 0, 0], [0, 1, 0], iv) 
    Out []: 1.0
    ```

- 堪培拉距离（Canberra Distance），曼哈顿距离的加权版本。
    
  $D(x,y) = \mathop{∑}\limits_{i=1}^{n}  \frac{ \vert x_i - y_i  \vert}{ \vert x_i  \vert +  \vert y_i  \vert} $

    ```python
    In []: distance.canberra([1, 0, 0], [0, 1, 0]) 
    Out []: 2.0
    ```

 - 布雷-柯蒂斯相异度（Bray-curtis dissimilarity），主要用于生物学/生态学，衡量地域间物种组成差异。
   
   $D(x,y) = \frac{ \mathop{∑}\limits_{i=1}^{n}  \vert x_i - y_i  \vert }{ \mathop{∑}\limits_{i=1}^{n}  \vert x_i + y_i  \vert}$


     ```python
    In []: distance.braycurtis([1, 0, 0], [0, 1, 0]) 
    Out []: 1.0
    ```

 - KL散度（Kullback–Leibler divergence），衡量两个分布的相似性，离散概率分布$P$、$Q$在同一样本空间$\chi$，$Q$到$P$的KL散度为：
  
   $D_{KL}( P || Q ) =  \mathop{∑}\limits_{x \in \chi} P(x) log (\frac{P(x)} {Q(x)})$

   当$P$、$Q$是绝对连续的，$Q$到$P$的KL散度可定义为：
    
     $D_{KL}( P || Q ) = \int_{x} log( \frac{P(dx)} {Q(dx) }) P(dx) = \int_{x}  \frac{P(dx)}{Q(dx)} log( \frac{P(dx)} {Q(dx) }) Q(dx) $
      
    $μ$是$\chi$上的任意测度，对于概率密度$p$存在 $P(dx) = p(x)μ(dx)$ 、$Q(dx) = q(x)μ(dx)$，$Q$到$P$的KL散度可定义为：

      $D_{KL}( P || Q ) = \int_{x} p(x) log( \frac{p(x)} {q(x) }) μ(dx) $

    ```python
    In []: def KL(p,q):
              epsilon = 0.00001
              p = np.asarray(p)+epsilon
              q = np.asarray(q)+epsilon

              return np.sum(p * np.log(p/q))

          KL([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    Out []: 11.512935464920231

    In []: entropy([1.0, 0.0, 0.0], [0.0, 1.0, 0.0],2)
    Out []: inf
    ```

- JS散度（Jensen–Shannon divergence），是KL散度的平滑版本。

  $D_{JS}(P || Q) = \frac{1}{2}D_{KL}(P || M)  + \frac{1}{2}D_{KL}(Q || M)$，其中$ M =\frac{1}{2}(P+Q) $

    ```python
    In []: distance.jensenshannon([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0)
    Out []: 1.0
    ```
 
相似性搜索（向量搜索），给定一组向量和查询向量，从向量集中找到最相似的项目。常用的有两种搜索方式：
  - KNN（k-nearest neighbors algorithm，k-最近邻居算法），在向量空间中为给定查询向量找到最近的向量，k-NN在查询时需要查询向量与向量集中每个向量的距离。
  - 为减少KNN这类算法的计算复杂度，通过建立索引结构来缩小搜索空间以缩短查询时间。
    - 精确搜索，空间索引/度量树
        - 欧几里得空间，空间索引，如kd树、R树、R*树
        - 一般度量空间，度量树，如M树、VP树、BK树
    - 近似搜索，ANN（approximately nearest neighbors，近似最近邻居）
      - 邻近邻域图搜索，如HNSW
      - 局部敏感散列
      - 基于压缩/聚类搜索

Faiss（Facebook AI Similarity Search）是一个流行的相似性搜索库，我们可以将向量存储到Faiss中进行索引，再用查询向量从Faiss中找到最相似的向量。

  ```python
    In []: %pip install pandas sentence-transformers torch faiss-gpu
           import pandas as pd
           from sentence_transformers import SentenceTransformer
           import faiss
           from google.colab import drive
  ```

复制<https://github.com/brightmart/nlp_chinese_corpus>中的中文语料到google drive，再挂载google drive到/content/drive/，解压语料到本地。
  ```python
  In []: drive.mount('/content/drive/')
         !mkdir -p mydata
         !unzip /content/drive/MyDrive/train_data/baike2018qa.zip -d mydata
  ```
读取语料，使用Sentence-BERT模型将语句转为密集向量。
- BERT模型编码后通常会有512个密集向量，每个密集向量包含768个值，Sentence-BERT对BERT进行修改允许创建能够表示完整序列的单个向量。

 ```python
  In []: data = pd.read_json('./mydata/baike_qa_train.json', lines=True, nrows=10000)

         sentences = data['title']

         # model = SentenceTransformer('uer/sbert-base-chinese-nli')
         model = SentenceTransformer('DMetaSoul/sbert-chinese-general-v2')
         sentence_embeddings = model.encode(sentences)

         sentence_embeddings.shape
  Out []: (10000, 768)
  ```

使用模型输出向量纬度初始化faiss IndexFlatL2索引，再将编码后向量加入索引
- IndexFlatL2能够测量查询向量与加载到索引向量间的L2（欧几里得）距离
- IndexFlatIP使用内积拉判断向量距离(归一化的余弦相似性)
- IndexFlat支持的其他类型度量:$L_1$距离、$L_∞$距离、$L_p$距离、堪培拉距离、马氏距离、雅卡尔指数、布雷-柯蒂斯相异度、JS散度

 ```python
  In []: d = sentence_embeddings.shape[1]

         index = faiss.IndexFlatL2(d)
         index.add(sentence_embeddings)
  ```

将一个查询语句用相同模型编码为向量，在索引中查询4个最相似向量

 ```python
  In []: k = 4
         xq = model.encode(["眼睛干涩发痒怎么办？"])
  ```

  ```python
  In []: %%time
         D, I = index.search(xq, k)
         print(I)
  Out []: [[8142 1502 6328 5274]]
          CPU times: user 8.78 ms, sys: 0 ns, total: 8.78 ms
          Wall time: 8.87 ms
  ```

  ```python
  In []: data.iloc[I[0]]
  Out []: 	qid	category	title	desc	answer
         8142	qid_2203978748591224574	健康-五官科-眼科	眼干眼痒怎么办	眼干眼痒怎么办	患有干眼症可吃些维生素A ,菊花里含有丰富的维生素A,是维护眼睛健康的重要物质,因此,也是中...
         1502	qid_5936099305506517681	健康-五官科-眼科	眼睛老是痛怎么办啊，老感觉眼睛里有东西。	老是痛怎么办啊，老感觉眼睛里有东西。	建议到医院眼科检查是否结膜炎、角膜炎等疾病。建议你到医院检查一下在服用药物，祝好。感觉眼睛有...
         6328	qid_8638202508904030888	健康-五官科-眼科	先谢谢了早晨起来眼肿怎么办？	早晨起来眼肿怎么办？	眼睛肿与饮水和睡觉的姿势都有关。\r\n\r\n睡前喝太多的水，身体水分过多，由于眼睑组织较...
         5274	qid_6527271401979662043	健康-外科-脑外科	有什么方法能快速缓解眼睛疲劳，脑神经疲劳呀！有点累了。。。	有什么方法能快速缓解疲劳，脑神经疲劳呀！有点累了。。。	眼睛干涩，平时就要注意用眼习惯，定时休息，看看远处的景物。补充维生素A，多吃胡萝卜、水果、海...
  ```

每次查询时会与将查询向量与索引上所有向量进行比较，大量数据时索引会变得越来越慢。有两种常用方法进行优化：
- 减少向量大小，进行降维或减少生成向量的维数；
- 缩小搜索范围，通过某些属性、向量距离将向量聚类或组织为树，并将搜索限制在最相似的簇或分支；

faiss通过缩小搜索范围来优化查询速度：
- IndexIVFFlat，使用k-means聚类，将数据分为nlist个簇，查询时先将查询向量与簇的特征向量进行比较，找到最相似的nprope个簇，再在簇中进行查询最相似的k个向量。IndexIVFFlat使用了聚类，在添加索引数据前需要对数据进行训练。

  ```python
  In []: nlist = 50  # 聚类数量

         quantizer = faiss.IndexFlatL2(d) # 聚类距离量化器

         indexIVF = faiss.IndexIVFFlat(quantizer, d, nlist)

         indexIVF.nprobe = 10 # 搜索簇数量

         indexIVF.train(sentence_embeddings)

         indexIVF.add(sentence_embeddings)
  In []: %%time
         D, I = indexIVF.search(xq, k)
         print(I)
  Out []: [[8142 1502 6328 5274]]
          CPU times: user 2.9 ms, sys: 999 µs, total: 3.9 ms
          Wall time: 3.74 ms
  ```

- IndexHNSWFlat，NSW是一种图，顶点连接到最近的邻居，HNSW将NSW分为多层，消除顶点间的中间连接。

  ```python
  In []: M = 64  # 顶点邻居数量
         ef_search = 32  # 搜索索引时查询层数量
         ef_construction = 64  # 加入索引时查询层数量

         indexHNSW = faiss.IndexHNSWFlat(d, M)
         indexHNSW.hnsw.efConstruction = ef_construction
         indexHNSW.hnsw.efSearch = ef_search

         indexHNSW.add(sentence_embeddings)
  In []: %%time
         D, I = IndexHNSWFlat.search(xq, k)
         print(I)
  Out []: [[8142 1502 6328 5274]]
          CPU times: user 2.18 ms, sys: 1e+03 ns, total: 2.18 ms
          Wall time: 2.18 ms
  ```

- IndexLSH，局部敏感Hash，通过Hash函数将向量分组，Hash函数需要最大化Hash冲突。

  ```python
  In []: nbits = 50  # 分组数量
         indexLSH = faiss.IndexLSH(d, nbits)
         indexLSH.add(sentence_embeddings)
  In []: %%time
         D, I = IndexLSH.search(xq, k)
         print(I)
  Out []: [[8142 4638 9553 3816]]
          CPU times: user 1.19 ms, sys: 0 ns, total: 1.19 ms
          Wall time: 1.07 ms
  ```

如果完整的存储向量，当数据集很大时，会带来很大的存储压力，通常会对数据进行压缩。

在faiss中可以使用Product Quantization (PQ，乘积量化)来压缩向量，PQ可以视为一个额外的近似步骤，IVF通过缩小搜索范围进行近似，PQ通过距离/相似性计算来进行近似。
- 将原始向量拆分为几个子向量；
- 对每个子向量执行聚类，为每个子向量集创建质心；
- 用距离子向量最近的质心id替换子向量；

  ```python
  In []: m = 8  #最终压缩向量中质心数量
         bits = 8 # 质心的位数

         quantizerPQ = faiss.IndexFlatL2(d)  # 继续使用L2距离进行量化
         indexIVFPQ = faiss.IndexIVFPQ(quantizerPQ, d, nlist, m, bits) 
         indexIVFPQ.nprobe = 10

         indexIVFPQ.train(sentence_embeddings)
         indexIVFPQ.add(sentence_embeddings)
  In []: %%time 
         D, I = indexIVFPQ.search(xq, k)
         print(I)
  Out []: [[6328 8142 1502 4918]]
          CPU times: user 2.61 ms, sys: 0 ns, total: 2.61 ms
          Wall time: 1.98 ms
  ```

<a href="https://colab.research.google.com/github/nananatsu/blog/blob/master/vector_learn.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

参考：
  - [Faiss: The Missing Manual](https://www.pinecone.io/learn/faiss-tutorial/)
  - [9 Distance Measures in Data Science](https://www.pinecone.io/learn/faiss-tutorial/)
  - wiki