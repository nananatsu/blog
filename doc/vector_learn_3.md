
使用[baike2018qa](https://github.com/brightmart/nlp_chinese_corpus)作为测试数据集。
```python
%pip install numpy pandas scipy sentence-transformers torch faiss-gpu
import pandas as pd

data = pd.read_json('./mydata/baike_qa_train.json', lines=True, nrows=5000)

sentences = data['title']
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('DMetaSoul/sbert-chinese-general-v2')
sentence_embeddings = model.encode(sentences)
```

量化是将输入值从大集合（通常是连续的）映射到较小集合（通常具有有限数量元素），舍入（用更短、更简单、更明确的近似值进行替换）和截断（限制小数点右边的位数）是典型的量化过程，量化构成了所有有损压缩算法的核心。输入值与量化值间的差异称为量化误差，执行量化的设备或算法功能称为量化器。
PQ（乘积量化）通过将向量空间分解为低位子空间的笛卡尔积，每个子空间独自进行量化，这样一个向量能够被子空间量化后的索引组成的短编码表示，两个向量间的L2距离可以通过量化后的编码进行估计。

定义$q$是一个量化函数，将D维向量$x \in \mathbb{R}^D$映射为向量$q(x) \in C $，$C = \{c_i;i \in I \}$是一个大小为k的编码簿， 其中$I = 0 ... k-1$是一个有限的索引集合，$c_i$被称为质心。

将所有映射到同一索引$i$的向量划分为一个单元(Voronoi cell) $V_i$：

$V_i \mathop{=}\limits^{Δ} \{x ∈ \mathbb{R}^D : q(x) = c_i \}$，

量化器的$k$个单元是向量空间$\mathbb{R}^D$的一个分区，在同一个单元$V_i$中向量都被质心$c_i$重构（用质心$C_i$表示单元$V_i$特征），量化器的好坏可以用输入向量与其再现值$q(x)$间的均方误差(MSE)来度量，使用$d(x,y) = \Vert x -y \Vert$表示两个向量间的L2距离，$p(x)$表示随机变量$X$的概率分布函数，则其均方误差为：

$MSE(q) = \mathbb{E}_X[d(q(x),x)^2] = ∫ p(x) d(q(x),x)^2 dx$
```python
from random import randint

x = sentence_embeddings[0]

D = len(x)
k = 2**8
from scipy.spatial import distance

# 找到最近的质心
def nearest(x, c, k):
  min_distance = 9e9
  idx = -1

  # 找到L2距离的质心
  for i in range(k):
    l2_distance = distance.euclidean(x,c[i])
    if l2_distance < min_distance:
      idx = i
      min_distance = l2_distance
  return idx
import numpy as np

# 随机建立质心创建编码簿
c = []
for i in range(k):
  c_i =  [randint(0, 9) for _ in range(D)]
  c.append(c_i)
```
```python
In []: # 测试向量x与其量化值c_i的均方误差
      i = nearest(x, c, k)
      mse = (np.square(x - c[i])).mean()
      mse

Out []: 26.871004007228425

```
为使量化器最优，需要满足劳埃德(Lloyd)最优条件：
- 向量$x$需要被量化到最近的质心，以L2距离作为距离函数则：$q(x) = arg \: \mathop{min}\limits_{c_i \in C} d(x,c_i)$，这样单元间被超平面划分。
- 质心必须是Voronoi单元中的向量的期望：$c_i = \mathbb{E}_X[x|i] = \int_{V_i} p(x) x dx$，劳埃德量化器迭代分配向量到质心并从分配后的向量集合中重新估计质心。

```python
# 迭代估计质心
def lloyd_estimate(cells,c,embeddings,k,ite_num):
  # 按新质心重新分配向量
  for i,v in enumerate(embeddings):
    idx = nearest(v,c,k)
    cells[idx].append(i)

  end = True
  # 遍历各单元，计算单元中向量的期望作为质心
  for i,cell in enumerate(cells):
    if len(cell) > 0:
      cell_vectors = []
      for idx in cell:
        cell_vectors.append(embeddings[idx])
      centroid = np.asarray(cell_vectors).mean(axis=0)

      if np.all(c[i] != centroid):
        c[i] = centroid
        end = end & False
      cells[i] = []

  ite_num-=1
  # 当所有单元质心不在变化或进行10次迭代后返回
  if end or ite_num <= 0 :
    return
  lloyd_estimate(cells,c,embeddings,k,ite_num)
# 重新估计质心
c = np.random.randint(1, int(sentence_embeddings.max()+1), (k, D))
cells = []
for i in range(k):
  cells.append([])

lloyd_estimate(cells,c,sentence_embeddings[:4000],k,10)
```
```python
# 随机检查向量与其量化值得均方误差
In []:  mses = []
        for i in range(10):
          idx = randint(4000, len(sentence_embeddings))
          x = sentence_embeddings[idx]
          c_idx = nearest(x,c,k)
          mse = (np.square(x - c[c_idx])).mean()
          mses.append(mse)
        mses

Out []:  [0.4213924281641214,
          0.44701967442937934,
          0.40987476818263135,
          0.3347929217051524,
          0.3945715719573131,
          0.3933250410274553,
          0.4046597370237311,
          0.4163056436451407,
          0.386595268688162,
          0.4268464239636341]
```
当有大量向量时，我们需要增加质心数量以减小均方误差，假设要将一个128维的向量将其量化为64位的编码，则需要$k=2^{64}$个质心与编码对应，每个质心为128维浮点数，需要$D × k = 2^{64} \times 128$个浮点值来存储质心，量化器的训练复杂度是$k=2^{64}$的好几倍，这样在内存进行向量量化是不可能的，乘积量化通过允许选择进行联合量化的组件数量来解决存储及复杂性问题。
将输入向量$x$切分为$m$个不同的子向量$u_j, 1 ≤ j ≤ m$，子向量维度$D^* = D/m$，D是m的倍数，将子向量用m个不同量化器分别进行量化，$q_j$是第j个子向量使用的量化器，通过子量化器$q_j$关联索引集合$I_j$，对应到编码簿$C_j$及相应的质心$c_{j,i}$

$ \underbrace{x_1,...,x_{D^*},}_{u_1(x)} ...,\underbrace{x_{D-D^*+1},...,x_D}_{u_m(x)} → q_1(u_1(x)),...,q_m(u_m(x))$
乘积量器再现值由索引集合的笛卡尔积$I = I_1 × ... × I_m $确定，相应的编码簿为$C = C_1 × ... × C_m$，集合中的元素对应向量经m个子量化器处理后的质心，假设所有的子量化器有着有限个数$k^*$个再现值，总质心数$k = (k^*)^m$
```python
# 子向量数量
m = 8
assert D % m == 0
assert k % m == 0

# 子向量纬度
D_ = int(D/m)
# 子编码簿大小
k_ = 256
k = k_*m
# 分割子向量
embeddings_split = sentence_embeddings[:4000].reshape(-1, m, D_)
# 生成随机编码簿
c_s = np.random.randint(1, int(sentence_embeddings.max()+1), (m, k_, D_))

cells = []
# 训练量化器
for i in range(m):
  cells_i = []
  for j in range(k_):
    cells_i.append([])
  lloyd_estimate(cells_i,c_s[i],embeddings_split[:,i],k_,10)
  cells.append(cells_i)
def quantization(v):
  u = v.reshape(m,D_)
  ids = []
  for j in range(m):
    idx = nearest(u[j], c_s[j], k_)
    ids.append(idx)
  return ids
```
```python
In []:  # 随机检查向量与其量化值得均方误差
        mses = []
        for i in range(10):
          v = sentence_embeddings[randint(4000, len(sentence_embeddings))]
          ids = quantization(v)
          q = []
          for j,u in enumerate(ids):
            q.extend(c_s[j][u])
          mse = (np.square(v - q)).mean()
          mses.append(mse)
        mses

Out []:  [0.38222087722969444,
          0.35351928153797624,
          0.4083119303245983,
          0.36553835592335915,
          0.4584325647498803,
          0.41869153930207464,
          0.4386770316646695,
          0.39155881868569337,
          0.39578527771000305,
          0.3789821753973439]
```

乘积量化的优势在于通过几个小的质心集合生成一个大的质心集合，只需要存储子量化器对应的$m \times k^*$个质心，总计$mk^*D^*$个浮点值，即相应内存使用及复杂性为$mk^*D^* = k^{(1/m)}D $，相较其他量化方法k-means、HKM，PQ能够在内存中对较大k值得向量进行索引。
在向量中连续的元素在结构上通常是相关的，最好使用同一个子量化器进行量化，由于子向量空间是正交的，量化器的均方误差可以表示为：

$MSE(q) = \mathop{\sum}\limits_{j} MSE(q_j) $

更高的$k^*$会造成更高的计算复杂度，更大的内存占用，通常$k^* = 256,m = 8$是一个合理的选择。
有两种方法来计算查询向量与量化后向量的距离
- 对称距离计算(SDC)：将查询向量也进行量化，计算量化后向量间的距离
- 非对称距离计算(ADC)：不对查询向量量化，直接计算距离

```python
In []:  quantization_res = []
        for i,v in enumerate(sentence_embeddings):
          ids = quantization(v)
          quantization_res.append(ids)
        qi = randint(4000, len(sentence_embeddings))
        qv = sentence_embeddings[qi]
        q_ids = quantization(qv)

        all_dist = []
        for i,ids in enumerate(quantization_res):
          dist = 0
          for j,id in enumerate(ids):
            dist+=distance.euclidean(c_s[j][id], c_s[j][q_ids[j]])
          all_dist.append((dist,i))
        all_dist.sort()

        print('查询:', sentences[qi])
        for d,idx in all_dist[:5]:
          print('   L2距离:',d,'匹配:',sentences[idx] )
Out []:  查询: 怎样才能安全删除硬盘中的无用文件？ 
            L2距离: 0.0 匹配: 我下边的全是白的还有地图是么也看不见? 
            L2距离: 0.0 匹配: 怎么看出款项已经到账？我怎样才能知道一笔款项是否已经到了公司的帐 
            L2距离: 0.0 匹配: 月球上总共有多少座环形山 
            L2距离: 0.0 匹配: 怎样才能安全删除硬盘中的无用文件？ 
            L2距离: 0.0 匹配: 红斑狼疮的发病率高吗？请问红斑狼疮是不是一种少见的病？ 
```

```python
In []:  qvs = qv.reshape(m, D_)
        all_dist_adc = []
        for i,ids in enumerate(quantization_res):
          dist = 0
          for j,id in enumerate(ids):
            dist+=distance.euclidean(c_s[j][id], qvs[j])
          all_dist_adc.append((dist,i))
        all_dist_adc.sort()

        print('查询:', sentences[qi])
        for d,idx in all_dist_adc[:5]:
          print('   L2距离:',d,'匹配:',sentences[idx] )
Out []:   查询: 怎样才能安全删除硬盘中的无用文件？ 
              L2距离: 51.58162599842743 匹配: 我下边的全是白的还有地图是么也看不见? 
              L2距离: 51.58162599842743 匹配: 怎么看出款项已经到账？我怎样才能知道一笔款项是否已经到了公司的帐 
              L2距离: 51.58162599842743 匹配: 月球上总共有多少座环形山 
              L2距离: 51.58162599842743 匹配: 怎样才能安全删除硬盘中的无用文件？ 
              L2距离: 51.58162599842743 匹配: 红斑狼疮的发病率高吗？请问红斑狼疮是不是一种少见的病？ 
```

<a href="https://colab.research.google.com/github/nananatsu/blog/blob/master/PQ.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

参考：
  - [Faiss: The Missing Manual](https://www.pinecone.io/learn/faiss-tutorial/)
  - [Product quantization for nearest neighbor search](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf)