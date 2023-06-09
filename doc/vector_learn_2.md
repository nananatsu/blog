
LSH(Locality sensitive hashing)，局部敏感哈希，通过最大化哈希冲突将相似的向量散列到相同的桶中，也可以将该计术视为一种降维方法，将高维度向量转为低维度向量，保留向量间的相对距离。

LSH有多种版本，使用不同的哈希函数和距离度量，最常使用的是最小哈希、随机投影。

使用[baike2018qa](https://github.com/brightmart/nlp_chinese_corpus)作为测试数据集。
```python
%pip install numpy pandas scipy sentence-transformers torch faiss-gpu
import pandas as pd

data = pd.read_json('./mydata/baike_qa_train.json', lines=True, nrows=5000)

sentences = data['title']
```

### 最小哈希

最小哈希是一种快速估计集合相似度的方法，$h$是将集合$U$的成员映射为不同整数的哈希函数，$perm$是集合$U$元素的随机排列，对于$U$的子集$S$，定义$h_{min}(S)$是集合$S$中使得$h(perm(x))$具有最小值的元素。

对于集合$A、B$，假设没有哈希冲突，当且仅当在所有元素$A ∪ B$集合中中具有最小Hash值的元素位于$A \cap B$集合中时存在$h_{min}(A) = h_{min}(B)$，对应概率便是雅卡尔指数

$P_r[(h_{min}(A) = h_{min}(B))] = \frac{ \vert  x \bigcap y \vert }{ \vert  x \bigcup y \vert }$

如果$r$是一个随机变量，当$h_{min}(A) = h_{min}(B)$时为1，其他时候为0，则r是集合A、B的雅卡尔指数的一个无偏估计，r总是0或1使得对应方差很高(值偏离均值的程度)，无法单独用来估计雅卡尔指数，MinHash通过相同方式构造多个变量进行平均以减少相应方差。

最小哈希的最简方案是使用k个不同的哈希函数，设y是满足$h_{min}(A) = h_{min}(B)$的哈希函数数量（可以视为无偏估计r的和），对应的雅卡尔指数估计为$y/k$，这k个函数的对应的$h_{min}(A)$集合保留了原集合与其他集合的相似度，可以用它来表示原集合。

最小哈希版本的LSH分为三步：k-shingling、MinHash、LSH Band。

k-shingling，分割字符串，提取特征向量。
  - 分割字符串为短字符串集合，沿着字符串移动一个长度k的窗口，将窗口中字符串写入集合；
```python
def split_shingles(sentence: str, k: int):
  # 将字符串分割为k个短字符串集合
  shingles = []
  length = len(sentence)
  if length > k:
    for i in range(length - k):
        shingles.append(sentence[i:i+k])
  else:
    shingles.append(sentence)
  return set(shingles)

def split_shingles_batch(sentences: list[str], k: int):
  shingles_list = []
  for sentence in sentences:
    shingles_list.append(split_shingles(sentence,k))
  return shingles_list
```
```python
In []:  k = 4
        shingles_list =split_shingles_batch(sentences, k)

        shingles_list[0]

Out []:  {'上为什么',
          '下的感觉',
          '为什么没',
          '么没有头',
          '人站在地',
          '什么没有',
          '在地球上',
          '地球上为',
          '头朝下的',
          '有头朝下',
          '朝下的感',
          '没有头朝',
          '球上为什',
          '站在地球'}
```
  - 合并所有字符串集合，生成一个包含所有词汇的词汇表，再依据词汇表使用one-hot编码将各个字符串集合转换为稀疏向量；
    - 为每个字符串集合，创建一个词汇表长度的特征向量，词汇表某个位置的词有出现在字符串集合，将该位置置为1，否则为0。
```python
import numpy as np

def build_vocab(shingle_sets: list):
    # 合并所有shingle集合到词汇表
    full_set = {item for set_ in shingle_sets for item in set_}
    vocab = {}
    for i, shingle in enumerate(list(full_set)):
        vocab[shingle] = i
    return vocab

def one_hot(shingles: set, vocab: dict):
    # 对短字符串集合进行one-hot编码
    vec = np.zeros(len(vocab))
    for shingle in shingles:
        idx = vocab[shingle]
        vec[idx] = 1
    
    return vec

def one_hot_batch(shingles_list: list[set[str]], vocab: dict):
  feature_matrix = []
  for shingles in shingles_list:
    feature_matrix.append(one_hot(shingles,vocab))

  return np.stack(feature_matrix)
```
```python
In []:  vocab = build_vocab(shingles_list)

        feature_matrix = one_hot_batch(shingles_list, vocab)

        feature_matrix.shape

Out []: (5000, 105391)

In []:   feature_matrix[:5]

Out []:  array([[0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.]])
```
  - MinHash 将稀疏向量转换为密集向量（签名）
    - 哈希函数是一个随机向量，是词汇表索引的随机排列
    - 从1开始取得哈希函数中值对应索引，通过索引访问稀疏向量，找到第1个为1的索引，该索引在哈希函数对应值便是需要的最小哈希
    - 遍历生成的哈希函数，将得到的最小哈希组成密集向量（签名）
```python
def create_minhash_func(vocab_size: int, nbits: int):
    hashes = np.zeros((nbits, vocab_size))
    for i in range(nbits):
      # 在[0,vocab_size)区间生成值随机排列
      permutation = np.random.permutation(vocab_size) + 1
      # 更新到minhash的第i行
      hashes[i,:] = permutation.copy()
    return hashes.astype(int)

def get_signature(vector: list, minhash_func):
    # 找到向量中第所有非0元素位置
    idx = np.nonzero(vector)[0].tolist()
    # 通过非0元素位置取得哈希函数值
    shingles = minhash_func[:, idx]
    # 从哈希值中找到每行的最小值，作为签名
    signature = np.min(shingles, axis=1)

    return signature

def get_signature_batch(feature_matrix:list,minhash_func:list):
  signatures = []
  for feature in feature_matrix:
    signature = get_signature(feature, minhash_func)
    signatures.append(signature)

  return np.stack(signatures)
```
```python
In []:  minhash_func = create_minhash_func(len(vocab), 100)

        signatures = get_signature_batch(feature_matrix,minhash_func)

        signatures.shape

Out []: (5000, 100)

In []:  signatures[0]

Out []:  array([1167,  892,  430, 3694,  309,  412, 1087,  344, 3089, 1946,  436,
                2559,  594, 1235, 1042,  112, 1157,  556, 1075,  367,  841, 1082,
                4229, 2456, 1750,   74, 1575, 2383, 2018,  786, 3276,  610, 3228,
                 408,  208,  557,  851,  614, 1594, 1327, 3177,  238,  854,  784,
                 105, 2940,  265, 4173, 4190,  412, 2397, 1291,  903,  123, 2295,
                 908,  323, 3946, 1654, 7152, 1843,  600, 5850,  171,  857,  294,
                1592,  348, 1235,  219,  758,  268,  601,  386,  815, 2019, 1866,
                 131, 2700, 1359,  712, 2438,  472, 4114,  996,  464, 1933, 1981,
                3083,  654, 3026, 2501,  481, 3804, 1713,  426,  329, 2969,  881,
                4350])
```
  - LSH Band 获取密集向量进行哈希，寻找哈希冲突，从而将向量放入桶
    - 如将向量整体进行哈希，很难通过哈希函数来识别向量间的相似性，我们通过将向量分为几个子段（称为Band），哈希向量的每个段独立进行哈希；
    - 查询时，两个向量的任意子向量存在冲突，都被视为候选；
```python
def split_vec(signature: list,band :int):
  # 分割向量为多个子向量
  l = len(signature)
  assert l % band == 0
  r = int(l/band)
  subvecs = []
  for i in range(0,l,r):
    subvecs.append(signature[i:i+r])
  return np.stack(subvecs)

def lsh_band(signatures:list,band:int):
  # 将输入向量哈希到不同桶中
  buckets = []
  for i in range(band):
    buckets.append({})

  for i,signature in enumerate(signatures):
    subvecs = split_vec(signature, band).astype(str)
    for j,subvec in enumerate(subvecs):
      subvec = ','.join(subvec)
      if subvec not in buckets[j].keys():
        buckets[j][subvec] = []
      buckets[j][subvec].append(i)

  return buckets
```
```python
def get_canidate(signature, band, buckets):
  # 从桶中取得存在哈希冲突的向量位置
  candidate = []
  subvecs = split_vec(signature,band).astype(str)
  for i,subvec in enumerate(subvecs):
    subvec = ','.join(subvec)
    if subvec in buckets[i].keys():
      candidate.extend(buckets[i][subvec])

  return set(candidate)

def one_hot_for_query(shingles: set, vocab: dict):
    vec = np.zeros(len(vocab))
    for shingle in shingles:
        idx = vocab.get(shingle)
        if idx != None:
          vec[idx] = 1
    return vec
```
```python
band = 50
buckets = lsh_band(signatures,band)
```
```python
from scipy.spatial import distance

In []:  for i in np.random.permutation(len(sentences))[:5]:
        query_sentence = sentences[i]

        query_vector = one_hot_for_query(split_shingles(query_sentence,k),vocab)
        query_signature = get_signature(query_vector, minhash_func)
        result_candidate =  get_canidate(query_signature, band, buckets)
        print('查询:', query_sentence)
        for idx in result_candidate:
            print('   L2距离:',distance.euclidean(query_vector,feature_matrix[idx]),'匹配:',sentences[idx])

Out []: 查询: 华菱000932成本3.66元，节后该如何操作？继续等待或择机卖 
            L2距离: 0.0 匹配: 华菱000932成本3.66元，节后该如何操作？继续等待或择机卖 
        查询: 输电线路电压偏差过大原因分析我公司有一段66KV输电线路LGJ
            L2距离: 0.0 匹配: 输电线路电压偏差过大原因分析我公司有一段66KV输电线路LGJ
        查询: 诛仙印象第5期的背景音乐哪里有下载的?? 
            L2距离: 0.0 匹配: 诛仙印象第5期的背景音乐哪里有下载的?? 
        查询: WPS转换为word文档时,数学公式怎么转?以前的一个WPS文档 
            L2距离: 0.0 匹配: WPS转换为word文档时,数学公式怎么转?以前的一个WPS文档 
        查询: QQ车音响如何改装我喜欢听流行音乐.比较轻柔的.请指教:最好是指 
            L2距离: 0.0 匹配: QQ车音响如何改装我喜欢听流行音乐.比较轻柔的.请指教:最好是指 
```

### 随机投影

随机投影，选择一个随机超平面（由法线单位向量r定义）对输入向量进行Hash处理，给定输入向量$v$和一个$r$定义的超平面，使$h(v) = sgn(v ⋅ r) $，$h(v)=±1$取决于v在超平面的哪侧。
    
对于向量$v,u$，存在$P_r[h(v) = h(u)] = 1 - \frac{θ(v,u)}{π}$，$\frac{θ(v,u)}{π}$与$1-cos(θ(v,u))$成正比，即两个向量位于超平面同一侧的概率与它们之间的余弦距离成正比。

将出现在超平面负侧的点分配值0，出现在正侧的点数据分配值1，通过将向量与超平面法线向量进行内积运算，可以确定向量位于超平面哪一侧，如果两个向量共享相同方向则其内积为正，如不共享相同方向，则为负，两个向量完全垂直其内积为0，将其与负侧向量分组。

单个二进制不能告诉我们关于向量相似性的信息，当我们添加更多超平面时，编码信息量会迅速增加，通过使用这些超平面将向量投影到低维空间，从而生成了新的散列向量。

使用SBERT模型生成输入向量，按输入向量维度生成随机超平面的发小向量。
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('DMetaSoul/sbert-chinese-general-v2')
sentence_embeddings = model.encode(sentences)
```
```python
In []:  nbits = 16 #超平面数量
        d = sentence_embeddings.shape[1]

        plane_norms = np.random.rand(nbits, d) - .5
        plane_norms

Out []: array([[-0.18880271, -0.18412385,  0.12060263, ...,  0.17444815,
                0.05596546,  0.48311439],
                [ 0.07260546, -0.49526173,  0.22410239, ..., -0.23384566,
                0.06609219,  0.20360384],
                [ 0.0005011 , -0.22629873, -0.05950452, ...,  0.12421859,
                0.45558575, -0.25598566],
                ...,
                [-0.09102182,  0.39235505,  0.17571101, ..., -0.22167728,
                0.26765463,  0.32822266],
                [-0.15349363, -0.407366  ,  0.40097011, ...,  0.3618219 ,
                -0.28554909, -0.27658608],
                [-0.02408455,  0.17956788,  0.02613843, ..., -0.1494725 ,
                0.22731098,  0.1298718 ]])
```
计算输入向量与超平面法线的内积，当内积大于0结果为1，其他情况为0，生成超平面数量大小的输出向量。
```python
def create_dot_vector(vec, plane_norms):
  dot = np.dot(vec, plane_norms.T)
  dot = dot > 0
  dot = dot.astype(int)
 
  return dot
```
```python
In []:  dot_arr = []
        for embedding in sentence_embeddings:
            dot_arr.append(create_dot_vector(embedding,plane_norms))
        dot_arr[0]
        
Out []: array([1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])
```
将降维后向量作为键计算哈希，将向量分配到不同桶中。
```python
In []:  hash_buckets = {}
        for i in range(len(dot_arr)):
            key = ''.join(dot_arr[i].astype(str))
            if key not in hash_buckets.keys():
                hash_buckets[key] = []
            hash_buckets[key].append(i)
        len(hash_buckets)

Out []: 2459
```
生成降维后向量所有可能形式，即所有可能桶的键。
```python
In []:  keys = []
        for i in range(1 << nbits):
            b = bin(i)[2:]
            # 按超平面数量补0
            b = '0' * (nbits - len(b))+b
            b = [int(i) for i in b]
            keys.append(b)
        keys = np.stack(keys)
        keys.shape

Out []: (65536, 16)
```
计算查询向量与桶可能键间的汉明距离，找到最近的桶，从桶中取出最相似的k个向量。
```python
def get_candidate(dot_vec, k):
  # 统计查询向量与所有桶的间的汉明距离并以列表示
  haming = np.count_nonzero(dot_vec != keys,axis=1).reshape(-1,1)
  # 将距离添加到键矩阵最后一列
  haming = np.concatenate((keys, haming), axis=1)
  # 将键矩阵行按汉明距离排序
  haming =haming[haming[:, -1].argsort()] 
  # 从桶中取出k个向量的对应位置
  vec_ids = []
  for row in haming:
    key = ''.join(row[:-1].astype(str))
    bucket = hash_buckets.get(key)
    if bucket is None:
      continue
    vec_ids.extend(bucket)
    if len(vec_ids) > k:
      vec_ids = vec_ids[:k]
      break
  return vec_ids
```
```python
In []:  import faiss

        indexLSH = faiss.IndexLSH(d, nbits)
        indexLSH.add(sentence_embeddings)
        for i in np.random.permutation(len(sentences)).copy()[:5]:
            query_sentence = sentences[i]

            query_vector = model.encode([query_sentence])
            query_dot_vector = create_dot_vector(query_vector,plane_norms)
            ids = get_candidate(query_dot_vector,2)
            print('查询:', query_sentence)
            for idx in ids:
                print('   L2距离:',distance.euclidean(query_vector[0],sentence_embeddings[idx]),'匹配:',sentences[idx] )
             # 测试faiss结果
            D, I = indexLSH.search(query_vector, 2)
            print('----Faiss----')
            for idx in I[0]:
                print('   L2距离:',distance.euclidean(query_vector[0],sentence_embeddings[idx]),'匹配:',sentences[idx] )

Out []:  查询: 生产主管这职位好吗？我是刚毕业的大学生,应聘了一家大型外资企业的 
            L2距离: 17.197383880615234 匹配: 鬼王高手进谢谢···一共150点技能点,按以下加法少侠重击　　　 
            L2距离: 16.48116111755371 匹配: 午间指导（2007/10/31）今日大盘继续走好但股指呈现高开低 
            ----Faiss----
            L2距离: 16.745290756225586 匹配: 烦请各位大虾指点一下阵容，先谢过了后卫：理查兹，默特萨克，甘贝里 
            L2距离: 17.793703079223633 匹配: 关于西安交大考研本人学金融的，但是对西方经济学不感兴趣,想考西安 
        查询: 古典中国风纯音乐古琴，琵琶，古筝之类的像女子十二乐坊的曲子 
            L2距离: 20.520275115966797 匹配: 求一篇思想政治的论文要求:写一篇[关于大学生思想道德现状产生原因 
            L2距离: 17.825180053710938 匹配: 她比我大2岁还跟过二个男友现在她选择了我她今年26她的前男友21 
            ----Faiss----
            L2距离: 0.0 匹配: 古典中国风纯音乐古琴，琵琶，古筝之类的像女子十二乐坊的曲子 
            L2距离: 19.309078216552734 匹配: 做肉丸要用什么机器?是不是一台碎肉机及一台成型机就可以了呢?主要 
        查询: 西安人有买钻石投资的吗？我要在西安买钻石？西安有钻石买吗？哪家珠? 
            L2距离: 0.0 匹配: 西安人有买钻石投资的吗？我要在西安买钻石？西安有钻石买吗？哪家珠? 
            L2距离: 20.764333724975586 匹配: 进不了奇迹世界为什么总是提示文件升级失败然后就退出大区选择界面， 
            ----Faiss----
            L2距离: 0.0 匹配: 西安人有买钻石投资的吗？我要在西安买钻石？西安有钻石买吗？哪家珠? 
            L2距离: 19.437467575073242 匹配: 世界历史人物世界上有那些国家的君主可以和秦皇汉武，唐宗宋祖相提并 
        查询: 请问金蛋编码怎么申请啊???金蛋编码要如何申请??怎么样才能领到 
            L2距离: 0.0 匹配: 请问金蛋编码怎么申请啊???金蛋编码要如何申请??怎么样才能领到 
            L2距离: 17.646564483642578 匹配: 我是新手请问大侠怎么可以投好3分 
            ----Faiss----
            L2距离: 19.34611701965332 匹配: 搏击中两人抱在一起是为什麽？经常看见拳击或自由搏击之中会有两个人 
            L2距离: 18.23182487487793 匹配: 异地恋真得就那摸难吗???我该怎么来处理我这段异地的恋情??大家? 
        查询: 大成优选可买否请高手告知大成优选是否配售，现可买否，谢谢同党! 
            L2距离: 18.105226516723633 匹配: 的鼻子很大,而且没有鼻梁,难看死了,有没有办法变小或者鼻梁变挺? 
            L2距离: 16.970487594604492 匹配: f(x)在x=a处可导，f(x)的绝对值在x=a处不可导的条件？ 
            ----Faiss----
            L2距离: 17.72905921936035 匹配: 斯特冈上市很久了，有没有人用过斯特冈喷剂？ 
            L2距离: 19.630430221557617 匹配: 毛孔粗大怎么办我脸上长豆豆,出油,额头,鼻子上,脸上都有毛孔,很 
```

<a href="https://colab.research.google.com/github/nananatsu/blog/blob/master/lsh.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

参考：
  - [Faiss: The Missing Manual](https://www.pinecone.io/learn/faiss-tutorial/)