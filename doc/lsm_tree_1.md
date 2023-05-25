## 使用go实现LSM Tree (1) - SSTable
---
在之前的文章中我们实现了raft的leader选举、日志同步功能，但是日志未持久化到硬盘，接下来我们实现lsm树来为raft添加存储。

lsm（log-structured merge-tree）日志结构合并树，将数据分层存储，数据总是合并到下一层。
通常的lsm树中会有两种数据结构：
- memtable/immutable memtable，键值对在内存的缓存结构，任意选择算法实现key有序即可；
- SSTable(Sorted Strings Table)，键值对在磁盘的存储形式，出自google bigtable论文，提供从key到value的持久化、有序、不可变映射，key、value都是任意字符串，一个SSTable由一系列块构成（块通常为64kb），SSTable文件尾不通常有索引定位块位置；

lsm树写入过程如下：
- 新数据写入memtable，当memtable达到阈值后，将该memtable标记为不可变，新数据写入到新的memtable中；
- 将immutable memtable组织为SSTable形式写入硬盘，每次写入创建新文件，多次memtable写入磁盘后，存在一系列按写入顺序排列的SSTable文件（level 0）；
- 当level 0的SSTable达到阈值后，选择一部分SSTable合并到下一层，合并时将重新分割sstble，保证多个文件key有序，当本层数据达到阈值，选取部分SSTable合并到下一层，树不断往下合并，每层的容量一般为上一层的10倍；

```
---------------------------------------------
        memtable
memory    -----------------------------------
        immutable memtable
---------------------------------------------
level 0   sst-0-0 sst-0-1 sst-0-2 ... sst-0-m   sst大小：s  总大小： m*s
----------------------------------------------
level 1   sst-1-0 sst-1-1 sst-1-2 ...           sst大小：s*10^1 总大小： m*s*10^1
----------------------------------------------
level 2   sst-2-0 sst-2-1 sst-2-2 ...           sst大小：s*10^2 总大小： m*s*10^2
----------------------------------------------
...
----------------------------------------------
level n  sst-n-0 sst-n-1 sst-n-2 ...            sst大小：s*10^n 总大小： m*s*10^n
----------------------------------------------
```

### 实现SSTable数据块
---
lsm树中保存键值对形式数据，键值对的长度都是不确定的，将键值对按 键长度|值长度|键|值 格式保存

<img src=./imgs/lsm/record_format_0.png width=40% />

SSTable中键是有序的，两个相邻键值对的键可能有一部分是一样的，可以在每个键值对键的部分只保存与上一个键值对不同的部分，减少空间占用

<img src=./imgs/lsm/record_format_1.png width=40% />

定义了键值对格式后，在键值对基础上将多条键值对组织为块
- 为方便在块中查询，将块划分为多个部分，每个部分保存一部分键值对，每部分键值对起始需有完整键数据
- 在块尾记录每部分起始偏移(称为restart point)
- 最后用固定字节记录restart point数量，这样可以读取块尾固定字节得到restart point数量，读取指定字节得到得到块中各个键值对分组的起始偏移，从而遍历块中键值对

<img src=./imgs/lsm/block_format_0.png width=60% />

为进一步利用磁盘空间，我们将块在写入前进行压缩，尾部记录CRC校验结果

<img src=./imgs/lsm/block_format_1.png width=30% />

定义块结构如下：
- 写入时将键值对写入record缓冲，将restart point写入trailer缓冲，最终将record与trailer合并得到完整块
```go
type Block struct {
	conf               *Config
	header             [30]byte      // 辅助填充 block、record 头
	record             *bytes.Buffer // 记录缓冲
	trailer            *bytes.Buffer // 块尾缓冲
	nEntries           int           // 数据条数
	prevKey            []byte        // 前次键
	compressionScratch []byte        // 压缩缓冲
}
```
实现键值对添加到块缓冲方法
- 当键值对在restart point后，将当前偏移记录到块尾，
- restart point处的共享键长度为0，之后的键需要与前一键比较得出可共享长度
- 按设计顺序将共享键长度、键剩余长度、值长度、键非共享部分、值写入块缓冲
```go
func (b *Block) Append(key, value []byte) {
	keyLen := len(key)
	valueLen := len(value)
	nSharePrefix := 0

	// 重启点，间隔一定量数据后，重新开始键共享
	if b.nEntries%b.conf.SstRestartInterval == 0 {
		// 重启点用4字节记录键对应偏移
		buf4 := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf4, uint32(b.record.Len()))
		b.trailer.Write(buf4)
	} else {
		nSharePrefix = SharedPrefixLen(b.prevKey, key)
	}

	// 按记录格式将记录写入记录缓冲
	n := binary.PutUvarint(b.header[0:], uint64(nSharePrefix))
	n += binary.PutUvarint(b.header[n:], uint64(keyLen-nSharePrefix))
	n += binary.PutUvarint(b.header[n:], uint64(valueLen))

	// data
	b.record.Write(b.header[:n])
	b.record.Write(key[nSharePrefix:])
	b.record.Write(value)

	b.prevKey = append(b.prevKey[:0], key...)
	b.nEntries++
}

func SharedPrefixLen(a, b []byte) int {
    i, n := 0, len(a)
    if n > len(b) {
        n = len(b)
    }
    for i < n && a[i] == b[i] {
        i++
    }
    return i
}
```
实现块压缩方法，压缩选择使用snappy库（提供非常快的压缩速度及合适的压缩率）
- 压缩前需统计restart point数量写入trailer，再将record和trailer合并为完整块数据
- 压缩后计算crc，添加到压缩块的最后，读取时用来校验数据是否损坏
```go
func (b *Block) compress() []byte {

	// 尾最后4字节记录重启点数量
	buf4 := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf4, uint32(b.trailer.Len())/4)
	b.trailer.Write(buf4)

	// 将重启点数据写入记录缓冲
	b.record.Write(b.trailer.Bytes())
	// 计算并分配压缩需要空间
	n := snappy.MaxEncodedLen(b.record.Len())
	if n > len(b.compressionScratch) {
		b.compressionScratch = make([]byte, n+b.conf.SstBlockTrailerSize)
	}

	// 压缩记录
	compressed := snappy.Encode(b.compressionScratch, b.record.Bytes())

	// 添加crc检验到块尾
	crc := utils.Checksum(compressed)
	size := len(compressed)
	compressed = compressed[:size+b.conf.SstBlockTrailerSize]
	binary.LittleEndian.PutUint32(compressed[size:], crc)

	return compressed
}

var crc32Table = crc32.MakeTable(crc32.Castagnoli)

func Checksum(data []byte) uint32 {
	return crc32.Checksum(data, crc32Table)
}
```

最后在实现将块写入到指定writer，在块写入完成后清空缓冲
```go
func (b *Block) FlushBlockTo(dest io.Writer) (uint64, error) {
	defer b.clear()

	n, err := dest.Write(b.compress())
	return uint64(n), err
}

func (b *Block) clear() {
	b.nEntries = 0
	b.prevKey = b.prevKey[:0]
	b.record.Reset()
	b.trailer.Reset()
}
```

### 实现SSTable
---

SSTable主要部分是由一系列数据块构成
- 为区分各数据块的起始偏移及快速遍历，在尾部加入索引定位到各个数据块，索引也使用块结构保存将两个数据块分隔，两个数据块的分隔键作为索引的key，前一个块的偏移、大小被作为索引的值
- 磁盘读取较慢，为了快速判断键不在SSTable中加入布隆过滤器，布隆过滤器同样使用块结构保存，对应数据块的起始偏移为键，数据块中键生成的布隆过滤器位数组为值
    - 布隆过滤器使用一个包含少量字节的位数组，判断一个值是否在集合中
    - 布隆过滤器得出值未在集合中时，实际一定不在，得出在集合时，实际可能在集合，也可能不在集合
- 使用固定字节记录布隆过滤器块起始偏移、大小及索引块起始偏移、大小
- 读取时从固定字节得到过滤器块位置、索引块位置，遍历时按索引找到数据块解压，再按数据块遍历方式读取，查询指定键时，与索引键进行比较得到可能存在的数据块，做按数据块偏移的的布隆过滤器位数组，检查键是否不在块中，如可能在块中，在解压数据块，按restart point查询

<img src=./imgs/lsm/sstable_format_0.png width=60% />

定义SSTable写入器
- 生成SSTable文件时，顺序写入数据块，缓存过滤器块及索引块，在数据块写入完成后再按顺序写入过滤器块、索引块、尾固定字节
```go
type SstWriter struct {
	conf            *Config
	fd              *os.File            // sst文件(写)
	dataBuf         *bytes.Buffer       // 数据缓冲
	filterBuf       *bytes.Buffer       // 过滤缓冲, key -> prev data block offset
	indexBuf        *bytes.Buffer       // 索引缓冲, offset->bloom fliter
	index           []*Index            // 索引数组,方便写入sst完成后直接加载到lsm树
	filter          map[uint64][]byte   // 过滤器map,方便写入sst完成后直接加载到lsm树
	bf              *filter.BloomFilter // 布隆过滤器生成
	dataBlock       *Block              // 数据块
	filterBlock     *Block              // 过滤器块
	indexBlock      *Block              // 索引块
	indexScratch    [20]byte            // 辅助byte数组，将uint64作为变长[]byte写入
	prevKey         []byte              // 前次key，生成分隔数据块的索引key
	prevBlockOffset uint64              // 前次数据块偏移, 生成分隔索引
	prevBlockSize   uint64              // 前次数据块大小, 生成分隔索引
	logger          *zap.SugaredLogger
}
```
实现键值对写入方法，调用块的写入方法将数据写入缓冲
- 当当前数据块为一个空块（新块）时，添加索引指向该数据块
- 将键记录到布隆过滤器，在将块写入时计算位数组
- 当数据块大小达到阈值，将数据块写入磁盘
```go
func (w *SstWriter) Append(key, value []byte) {
	// 数据块数据量为0,添加分隔索引
	if w.dataBlock.nEntries == 0 {
		skey := make([]byte, len(key))
		copy(skey, key)
		w.addIndex(skey)
	}

	// 添加数据到数据块、布隆过滤器
	w.dataBlock.Append(key, value)
	w.bf.Add(key)
	// 记录前次key，以便生成分隔索引
	w.prevKey = key

	// 数据块大小超过阈值，打包写入数据缓冲
	if w.dataBlock.Size() > w.conf.SstDataBlockSize {
		w.flushBlock()
	}
}
```
实现索引添加方法，添加一条索引到索引块
- 将数据块偏移、大小以以变长形式写入缓冲
- 依据排序规则，计算分隔键，该键需要大于等于上一个数据块的最后一个键，小于当前数据块的第一个键，查询是通过比较该键得知，键在之前还是之后
- 将分隔键、数据块偏移/大小写入索引块
```go
func (w *SstWriter) addIndex(key []byte) {
	n := binary.PutUvarint(w.indexScratch[0:], w.prevBlockOffset)
	n += binary.PutUvarint(w.indexScratch[n:], w.prevBlockSize)
	separator := GetSeparator(w.prevKey, key)

	w.indexBlock.Append(separator, w.indexScratch[:n])

	w.index = append(w.index, &Index{Key: separator, Offset: w.prevBlockOffset, Size: w.prevBlockSize})
}

func GetSeparator(a, b []byte) []byte {
	if len(a) == 0 {
		n := len(b) - 1
		c := b[n] - 1
		return append(b[0:n], c)
	}

	n := SharedPrefixLen(a, b)
	if n == 0 || n == len(a) {
		return a
	} else {
		c := a[n] + 1
		return append(a[0:n], c)
	}
}
```
实现布隆过滤器，计算数据块的键对应的位数组
- 布隆过滤器使用多个hash函数将值映射到位数组的多个位置(被映射到的位置置为1)
- 检查时使用同样方式将值映射到位数组多个位置，任意位置为0则值一定不在集合中，都为1时，值可能在集合中，也可能是集合中其他值映射到了这些位置，导致误判
- hash函数最佳的数量k由公式：$k= \frac{m}{n}{ \ln 2}$，m所需位数，n插入元素数量
- 双重Hash能够达成和多个hash函数一致的效果，双重hahs选取两个独立hash函数，先用第一个hash函数计算hash，再用第二个hash函数计算步进，得到一个新的hash值，$g_i(x) = h_1(x) + ih_2(x) \bmod m$
	- [Less Hashing, Same Performance:
Building a Better Bloom Filter](https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf)

<img src=./imgs/lsm/Bloom_filter.svg width=60% />

定义布隆过滤器
- 初始时设置所需位数
- 在向数据块添加键值对时，将键添加到过滤器键hash切片，在数据块写入后再计算位数组
```go
type BloomFilter struct {
	bitsPerKey int
	hashKeys   []uint32
}
```
实现添加键到布隆过滤器中
- 使用双重hash代替k个hash函数，将键添加到切片是添加第一个hash后的结果，无需实际值
	- 第一个hash函数选取了MurmurHash3，这是一种计算很快的hash函数
	- 第二个hash函数使用 $g_i(x) = h_i(x) >> 17 | h_i(x) << 15$
```go
func (b *BloomFilter) Add(key []byte) {
	b.hashKeys = append(b.hashKeys, MurmurHash3(key, 0xbc9f1d34))
}
```
实现计算位数组
- 通过公式算出最佳hash函数数量，对应双重hash中k次步进（1<=k<=30）
- 从切片中取出第一个hash函数结果，进行k次步进，并将结果映射到位数组将指定位置为1
```go
func (b *BloomFilter) Hash() []byte {
	n := len(b.hashKeys)
	k := uint8(b.bitsPerKey * 69 / (100 * n))

	if k < 1 {
		k = 1
	} else if k > 30 {
		k = 30
	}
	// 布隆过滤器bit数组长度
	nBits := uint32(n * b.bitsPerKey)

	if nBits < 64 {
		nBits = 64
	}

	nBytes := (nBits + 7) / 8
	nBits = nBytes * 8

	dest := make([]byte, nBytes+1)
	dest[nBytes] = k

	// hash1(key)+i*hash2(key)
	for _, h := range b.hashKeys {
		delta := (h >> 17) | (h << 15)
		for i := uint8(0); i < k; i++ {
			bitpos := h % nBits
			dest[bitpos/8] |= 1 << (bitpos % 8)
			h += delta
		}
	}
	return dest
}
```
实现数据块写入数据缓冲,写入时一并计算布隆过滤器位数组，添加到过滤器块，之后重置布隆过滤器，以便下个数据块使用
```go
func (w *SstWriter) flushBlock() {
	var err error
	// 记录当前数据缓冲大小，在下次添加分隔索引时使用
	w.prevBlockOffset = uint64(w.dataBuf.Len())
	n := binary.PutUvarint(w.indexScratch[0:], uint64(w.prevBlockOffset))

	// 生成布隆过滤器Hash，记录到map: 数据块偏移->布隆过滤器
	filter := w.bf.Hash()
	w.filter[w.prevBlockOffset] = filter
	// 添加数据块偏移->布隆过滤器关系到过滤块
	w.filterBlock.Append(w.indexScratch[:n], filter)
	// 重置布隆过滤器
	w.bf.Reset()

	// 将当前数据块写入数据缓冲
	w.prevBlockSize, err = w.dataBlock.FlushBlockTo(w.dataBuf)
	if err != nil {
		w.logger.Errorln("写入data block失败", err)
	}
}

func (b *BloomFilter) Reset() {
	b.hashKeys = b.hashKeys[:0]
}
```
实现数据落盘，键值对写入完成后，将数据缓冲、过滤器块、索引块写入到磁盘
```go
func (w *SstWriter) Finish() (int64, map[uint64][]byte, []*Index) {

	if w.bf.KeyLen() > 0 {
		w.flushBlock()
	}
	// 将过滤块写入过滤缓冲
	if _, err := w.filterBlock.FlushBlockTo(w.filterBuf); err != nil {
		w.logger.Errorln("写入filter block失败", err)
	}

	// 添加分隔索引，将索引块写入索引缓冲
	w.addIndex(w.prevKey)
	if _, err := w.indexBlock.FlushBlockTo(w.indexBuf); err != nil {
		w.logger.Errorln("写入index block失败", err)
	}

	// 生成sst文件footer，记录各部分偏移、大小
	footer := make([]byte, w.conf.SstFooterSize)
	size := w.dataBuf.Len()
	// metadata 索引起始偏移，整体长度
	n := binary.PutUvarint(footer[0:], uint64(size))
	n += binary.PutUvarint(footer[n:], uint64(w.filterBuf.Len()))
	size += w.filterBuf.Len()
	n += binary.PutUvarint(footer[n:], uint64(size))
	n += binary.PutUvarint(footer[n:], uint64(w.indexBuf.Len()))
	size += w.indexBuf.Len()
	size += w.conf.SstFooterSize

	// 将缓冲写入文件
	w.fd.Write(w.dataBuf.Bytes())
	w.fd.Write(w.filterBuf.Bytes())
	w.fd.Write(w.indexBuf.Bytes())
	w.fd.Write(footer)

	// 返回lsm树属性
	return int64(size), w.filter, w.index
}
```
添加新建函数，新建SSTable Writer打开指定文件，通过Append方法添加键值对，调用Finish方法将数据写入文件。
```go
func NewSstWriter(file string, conf *Config, logger *zap.SugaredLogger) (*SstWriter, error) {
	fd, err := os.OpenFile(path.Join(conf.Dir, file), os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		return nil, fmt.Errorf("创建 %s 失败: %v", file, err)
	}

	return &SstWriter{
		conf:        conf,
		fd:          fd,
		dataBuf:     bytes.NewBuffer(make([]byte, 0)),
		filterBuf:   bytes.NewBuffer(make([]byte, 0)),
		indexBuf:    bytes.NewBuffer(make([]byte, 0)),
		filter:      make(map[uint64][]byte),
		index:       make([]*Index, 0),
		bf:          filter.NewBloomFilter(10),
		dataBlock:   NewBlock(conf),
		filterBlock: NewBlock(conf),
		indexBlock:  NewBlock(conf),
		prevKey:     make([]byte, 0),
		logger:      logger,
	}, nil
}

```
本篇讲解了lsm树中SSTable文件格式，实现了一个SSTable Writer生成SSTable文件，后续将继续memtable实现及SSTable的压缩合并。


参考:
- [Bigtable: A Distributed Storage System for Structured Data](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/68a74a85e1662fe02ff3967497f31fda7f32225c.pdf)
- [leveldb](https://github.com/google/leveldb)
- [goleveldb](https://github.com/syndtr/goleveldb)
