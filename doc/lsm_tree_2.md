## 使用go实现LSM Tree (2) - memtable & wal
---

在上篇文章中我们实现了SSTable文件的写入，现在我们需要在内存中实现memtable，并在memtable达到阈值时将其冻结，转换为SSTable写入磁盘。

### 使用跳表实现memtable
---
我们选择跳表作为memtable实现
- 跳表是一种分层数据结构，最下层是一个有序链表，链表上方存在几层索引
- 下层元素以一定概率p（一般为1/2、1/4）出现在上层
- 查询时从顶层所以开始，水平前进，直到元素大于等于查询目标
     - 如元素等于目标，得到查询结果
     - 如元素大于目标，回退到上一个元素，垂直下降一层，继续水平前进，重复该过程

<img src=./imgs/lsm/Skip_list.png width=80% />

定义跳表结构，将键值对数据保存到一个字节切片kvData，使用一个索引切片kvNode记录键值对位置，切片记录内容如下：

<img src=./imgs/lsm/Skip_list_format.png width=100% />

- 0 键值对在字节切片中的位置
- 1 键值对中键的长度
- 2 键值对中值得长度
- 3 当前键值对所在层高度
- 4 ~ (4 + h - 1) 键值对在各层中下一跳在索引切片的位置

```go
const (
	nKV     = iota
	nKey    // key偏移
	nVal    // value偏移
	nHeight // 高度偏移
	nNext   // 下一跳位置偏移
)

type SkipList struct {
	mu     sync.RWMutex
	rand   *rand.Rand // 随机函数,判断数据是否存在于某层
	kvData []byte     // 跳表实际数据
	// 0 kvData 偏移
	// 1 key长度
	// 2 value长度
	// 3 高度
	// 4 ... 4+h-1 各高度下一跳位置
	kvNode    []int
	maxHeight int             // 最大高度
	prevNode  [tMaxHeight]int // 上一跳位置,查询数据时回溯
	kvSize    int             // 数据大小
}
```
实现跳表查询方法，找到键值对的位置或键值对应该在的位置(添加键值对时使用)，对应键查询只需依据键值对位置，取出值即可
- 首先从索引切片取得最高层的第一个键值对在字节切片位置
- 在字节切片取出对应键值对的键与查询目标比较
    - 相等，则找到目标，返回当前键值对位置、查询失败
    - 当前键小于目标，更新当前位置为原下一跳位置，继续查询
    - 当前键大于目标，将高度减一，记录当前层的键值对位置，到下一层继续查询，如当前已是最下层，则返回当前键值对位置、查询失败
        - 额外保存当前层的键值对位置，使得添加键值对时能够依据该位置直接将新键值对插入各层链表，因prevNode在跳表中共享，使用时需加锁
```go
func (s *SkipList) getNode(key []byte) (int, bool) {
	n := 0
	h := s.maxHeight - 1

	for {
		// 获取下一跳位置
		next := s.kvNode[n+nNext+h]
		cmp := 1

		// 下一跳存在，获取键与请求键比较
		if next != 0 {
			keyStart := s.kvNode[next]
			keyLen := s.kvNode[next+nKey]
			cmp = bytes.Compare(s.kvData[keyStart:keyStart+keyLen], key)
		}

		// 找到请求键
		if cmp == 0 {
			return next, true
		} else if cmp < 0 { // 当前键小于请求键,继续下一跳
			n = next
		} else { // 当前键大于请求键，不前进到下一跳，降低高度查询
			s.prevNode[h] = n
			if h > 0 {
				h--
			} else { // 已到最底层，请求键不存在
				return next, false
			}
		}
	}
}

func (s *SkipList) Get(key []byte) []byte {
	s.mu.RLock()
	defer s.mu.RUnlock()

	n, b := s.getNode(key)
	if b { //找到键，返回数据
		keyStart := s.kvNode[n]
		keyLen := s.kvNode[n+nKey]
		valueLen := s.kvNode[n+nVal]

		return s.kvData[keyStart+keyLen : keyStart+keyLen+valueLen]
	} else {
		return nil
	}

}
```
实现键值对添加方法
- 将键值对追加导致字节切片
- 查询当前跳表，找到插入位置
    - 键已存在，直接更新索引切片中的键对应偏移
    - 键不存在，使用随机函数，概率选择1/4，生成键值对所在位置高度
        - 当随机高度大于当前跳表高度时，新键值对是新层的起始，初始化prevNode中新层的数据
        - 通过prevNode遍历指定高度及以下层链表键值对插入位置，更新链表指针，使新键值对前一键值对指向新键值对，新键值对指向原后续键值对
```go
func (s *SkipList) Put(key []byte, value []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()

	lenv := len(value)
	lenk := len(key)
	// 查询键位置
	n, b := s.getNode(key)
	keyStart := len(s.kvData)
	// 追加键值对
	s.kvData = append(s.kvData, key...)
	s.kvData = append(s.kvData, value...)

	// 键已存在，更新值与偏移位置
	if b {
		s.kvSize += lenv - s.kvNode[n+nVal]
		s.kvNode[n] = keyStart
		s.kvNode[n+nVal] = lenv
        return
	}

	// 生成随机高度
	h := s.randHeight()
	if h > s.maxHeight {
		for i := s.maxHeight; i < h; i++ {
			s.prevNode[i] = 0
		}
		s.maxHeight = h
	}

	n = len(s.kvNode)
	// 添加偏移到指定高度
	s.kvNode = append(s.kvNode, keyStart, lenk, lenv, h)
	for i, node := range s.prevNode[:h] {
		m := node + nNext + i
		s.kvNode = append(s.kvNode, s.kvNode[m])
		s.kvNode[m] = n
	}

	s.kvSize += lenk + lenv
}

func (s *SkipList) randHeight() (h int) {
	const branching = 4
	h = 1
	for h < tMaxHeight && s.rand.Int()%branching == 0 {
		h++
	}
	return
}
```
实现一个迭代器遍历跳表，直接遍历跳表第0层，返回键值对数据。
```go
type SkipListIter struct {
	sl         *SkipList
	node       int    // 当前位置
	Key, Value []byte // 键值对数据
}

// 获取下一条数据
func (i *SkipListIter) Next() bool {

	// 下一跳数据
	i.node = i.sl.kvNode[i.node+nNext]
	// 存在下一条数据
	if i.node != 0 {
		// 解析键值数据
		keyStart := i.sl.kvNode[i.node]
		keyEnd := keyStart + i.sl.kvNode[i.node+nKey]
		valueEnd := keyEnd + i.sl.kvNode[i.node+nVal]

		i.Key = i.sl.kvData[keyStart:keyEnd]
		i.Value = i.sl.kvData[keyEnd:valueEnd]
		return true
	}
	return false
}

// 将跳表包装为迭代器
func NewSkipListIter(sl *SkipList) *SkipListIter {
	return &SkipListIter{sl: sl}
}
当跳表达到阈值后，不再向其写入数据，而通过迭代器遍历键值对通过SSTable Writer写入文件，这样就完成了lsm树内存写入磁盘部分。

```
### 实现WAL
---
memtable只存在于内存中，达到阈值后才以SSTable形式持久化到硬盘，一旦发生程序崩溃/断电就会导致数据丢失，我们一般通过WAL(Write-ahead logging，预写式日志)来进行保证数据原子性、持久性，WAL将所有操作先记录到磁盘再执行操作，这样在崩溃后依据WAL文件还原数据。对应到memtable，在将键值对写入跳表前先将键值对记录到磁盘，磁盘中以页作为单位读写，页大小默认为4k，在wal写入时也按按块为单位写入，块大小是页大小的倍数。

将键值对按 键长度|值长度|键|值 格式记录，块中键值对连续排列，块前7字节保存元数据：CRC、数据长度、块类型
<img src=./imgs/lsm/wal_block_format.png width=60% />

因键值对大小不确定，导致一条数据再块中可能完整也可能不完整，依据数据完整程度将块分为下述几种类型：
- kFull 数据完整
- kFirst 块最后一条数据只有起始部分
- kMiddle 整个块都是某条数据的中间部分
- kLast 块第一条数据是前面一个/多个块最后一条数据的结尾部分
```go
const (
	kFull = iota
	kFirst
	kMiddle
	kLast
)
```
定义WAL写入器:
- 写入时将数据加到缓冲，当缓冲达到块大小时写入磁盘
- 同时间隔一定时间检查是否有块未写入磁盘，将块填充到指定大小再写入磁盘
```go
type WalWriter struct {
	mu            sync.RWMutex
	dir           string
	seqNo         int
	fd            *os.File
	header        [20]byte
	buf           *bytes.Buffer
	prevBlockType uint8
	logger        *zap.SugaredLogger
}
```
实现WAL写入方法:
- 将键值对编码为 键长度|值长度|键|值 格式
- 块大小固定，在写入前需判断块是否有足够空间写入
    - 空间足够，将数据写入块，写入完成后检查块剩余大小，当剩余空间不足进行下次写入时，将块写入磁盘
    - 空间不足，将数据截断，分为两个部分，一部分填充到当前块，一部分写入下一块
```go
func (w *WalWriter) Write(key, value []byte) {
	w.mu.Lock()
	defer w.mu.Unlock()

	n := binary.PutUvarint(w.header[0:], uint64(len(key)))
	n += binary.PutUvarint(w.header[n:], uint64(len(value)))
	length := len(key) + len(value) + n

	b := make([]byte, length)
	copy(b, w.header[:n])
	copy(b[n:], key)
	copy(b[n+len(key):], value)

	size := walBlockSize - w.buf.Len()
	if size < length {
		w.buf.Write(b[:size])
		w.PaddingBlock(size-length, false)
		w.buf.Write(b[size:])
	} else {
		w.buf.Write(b)
		w.PaddingBlock(size-length, false)
	}
}
```
实现块空间填充方法
- 当空间已满无需填充时，判断块类型，将块写入到磁盘
- 当空间剩余小于7(uint64变长写入二进制最大占7字节)，块无法满足下次写入，填充剩余字节，判断块类型，将块写入到磁盘
    - 定时写入块时，强制填充剩余空间以写入
```go
func (w *WalWriter) PaddingBlock(remian int, force bool) {
	var blockType uint8
	if remian < 0 {
		if w.prevBlockType == kFirst || w.prevBlockType == kMiddle {
			blockType = kMiddle
		} else {
			blockType = kFirst
		}
		w.WriteBlock(blockType, uint16(w.buf.Len())-7)
		w.prevBlockType = blockType
	} else if remian < 7 || force {
		w.buf.Write(make([]byte, remian))
		if w.prevBlockType == kFirst || w.prevBlockType == kMiddle {
			blockType = kLast
		} else {
			blockType = kFull
		}
		w.WriteBlock(blockType, uint16(w.buf.Len()-remian-7))
		w.prevBlockType = blockType
	}
}
```
实现实际磁盘写入方法
- 将数据长度、块类型写入缓冲指定位置
- 计算块CRC校验码，再将CRC写入到块头，将块写入文件
- 截断缓冲前7字节外数据，以便继续使用
```go
func (w *WalWriter) WriteBlock(blockType uint8, length uint16) {

	data := w.buf.Bytes()
	binary.LittleEndian.PutUint16(data[4:6], length)
	data[6] = byte(blockType)
	crc := utils.Checksum(data[4:])
	binary.LittleEndian.PutUint32(data[:4], crc)
	w.fd.Write(data)

	w.buf.Truncate(7)
}
```
添加定时检查块方法，当块大小大于7时，强制将块写入磁盘
```go
func (rs *RaftStorage) checkFlush() {
	go func() {
		ticker := time.NewTicker(WAL_FLUSH_INTERVAL)
		for {
			select {
			case <-ticker.C:
				rs.walw.Flush()
			case <-rs.stopc:
				rs.walw.Flush()
				return
			}
		}
	}()
}

func (w *WalWriter) Flush() {
	if w.buf.Len() > 7 {
		w.mu.Lock()
		w.PaddingBlock(walBlockSize-w.buf.Len(), true)
		w.mu.Unlock()
	}
}
```
添加新建函数，创建WALWriter实例，在memtable写入前先写入WalWriter
 - 创建实例时，检查WAL大小是否与块大小对齐，否则将填充文件到块大小倍数，读取WAL文件按块大小读取，如当前文件大小不为块大小倍数（上次写入完成），后续写入块会在读取时错位
```go
func NewWalWriter(dir string, seqNo int, logger *zap.SugaredLogger) (*WalWriter, error) {

	walFile := path.Join(dir, strconv.Itoa(seqNo)+".wal")

	fd, err := os.OpenFile(walFile, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		return nil, fmt.Errorf("打开 %d.wal 失败: %v", seqNo, err)
	}

	w := &WalWriter{
		dir:    dir,
		seqNo:  seqNo,
		fd:     fd,
		buf:    bytes.NewBuffer(make([]byte, 7)),
		logger: logger,
	}

	w.PaddingFile()

	return w, nil
}

func (w *WalWriter) PaddingFile() {
	w.mu.Lock()
	defer w.mu.Unlock()

	info, _ := w.fd.Stat()
	n := info.Size() % walBlockSize
	if n > 0 {
		if _, err := w.fd.Write(make([]byte, walBlockSize-n)); err != nil {
			w.logger.Warnf("填充未完成写入文件块失败：%d", err)
		}
	}
}
```
WAL读取还原需拿到对应WAL文件，按定义的块大小读取，校验CRC，按块类型、数据长度将键值对二进制表现形式，再解码取得原始键值对。
定义WAL读取器，每次读取一个块
```go
type WalReader struct {
	fd    *os.File
	block []byte
	data  []byte
	buf   *bytes.Buffer
}
```
实现块读取功能，每次读取将块缓冲填充满
```go
func (r *WalReader) Read() error {
	_, err := io.ReadFull(r.fd, r.block)
	if err != nil {
		return err
	}
	return nil
}
```
实现键值对解码函数，按顺序读取键长度、值长度，在从后续位置按长度读出键、值
```go
func ReadRecord(buf *bytes.Buffer) ([]byte, []byte, error) {
	keyLen, err := binary.ReadUvarint(buf)
	if err != nil {
		return nil, nil, err
	}

	valueLen, err := binary.ReadUvarint(buf)
	if err != nil {
		return nil, nil, err
	}

	key := make([]byte, keyLen)
	_, err = io.ReadFull(buf, key)
	if err != nil {
		return nil, nil, err
	}

	value := make([]byte, valueLen)
	_, err = io.ReadFull(buf, value)
	if err != nil {
		return nil, nil, err
	}
	return key, value, nil
}
```
实现WAL键值对遍历方法
- 循环读取块，直到读取结束，或得到一段完整数据
    - 读取块头CRC，对快进行校验，校验失败块已损坏，返回错误
    - 校验成功，继续读取块类型，并将块数据部分写入缓冲，如数据不完整便继续读取块，直到读到块的数据完整
- 尝试解码键值对数据，如解码成功则返回结果，调用方继续调用Next()取得下一条数据
- 当前缓冲读取结束，递归调用Next()加载后续块到缓冲
```go
func (r *WalReader) Next() ([]byte, []byte, error) {
	var prevBlockType uint8
	for r.buf == nil {
		err := r.Read()
		if err != nil {
			if err == io.EOF {
				return nil, nil, nil
			}
			return nil, nil, fmt.Errorf("读取预写日志块失败:%v", err)
		}
		crc := binary.LittleEndian.Uint32(r.block[0:4])
		length := binary.LittleEndian.Uint16(r.block[4:6])
		blockType := uint8(r.block[6])

		if crc == utils.Checksum(r.block[4:]) {
			switch blockType {
			case kFull:
				r.data = r.block[7 : length+7]
				r.buf = bytes.NewBuffer(r.data)
			case kFirst:
				r.data = make([]byte, length)
				copy(r.data, r.block[7:length+7])
			case kMiddle:
				if prevBlockType == kMiddle || prevBlockType == kFirst {
					d := r.block[7 : length+7]
					r.data = append(r.data, d...)
				}
			case kLast:
				if prevBlockType == kMiddle || prevBlockType == kFirst {
					r.data = append(r.data, r.block[7:length+7]...)
					r.buf = bytes.NewBuffer(r.data)
				}
			}
			prevBlockType = blockType
		} else {
			return nil, nil, fmt.Errorf("预写日志校验失败")
		}
	}

	key, value, err := ReadRecord(r.buf)
	if err == nil {
		return key, value, nil
	}

	if err != io.EOF {
		return nil, nil, fmt.Errorf("读取预写日志失败: %v", err)
	}

	r.buf = nil
	return r.Next()

}
```
实现新建函数，创建WalReader实例，调用Next()方法读取WAL文件中的键值对
```go
func NewWalReader(fd *os.File) *WalReader {
	return &WalReader{
		fd:    fd,
		block: make([]byte, walBlockSize),
	}
}
```
实现WAL还原跳表方法，逐个读取键值对，添加到跳表
```go
func Restore(walFile string) (*skiplist.SkipList, error) {
	fd, err := os.OpenFile(walFile, os.O_RDONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("打开预写日志文件%s 失败: %v", walFile, err)
	}

	sl := skiplist.NewSkipList()
	r := NewWalReader(fd)
	defer r.Close()

	for {
		k, v, err := r.Next()
		if err != nil {
			return sl, err
		}

		if len(k) == 0 {
			break
		}
		sl.Put(k, v)
	}

	return sl, nil
}
```

参考:
- [Bigtable: A Distributed Storage System for Structured Data](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/68a74a85e1662fe02ff3967497f31fda7f32225c.pdf)
- [leveldb](https://github.com/google/leveldb)
- [goleveldb](https://github.com/syndtr/goleveldb)