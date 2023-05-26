## 使用go实现LSM Tree (3) - compaction
---

之前我们已经实现的LSM树在在磁盘、内存的基本单元结构，现在我们来将这些基本单元来构成实际的树，及实现树节点的压缩合并。
LSM中压缩分两种：
- minor compaction：内存数据持久化，一次minor compaction会产生一个0层SSTable文件，多个SSTable文件间数据有重叠
	- 仅有minor compaction时，0层存在大量分散的有重叠的SSTable文件，一次查询可能需要读取全部文件
- merging compaction/major compaction：为了提高查询效率，将0层多个SSTable归并为1层没有数据重叠的SSTable文件，当仅有0层、1层时，随数据量增加每次1层要合并的文件会越来越多，为降低每次compaction的io开销需要继续分层

首先定义一个树的节点:
- 一个节点代表一个SSTable文件，对应一个SstReader
    - 文件名为level_seqNo_extra.sst
        - level 为节点位于树的第几层(从0开始)
        - seqNo 为文件在该层的序号，单调增加，越新的文件序号越大
        - extra 一些文件额外信息，在我们的实现中会额外保存一个raft状态机的信息，单独在LSM树中无作用
- 节点缓存SSTable对应的布隆过滤器、索引的内存表现形式，启动时读取到内存，以便在查询时直接使用不用在重新读取磁盘
```go
type Node struct {
	wg         sync.WaitGroup    // 等待外部读取文件完成
	sr         *SstReader        // sst文件读取
	filter     map[uint64][]byte // 布隆过滤器
	startKey   []byte            // 起始键
	endKey     []byte            // 结束键
	index      []*Index          // 索引数组
	Level      int               //lsm层
	SeqNo      int               // lsm 节点序号
	Extra      string            // 文件额外信息,手动添加
	FileSize   int64             //文件大小
	compacting bool              //已在合并处理
	// 遍历节点数据用
	curBlock int           // 当前读取块
	curBuf   *bytes.Buffer // 当前读取到缓冲
	prevKey  []byte        // 前次读取到键
	logger   *zap.SugaredLogger
}
```
基于节点定义LSM树如下：
- 使用二维切片来保存树结构
- LSM树中会出现重复数据，读取数据时以新的数据为准，层数越低的文件数据越新，各层编号越新的文件数据也越新
```go
type Tree struct {
	mu      sync.RWMutex
	conf    *Config
	tree    [][]*Node     // lsm
	seqNo   []int         // 各层sst文件最新序号
	compacc chan int      // 合并通知通道
	stopc   chan struct{} // 停止通知通道
	logger  *zap.SugaredLogger
}
```
然后我们来实现minor compaction，创建树第0层的节点，将内存的immutable memtable转换为SSTable文件
- 我们的memtable由跳表实现，SSTable转换过程可视为遍历跳表将键值对写入SstWriter；
- 每次写入都会重新创建一个新文件，文件名中的序号单调增加；
- 文件写入完成后将其作为树0层的节点加入到树中,
	- 0层，按序列号，将节点插入到切片对应位置;
	- 其他层，按节点起始key，插入到切片对应位置；
- 添加到树后通知检查0层节点数量是否已达到阈值;
```go
func (t *Tree) FlushRecord(sl *skiplist.SkipListIter, extra string) error {
	level := 0
	seqNo := t.NextSeqNo(level)

	file := formatName(level, seqNo, extra)
	w, err := NewSstWriter(file, t.conf, t.logger)
	if err != nil {
		return fmt.Errorf("创建sst writer失败: %v", err)
	}
	defer w.Close()

	// 遍历跳表，将键值对写入sst文件
	for sl.Next() {
		w.Append(sl.Key, sl.Value)
	}

	// 完成写入
	size, filter, index := w.Finish()

	// 添加节点到内存lsm
	node, err := NewNode(level, seqNo, extra, file, size, filter, index, t.conf)
	if err != nil {
		return fmt.Errorf("创建lsm节点失败: %v", err)
	}
	t.insertNode(node)
	// 检查level是否合并
	t.compacc <- level

	return nil
}

func (t *Tree) insertNode(node *Node) {
	t.mu.Lock()
	defer t.mu.Unlock()

	// 按序号将节点插入合适位置
	level := node.Level
	length := len(t.tree[level])
	if length == 0 {
		t.tree[level] = []*Node{node}
		return
	}

	if level == 0 {
		idx := length - 1
		for ; idx >= 0; idx-- {
			if node.SeqNo > t.tree[level][idx].SeqNo {
				break
			} else if node.SeqNo == t.tree[level][idx].SeqNo {
				t.tree[level][idx] = node
				return
			}
		}
		t.tree[level] = append(t.tree[level][:idx+1], t.tree[level][idx:]...)
		t.tree[level][idx+1] = node
	} else {
		for i, n := range t.tree[level] {
			cmp := bytes.Compare(n.startKey, node.startKey)
			if cmp < 0 {
				t.tree[level] = append(t.tree[level][:i+1], t.tree[level][i:]...)
				t.tree[level][i] = node
				return
			}
		}
		t.tree[level] = append(t.tree[level], node)
	}
}
```
接下来我们来实现major compaction
- 在每次0层写入SSTable文件，检查0层文件数量是否达到阈值，
- 0层合并后会写入下一层，每次合并后检查下一层数据大小是否达到阈值
- 数据合并时会取当前层的部分节点和下一层数据有重叠的节点，对全部数据排序去重，重新写入下一层，如数据大小超过下一层SSTable最大大小，则将其写入多个SSTable文件

实现合并节点挑选方法，选取需合并数据的节点
- 0层，每次选择将最旧的节点为检查条件， 其他层选择将前一半的节点为检查条件
- 统计得到检查条中键的起始值、结束值，遍历下一层及当前层节点，找出值域有交叉的节点，并使用下一层交叉节点的起始键/结束键扩大检查条件
- 节点加入切片后，各节点数据新旧有序（低层数据在后，0层数据节点序列号较大的在后），切片中下标越大的节点值越新
```go
func (t *Tree) PickupCompactionNode(level int) []*Node {
	t.mu.Lock()
	defer t.mu.Unlock()

	compactionNode := make([]*Node, 0)

	if len(t.tree[level]) == 0 {
		return compactionNode
	}

	// 第0层 ，第一个节点起始，结束键作为检查条件
	startKey := t.tree[level][0].startKey
	endKey := t.tree[level][0].endKey
	// 其他层，取前半节点起始，结束键作为检查条件
	if level != 0 {
		node := t.tree[level][(len(t.tree[level])-1)/2]
		if bytes.Compare(node.startKey, startKey) < 0 {
			startKey = node.startKey
		}
		if bytes.Compare(node.endKey, endKey) > 0 {
			endKey = node.endKey
		}
	}

	// 检查与当前层、下一层有数据交叉的节点
	for i := level + 1; i >= level; i-- {
		for _, node := range t.tree[i] {
			if node.index == nil {
				continue
			}
			nodeStartKey := node.index[0].Key
			nodeEndKey := node.index[len(node.index)-1].Key
			if bytes.Compare(startKey, nodeEndKey) <= 0 && bytes.Compare(endKey, nodeStartKey) >= 0 && !node.compacting {
				compactionNode = append(compactionNode, node)
				node.compacting = true
				if i == level+1 {
					if bytes.Compare(nodeStartKey, startKey) < 0 {
						startKey = nodeStartKey
					}
					if bytes.Compare(nodeEndKey, endKey) > 0 {
						endKey = nodeEndKey
					}
				}
			}
		}
	}
	return compactionNode
}
```
现在我们得到了了多个节点，每个节点中数据有序，可以使用归并排序对数据排序，每次取各节点中最小值得最小值，写入新节点。
定义一个链表，用来排序各节点最小值，最小值排序使用插入排序方式。
```go
type Record struct {
	Key   []byte  // 键
	Value []byte  // 值
	Idx   int     // 数据来源数组下标
	next  *Record // 下一数据位置
}
```
实现添加键值对到链表方法，
- 出现相同键的数据时，依据节点的层数、序列号判断保留哪条数据，层数越低数据越新，同层中序列号越大数据越新，在选取合并节点时已按前述规则排序，在切片中下标越大的节点数据越新，添加键值对时传入键值对来源节点在切片的下标，表示值得新旧程度
- 键不同时按SSTable中键的比较规则，进行排序插入当前链表
```go
func (r *Record) push(key, value []byte, idx int) (*Record, int) {
	h := r
	cur := r
	var prev *Record
	for {
		if cur == nil {
			// 添加数据到链表尾
			if prev != nil {
				prev.next = &Record{Key: key, Value: value, Idx: idx}
			} else { // 添加数据到链表头
				h = &Record{Key: key, Value: value, Idx: idx}
			}
			break
		}

		cmp := bytes.Compare(key, cur.Key)
		// 链表存在相同键数据,保留更新来源数据
		if cmp == 0 {
			// 节点按旧到新排序，下标更大表示数据更新
			if idx >= r.Idx {
				old := cur.Idx
				cur.Key = key
				cur.Value = value
				cur.Idx = idx
				return h, old
			} else {
				return h, idx
			}
		} else if cmp < 0 { // 新增键小于当前位置键，已找到目标位置插入
			if prev != nil {
				prev.next = &Record{Key: key, Value: value, Idx: idx, next: cur}
			} else {
				h = &Record{Key: key, Value: value, Idx: idx, next: cur}
			}
			break
		} else { // 新增键大于当前位置键，继续查找
			prev = cur
			cur = cur.next
		}
	}
	return h, -1
}
```
实现从待合并节点切片指定位置读取数据，加入排序链表
- 从指定节点每次读入一条数据，插入到链表
- 当新键在链表中已存在，且旧键被替换时，被替换的键值对所在节点需再读入一条数据，包装链表中存在所以待合并节点的数据，直到某个节点数据已读取完毕
```go
func (r *Record) Fill(source []*Node, idx int) *Record {
	record := r
	// 读取节点数据
	k, v := source[idx].nextRecord()
	if k != nil {
		// 添加数据到链表
		record, idx = record.push(k, v, idx)

		//	如存在键被替换，重新填充被替换来源数据
		for idx > -1 {
			k, v := source[idx].nextRecord()
			if k != nil {
				record, idx = record.push(k, v, idx)
			} else {
				idx = -1
			}
		}
	}
	return record
}
```
实现从指定节点遍历键值对方法
- 当数据块未读入内存，从SSTable文件读取块解码存到到节点缓冲
- 从块缓冲解码一条键值对（当缓冲读取完成递归调用本方法读取数据），返回调用方
```go
func (n *Node) nextRecord() ([]byte, []byte) {
	// 当前缓冲数据为空，加载数据
	if n.curBuf == nil {
		// 读取完成
		if n.curBlock > len(n.index)-1 {
			return nil, nil
		}

		// 读取数据块
		data, err := n.sr.ReadBlock(n.index[n.curBlock].Offset, n.index[n.curBlock].Size)
		if err != nil {
			if err != io.EOF {
				n.logger.Errorf("%s 读取data block失败: %v", n.Level, n.SeqNo, err)
			}
			return nil, nil
		}

		// 解析记录缓冲，更新相关属性
		record, _ := DecodeBlock(data)
		n.curBuf = bytes.NewBuffer(record)
		n.prevKey = make([]byte, 0)
		n.curBlock++
	}

	// 读取记录
	key, value, err := ReadRecord(n.prevKey, n.curBuf)
	if err == nil {
		n.prevKey = key
		return key, value
	}

	if err != io.EOF {
		n.logger.Errorf("%s 读取记录失败: %v", n.Level, n.SeqNo, err)
		return nil, nil
	}

	// 当前缓冲读取完成，加载下一缓冲
	n.curBuf = nil
	return n.nextRecord()
}
```
实现节点合并方法
- 按当前层，选取待合并节点
- 在下一层创建一个新的SSTable Writer，准备写入合并后数据
- 创建一个有序链表用来归并各节点数据
	- 从各节点取出最小数据，插入链表
	- 循环从链表取出最小值，写入SSTable，再从最小值的来源节点补一条数据到链表，继续循环
		- 当前SSTable文件写满后，完成当前文件的写入，将其作为一个节点加入LSM树，再创建一个新的SSTable Writer写入后续数据
- 当前层合并完成后，移除已合并的节点，删除对应文件，再检查下一层是否需合并
```go
func (t *Tree) compaction(level int) error {
	// 获取需合并节点
	nodes := t.PickupCompactionNode(level)
	lenNodes := len(nodes)
	if lenNodes == 0 {
		return nil
	}

	// 创建新节点写入
	nextLevel := level + 1
	seqNo := t.NextSeqNo(nextLevel)
	extra := nodes[lenNodes-1].Extra
	file := formatName(nextLevel, seqNo, extra)
	writer, err := NewSstWriter(file, t.conf, t.logger)

	if err != nil {
		t.logger.Errorf("%s 创建writer失败,无法合并lsm日志:%v", file, err)
		return err
	}

	var record *Record
	var files string
	maxNodeSize := t.conf.SstSize * int(math.Pow10(nextLevel))

	// 从各节点填充数据到链表
	for i, node := range nodes {
		files += fmt.Sprintf("%d_%d_%s.sst ", node.Level, node.SeqNo, node.Extra)
		record = record.Fill(nodes, i)
	}
	t.logger.Debugf("合并: %v", files)

	// 遍历链表归并节点数据到新节点
	for record != nil {
		writeCount++
		// 写入数据
		i := record.Idx
		writer.Append(record.Key, record.Value)
		// 从消费了数据的节点填充数据
		record = record.next.Fill(nodes, i)

		// 节点数据大于当前层节点大小，完成节点节点，新建节点再写入
		if writer.Size() > maxNodeSize {
			size, filter, index := writer.Finish()
			writer.Close()
			// 添加新节点到树
			node, err := NewNode(nextLevel, seqNo, extra, file, size, filter, index, t.conf)
			if err != nil {
				return fmt.Errorf("创建lsm节点失败: %v", err)
			}
			t.insertNode(node)
			// 创建新节点写入
			seqNo = t.NextSeqNo(nextLevel)
			file = formatName(nextLevel, seqNo, extra)
			writer, err = NewSstWriter(file, t.conf, t.logger)
			if err != nil {
				t.logger.Errorf("%s 创建writer失败,无法合并lsm日志:%v", file, err)
				return err
			}
		}
	}

	//完成节点写入
	size, filter, index := writer.Finish()
	// 添加到树
	node, err := NewNode(nextLevel, seqNo, extra, file, size, filter, index, t.conf)
	if err != nil {
		return fmt.Errorf("创建lsm节点失败: %v", err)
	}
	t.insertNode(node)
	t.removeNode(nodes)
	// 检查是否继续合并
	t.compacc <- nextLevel

	return nil
}

func (t *Tree) removeNode(nodes []*Node) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for _, node := range nodes {
		t.logger.Debugf("移除: %d_%d_%s.sst ", node.Level, node.SeqNo, node.Extra)
		for i, tn := range t.tree[node.Level] {
			if tn.SeqNo == node.SeqNo {
				t.tree[node.Level] = append(t.tree[node.Level][:i], t.tree[node.Level][i+1:]...)
				break
			}
		}
	}
	go func() {
		for _, n := range nodes {
			n.destory()
		}
	}()
}
```
合并压缩使用单独协程完成，防止阻塞主协程写入数据
- 0层依据节点数量判断阈值
- 其他层计算文件总大小，判断阈值
```go
func (t *Tree) CheckCompaction() {
	level0 := make(chan struct{}, 100)
	levelN := make(chan int, 100)

	// 第0层合并协程
	go func() {
		for {
			select {
			case <-level0:
				if len(t.tree[0]) > 4 {
					t.logger.Infof("Level 0 执行合并, 当前数量: %d", len(t.tree[0]))
					t.compaction(0)
				}
			case <-t.stopc:
				close(level0)
				return
			}
		}
	}()

	// 非0层合并协程
	go func() {
		for {
			select {
			case lv := <-levelN:
				var prevSize int64
				maxNodeSize := int64(t.conf.SstSize * int(math.Pow10(lv+1)))
				for {
					var totalSize int64
					for _, node := range t.tree[lv] {
						totalSize += node.FileSize
					}
					if totalSize > maxNodeSize && (prevSize == 0 || totalSize < prevSize) {
						t.compaction(lv)
						prevSize = totalSize
					} else {
						break
					}
				}
			case <-t.stopc:
				close(levelN)
				return
			}
		}
	}()

	// 合并通知处理
	go func() {
		for {
			select {
			case <-t.stopc:
				return
			case lv := <-t.compacc:
				if lv == 0 {
					level0 <- struct{}{}
				} else {
					levelN <- lv
				}
			}
		}
	}()
}
```
添加新建函数，创建LSM树实例，树创建后启动协程等待合并压缩任务
```go
func NewTree(conf *Config) *Tree {
	compactionChan := make(chan int, 100)
	levelTree := make([][]*Node, conf.MaxLevel)

	for i := range levelTree {
		levelTree[i] = make([]*Node, 0)
	}

	seqNos := make([]int, conf.MaxLevel)
	lt := &Tree{
		conf:    conf,
		tree:    levelTree,
		seqNo:   seqNos,
		compacc: compactionChan,
		stopc:   make(chan struct{}),
		logger:  conf.Logger,
	}

	lt.CheckCompaction()
	return lt
}
```
最后实现LSM树的还原函数，从磁盘文件重新构成内存中的树
```go
func RestoreTree(conf *Config) (*Tree, error) {
	lt := NewTree(conf)
	callbacks := []func(string, fs.FileInfo){
		func(name string, fileInfo fs.FileInfo) {
			err := lt.LoadNode(name)
			if err != nil {
				conf.Logger.Errorf("加载文件%s 到lsm树失败: %v", name, err)
			}
		},
	}

	if err := utils.CheckDir(conf.Dir, callbacks); err != nil {
		return lt, fmt.Errorf("还原LSM Tree状态失败: %v", err)
	}

	return lt, nil
}

func RestoreNode(level, seqNo int, extra string, file string, conf *Config) (*Node, error) {

	r, err := NewSstReader(file, conf)
	if err != nil {
		return nil, fmt.Errorf("%s 创建sst Reader: %v", file, err)
	}

	filter, err := r.ReadFilter()
	if err != nil {
		r.Close()
		return nil, fmt.Errorf("%s 读取过滤块失败: %v ", file, err)
	}

	index, err := r.ReadIndex()
	if err != nil {
		r.Close()
		return nil, fmt.Errorf("%s 读取索引失败: %v ", file, err)
	}

	return &Node{
		sr:       r,
		Level:    level,
		SeqNo:    seqNo,
		Extra:    extra,
		curBlock: 1,
		FileSize: r.IndexOffset + r.IndexSize + int64(conf.SstFooterSize),
		filter:   filter,
		index:    index,
		startKey: index[0].Key,
		endKey:   index[len(index)-1].Key,
		logger:   conf.Logger,
	}, nil
}
```

[完整代码](https://github.com/nananatsu/simple-raft/tree/master/pkg/lsm)

参考:
- [Bigtable: A Distributed Storage System for Structured Data](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/68a74a85e1662fe02ff3967497f31fda7f32225c.pdf)
- [leveldb](https://github.com/google/leveldb)
- [goleveldb](https://github.com/syndtr/goleveldb)