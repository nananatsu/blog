实现raft集群中查询方法，查询时需要保证已提交的数据是最新的，在raft通过ReadIndex保证数据最新
- leader数据总是最新的，查询时要保证当前leader是一个合法的leader（向集群追加一条日志）；
- 为优化读取性能，一般会让follower也提供数据查询功能，follower在读取前需保证数据提交进度和leader一致；
```go
func (s *RaftServer) get(key []byte) ([]byte, error) {

	var commitIndex uint64
	var err error

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	if s.node.IsLeader() {
		start := time.Now().UnixNano()
		if start < s.leaderLease {
			commitIndex = s.node.GetLastLogIndex()
		} else {
			commitIndex, err = s.readIndex(ctx)
			if err == nil {
				s.leaderLease = start + int64(s.node.GetElectionTime())*1000000000
			}
		}
	} else {
		commitIndex, err = s.readIndex(ctx)
	}

	if err != nil {
		return nil, err
	}

	err = s.node.WaitIndexApply(ctx, commitIndex)
	if err != nil {
		return nil, err
	}

	return s.storage.GetValue(s.encoding.DefaultPrefix(key)), nil
}