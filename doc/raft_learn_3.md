客户端在进行提案，需要leader在提案作为日志提交后响应客户端，在提交日志后通知等待队列
- 从follower读取需要数据时，需要Readindex机制，也需要等待指定日志提交完成
- 一条日志提交后，检查等待队列中日志编号小于当前提交日志的对象，通知日志提交完成，再从队列中移除已通知对象
```go
func (l *RaftLog) NotifyReadIndex() {
	cur := 0
	for _, wa := range l.waitQueue {
		if wa.index <= l.lastAppliedIndex {
			if wa.done {
				close(wa.ch)
			} else {
				select {
				case wa.ch <- struct{}{}:
					close(wa.ch)
				default:
					close(wa.ch)
				}
			}
			cur++
		} else {
			break
		}
	}

	if cur > 0 {
		l.waitQueue = l.waitQueue[cur:]
	}
}
```