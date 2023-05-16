B+树在网络上有看到两种实现：

- 一种是wiki上显示的,一个node有n个key，n+1个pointer。<br/>
<img src=./imgs/Bplustree.png width=60% />
- 一种是在相关innodb的文章中出现的，一个node有n个key，n个pointer。<br/>
<img src=./imgs/B_Tree_Structure.png width=60% />

在网络上搜索为什么会出现两种B+树没有得到直接答案，猜测是在内存实现和将树落盘产生的差异，磁盘中数据以页作为读写单位，innodb基于页实现在磁盘的B+树，为了充分使用磁盘，innodb设计了特定数据结构存储B+树，为了适应该结构而对B+树做了一定变形。<br/>

innodb中key + value、key + child pointer被抽象为统一的record结构，在页中以升序排列，通常B+实现中节点的第一个指针在innodb的叶子页、非叶子页中都不是完整的record结构，将B+树稍作变形丢弃第一个左指针（给第一个左指针添加key）从而保证record都是有效的。<br/>
<img src=./imgs/INDEX_Page_Overview.png width=40% />

加载页数据后，依据System Records找到record偏移以访问record数据：<br/>
<img src=./imgs/INDEX_System_Records.png width=40% />
- infimum指向该页中key最小record的偏移<br/>
- supremum始终为0，页中key最大的记录的next record指向该位置<br/>
- record type判断该页记录的是指针还是数据<br/>
    - 数据类record key为表主键字段数据，value为非主键外字段<br/>
    - 指针类record key为子页中最小key，value为子页编号<br/>
    <img src=./imgs/Record_Clustered_Leaf.png width=40% /><br/>
    <img src=./imgs/Record_Clustered_Non_Leaf.png width=40% /><br/>

 <img src=./imgs/B_Tree_Page_Directory_Structure.png width=60% />

参考：
- <https://blog.jcole.us/2013/01/07/the-physical-structure-of-innodb-index-pages>
- <https://blog.jcole.us/2013/01/10/the-physical-structure-of-records-in-innodb>
- <https://blog.jcole.us/2013/01/14/efficiently-traversing-innodb-btrees-with-the-page-directory/>



