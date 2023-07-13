
#### 准备硬盘

检查新硬盘在系统中名称，第一块硬盘为sda，第二块为sdb，第三块为sdc，以此类推
```shell
sudo fdisk -l
```
按看到的硬盘名称，格式化新硬盘，这里原本有一块硬盘，新加了一块硬盘对应sdb
```shell
sudo parted /dev/sdb
```
进入parted命令行后，设置分区表格式gpt
```shell
mklabel gpt
```
设置分区表格式后，就可以格式化硬盘了，退出parted
```shell
quit
```
将硬盘格式化为ext4
```shell
sudo mkfs.ext4 /dev/sdb
```
临时挂载硬盘到/mnt/sdb
```shell
sudo mkdir /mnt/sdb
mount /dev/sdb /mnt/sdb
```
编辑/etc/fstab，启动后自动挂载硬盘
```
/dev/sdb /mnt/sdb ext4 defaults 0 0
```

#### 配置LocalPath卷使用新硬盘

修改LocalPath配置配置config.json，设置添加了新硬盘的节点能够挂载卷到新硬盘
- 当一个节点有配置到nodePathMap时，会按节点配置的paths路径创建卷，未配置时按DEFAULT_PATH_FOR_NON_LISTED_NODES的路径创建卷
- paths为[]时，拒绝在在该节点创建卷，
- paths中有多个路径则随机选择一个路径创建卷
    - 配置StorageClass时指定路径参数（路径需在nodePathMap中存在），可选择特定路径创建卷，如下配置时将使用/mnt/sdb/local-path-provisioner路径创建卷（调度节点没有该目录会失败并在节点一直重试）
    ```yaml
    apiVersion: storage.k8s.io/v1
    kind: StorageClass
    metadata:
      name: local-path-new-ssd
    provisioner: rancher.io/local-path
    parameters:
      nodePath: /mnt/sdb/local-path-provisioner
    volumeBindingMode: WaitForFirstConsumer
    reclaimPolicy: Delete
    ```
如下添加配置，节点server-dev-3，路径/mnt/sdb/local-path-provisioner
```yaml
kind: ConfigMap
apiVersion: v1
metadata:
  name: local-path-config
  namespace: persistent-storage
data:
  config.json: |-
    {
            "nodePathMap":[
            {
                    "node":"DEFAULT_PATH_FOR_NON_LISTED_NODES",
                    "paths":["/opt/local-path-provisioner"]
            },
            {
                    "node":"server-dev-3",
                    "paths":["/mnt/sdb/local-path-provisioner"]
            }
            ]
    }
  setup: |-
    #!/bin/sh
    set -eu
    mkdir -m 0777 -p "$VOL_DIR"
  teardown: |-
    #!/bin/sh
    set -eu
    rm -rf "$VOL_DIR"
  helperPod.yaml: |-
    apiVersion: v1
    kind: Pod
    metadata:
      name: helper-pod
    spec:
      containers:
      - name: helper-pod
        image: busybox
        imagePullPolicy: IfNotPresent
```

在server-dev-3创建新LocalPath卷，测试是否能够在/mnt/sdb/local-path-provisioner下创建卷目录
```yaml
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: test-pvc
  namespace: test
  annotations:
    volume.kubernetes.io/selected-node: server-dev-3
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: local-path
```

#### 迁移数据

将挂载在旧硬盘的LocalPath卷数据迁移到新硬盘上的新LocalPath卷

为具有新硬盘的节点打上标签

```shell
kubectl label nodes server-dev-3 new-ssd=true
```
修改pod（Deployment/StatefulSet）配置，添加额外nodeSelector配置使pod调度到具有新硬盘的节点：

```yaml
nodeSelector:
    new-ssd: 'true'
```
pod绑定了原节点Localpath卷，又指定了调度到其他节点，此时找不到符合条件的节点从而无法调度
- 如需迁移pod存在数据副本/同步机制，通过删除绑定的卷，使pod能够调度到新节点，再自动在新节点创建卷，数据会自动同步到新的pvc
    - 找到pod绑定的pvc删除
    - 重建pod
    - 等待创建新卷
    - 等待数据同步
- 如需迁移pod没有数据同步/副本机制，需要手动将数据复制到新的pvc
    - 找到pod绑定pvc
    - 到宿主机中pvc挂载位置将数据复制到临时目录
    - 删除pod、pvc（一并缩小集群数量，否则无法将卷删除）
    - 在目标节点手动创建同名pvc
    ```
    kind: PersistentVolumeClaim
    apiVersion: v1
    metadata:
      name: pvc-0
      annotations:
    volume.kubernetes.io/selected-node: server-dev-3
    spec:
      accessModes:
        - ReadWriteOnce
      resources:
        requests:
          storage: 50Gi
      storageClassName: local-path
    ```
    - 等待pvc创建完成后，将数据从临时目录负责到新pvc挂载宿主机目录
    - 放大集群数量，调度pod到新pvc所在节点






