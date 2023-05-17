## istio中自定义过滤器

istio中使用EnvoyFilter来自定义过滤器，istio中的边车容器istio-proxy是一个envoy实例，istio使用xDS协议更新Envoy配置，在Envoy中添加过滤器。

Envoy中一次Http请求流程：-> tcp listener -> listener filter -> network filters -> HTTP filters -> router filter -> cluster manager
- listener：绑定IP/端口，接受TCP连接/UDP数据报，
    - filter_chains：listener中存在多个filter_chain，通过FilterChainMatch选择特定filter_chain
    - filter_chain：由一个或多个network filter（L3/L4）构成
    - listener filter: 接受socket后，network filter前处理
    - network filter：进行连接处理，分为三种：
        - read从下游接收数据时调用
        - write向下游发送数据时调用
        - read/write从下游接收数据及向下游发送数据时都会调用
    - HTTP connection manager：链中最后一个network filter，进行Http请求的通用处理
    - HTTP filters：可以是下游过滤器在路由前处理下游请求，或是上游过滤器与指定集群关联，在路由后处理上游请求，分为三种：
        - Decoder在解码请求流时调用
        - Encoder在即将编码响应流时调用
        - Decoder/Encoder在解码请求流及即将编码响应流时都会调用
    - router filter：位于HTTP filters最后，选择一个集群转发请求
- cluster manager：管理上游集群信息，允许过滤器获取上游集群连接(L3/L4)

我们可以在请求的各个阶段加入新的过滤器，以外部授权为例，envoy的http filter中有一个外部授权过滤器（envoy.filters.http.ext_authz），通过调用外部GRPC或HTTP服务来检查请求，下面是一个外部grpc授权配置示例：

```yaml
http_filters:
  - name: envoy.filters.http.ext_authz
    typed_config:
      "@type": type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthz
      grpc_service:
        envoy_grpc:
          cluster_name: ext-authz
        timeout: 0.5s
      include_peer_certificate: true
```

要使用这个过滤器，我们需要实现Check(CheckRequest)，可以自己实现也可以选择已实现的插件，这里我们选择OPA-Envoy，这是一个实现了envoy外部授权api各grpc服务器，使用Rego语言定义验证授权策略。授权流程：外部授权过滤器通过grpc调用 opa-envoy服务器的Check()函数，Check()函数通过网络调用实际授权服务。

```protobuf
service Authorization {
  // Performs authorization check based on the attributes associated with the
  // incoming request, and returns status `OK` or not `OK`.
  rpc Check(CheckRequest) returns (CheckResponse) {
  }
}

message CheckRequest {
  option (udpa.annotations.versioning).previous_message_type = "envoy.service.auth.v2.CheckRequest";

  // The request attributes.
  AttributeContext attributes = 1;
}
```

首先在istio中配置EnvoyFilter，在istio的网关的http router过滤器前添加外部授权过滤器，使用grpc访问127.0.0.1:9191进行授权
- applyTo：指定配置应用到Envoy的哪个位置，主要可应用目标是Envoy请求流程中的各个过滤器
- patch：定义要应用的配置
  - pack_as_bytes: true，默认情况下会以UTF-8字符串传输http请求体，如请求题含字节流，需设置pack_as_bytes为true

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: ext-authz
  namespace: istio-system
spec:
  configPatches:
    - applyTo: HTTP_FILTER
      match:
        context: GATEWAY
        listener:
          filterChain:
            filter:
              name: "envoy.filters.network.http_connection_manager"
              subFilter:
                name: "envoy.filters.http.router"
      patch:
        operation: INSERT_BEFORE
        value:
          name: envoy.ext_authz
          typed_config:
            '@type': type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthz
            transport_api_version: V3
            status_on_error:
              code: ServiceUnavailable
            with_request_body:
              max_request_bytes: 8192
              allow_partial_message: true
              pack_as_bytes: true
            grpc_service:
              google_grpc:
                target_uri: 127.0.0.1:9191
                stat_prefix: "ext_authz"
```

编写授权策略policy.rego，解析token，访问实际授权服务器

```rego
package istio.authz

import future.keywords

import input.attributes.request.http as http_request

default allow := false

path := http_request.path

# OPTIONS请求放行
allow := {"allowed": true} if {
  http_request.method == "OPTIONS"
} else := {"allowed": true, "headers": get_authn_header("authn")} if { # 允许匿名访问地址
  check_open_access_by_prefix("open_access")
} else := r if { # 检查jwt token
  r = check_token(get_jwks("jwks"))
}

# 从header获取Authorization
get_authn_header(_) := {"Authorization": http_request.headers.authn} if {
  check_authn_auth_path(path)
} else := {}

# 检查放行前缀
check_open_access_by_prefix(path) if {
  prefixes := [
    "/auth",
    "/oauth",
    "/auth/logout",
    "/health",
    "/api/socket",
    "/login",
    "/logout",
    "/index",
    "/public"
  ]
  some prefix in prefixes
  startswith(path, prefix)
}

# 检查jwt token
check_token(jwks) := ret if {
  token := bearer_token(http_request.headers.authorization)
  # 验证token
  io.jwt.verify_rs256(token, jwks)

  # 请求授权
  resp := http.send({
    "method": "get",
    "url": sprintf("%v/oauth/token/check?token=%v", [get_auth_server(""), token]),
  })

  ret := check_response(resp)

} else := object.union({"body": json.marshal({"code": 10001, "message": "token verify fail"})}, get_deny_response("deny"))

# 检查验证结果
check_response(resp) := {"allowed": true, "headers": {"userId": to_string(resp.body.data.userId)}} if {
  resp.body.code == 200
} else := object.union({"body": json.marshal(resp.body)}, get_deny_response("deny"))

to_string(v) := "" if {
  v == null
} else := v

# 分割bearer token
bearer_token(v) := t if {
  startswith(v, "bearer ")
  t := substring(v, count("bearer "), -1)
} else := v

# 获取授权服务地址
get_auth_server(_) := "http://authz.test.svc.cluster.local"

# 获取jwks
get_jwks(_) := jwks if {
  jwks := `{"kty":"RSA","e":"AQAB","n":"xxxxxxxxxx"}`
}

# 生成拒绝响应
get_deny_response(_) := deny if {
  deny := {"allowed": false, "http_status": 200, "headers": {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json;charset=UTF-8"}}
}
```

添加opa-envoy配置，通过istio/authz/allow路径取得授权结果

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: opa-istio-config
  namespace: istio-system
data:
  config.yaml: |
    plugins:
      envoy_ext_authz_grpc:
        addr: :9191
        path: istio/authz/allow
    decision_logs:
      console: true
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: opa-policy
  namespace: istio-system
data:
  policy.rego: |
    ...

```

给网关添加一个opa-envoy边车容器，从kubernetes ComfigMap中读取授权策略

```yaml
- name: opa-istio
  image: 'openpolicyagent/opa:latest-istio'
  args:
    - run
    - '--server'
    - '--config-file=/config/config.yaml'
    - '--addr=localhost:8181'
    - '--diagnostic-addr=0.0.0.0:8282'
    - /policy/policy.rego
  resources: {}
  volumeMounts:
    - name: opa-istio-config
      mountPath: /config
    - name: opa-policy
      mountPath: /policy
    - name: kube-api-access
      readOnly: true
      mountPath: /var/run/secrets/kubernetes.io/serviceaccount
  livenessProbe:
    httpGet:
      path: /health?plugins
      port: 8282
      scheme: HTTP
    timeoutSeconds: 1
    periodSeconds: 10
    successThreshold: 1
    failureThreshold: 3
  readinessProbe:
    httpGet:
      path: /health?plugins
      port: 8282
      scheme: HTTP
    timeoutSeconds: 1
    periodSeconds: 10
    successThreshold: 1
    failureThreshold: 3
```

运行测试，结果如下，成功调用ext_authz filter进行了授权

```log
debug envoy conn_handler	[C218204] new connection from 192.168.192.48:28380
debug envoy http	[C218204] new stream
debug envoy http	[C218204][S8831040841744865726] request headers complete (end_stream=false):
debug envoy connection	[C218204] current connecting state: false
debug envoy filter	override with 3 ALPNs
debug envoy filter	[C218204][S8831040841744865726] ext_authz filter is buffering the request
debug envoy http	[C218204][S8831040841744865726] request end stream
debug envoy filter	[C218204][S8831040841744865726] ext_authz filter finished buffering the request since stream is ended
debug envoy grpc	Finish with grpc-status code 0
debug envoy grpc	notifyRemoteClose 0 
debug envoy router	[C218204][S8831040841744865726] cluster 'outbound|80||authz.test.svc.cluster.local' match for URL '/test/authz'
debug envoy router	[C218204][S8831040841744865726] router decoding headers:
debug envoy pool	[C215020] using existing connection
debug envoy pool	[C215020] creating stream
debug envoy router	[C218204][S8831040841744865726] pool ready
debug envoy grpc	Stream cleanup with 0 in-flight tags
debug envoy grpc	Deferred delete
debug envoy grpc	GoogleAsyncStreamImpl destruct
debug envoy router	[C218204][S8831040841744865726] upstream headers complete: end_stream=false
debug envoy http	[C218204][S8831040841744865726] closing connection due to connection close header
debug envoy http	[C218204][S8831040841744865726] encoding headers via codec (end_stream=false):
debug envoy client	[C215020] response complete
debug envoy wasm	wasm log stats_outbound stats_outbound: [extensions/stats/plugin.cc:645]::report() metricKey cache hit , stat=16
debug envoy wasm	wasm log stats_outbound stats_outbound: [extensions/stats/plugin.cc:645]::report() metricKey cache hit , stat=18
debug envoy wasm	wasm log stats_outbound stats_outbound: [extensions/stats/plugin.cc:645]::report() metricKey cache hit , stat=22
debug envoy wasm	wasm log stats_outbound stats_outbound: [extensions/stats/plugin.cc:645]::report() metricKey cache hit , stat=26
debug envoy connection	[C218204] closing data_to_write=16590 type=0
debug envoy connection	[C218204] setting delayed close timer with timeout 1000 ms
debug envoy pool	[C215020] response complete
debug envoy pool	[C215020] destroying stream: 0 remaining
```
```json
{
    "decision_id": "74c6c9d7-369b-4a78-a5f9-eaefad5fa5fe",
    "labels": {
        "id": "d9688517-cb82-4dc8-b728-9ccb7e406dc7",
        "version": "0.49.2-envoy"
    },
    "level": "info",
    "metrics": {
        "timer_rego_builtin_http_send_ns": 5032217,
        "timer_rego_query_eval_ns": 5628333,
        "timer_server_handler_ns": 5820984
    },
    "msg": "Decision Log",
    "path": "istio/authz/allow",
    "result": {
        "allowed": true,
        "headers": {
            "userId": "111111111111111"
        }
    },
    "type": "openpolicyagent.org/decision_logs"
}
```


