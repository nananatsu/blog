## 使用goyacc实现sql语法解析

程序设计语言的分析分为三个阶段：
- 线性分析（词法分析），读取源程序为字节流，将字节流分组为多个记号（token，具有整体含义的字符序列）
- 层次分析（语法分析），将字符串或记号划分为具有一定层次的多个嵌套组，一颗分析树
- 语义分析，检查语义错误，收集类型信息

语言的语法一般会用上下文无关文法来表示，上下文无关文法，由终结符、非终结符、开始符号、产生式组成：
- 终结符：组成字符串的基本符号
- 非终结符：表示字符串集合的语法变量
- 开始符号：文法中有一个非终结符被指定为开始符号，开始符号表示的字符串集合就是要定义的语言
- 产生式：终结符和非终结符组合成串的方式，由非终结符开始，跟随一个箭头，后面是终结符非终结符组成的串

语法解析方法分为两类类：
- 自顶向下，按从根结点到叶节点的顺序构造分析树，常见分析法：LL(1)
- 自底向上，按从叶节点到根结点的顺序构造分析树，常见分析法：LR(1)、LALR(1)

yacc是基于LALR算法的语法分析器，在程序设计语言编译中被广泛使用，goyacc是yacc的go版本，我们使用[goyacc](https://pkg.go.dev/modernc.org/goyacc)来解析sql语句生成抽象语法书。

yacc语法文件由三个部分构成：
- 声明分为两节：
    - 第一节是目标语言声明，包含在分界符%{和}%之间，使用goyacc时中存放go的一些声明，如包名、import等
    ```bnf
    %{
        package sql

        import (
            "strconv"
            "strings"
        )

     %}
    ```
    - 第二节是文法记号，声明一些符号，用于后续翻译规则， 如%union、%token、%type等
        - %union 声明要使用的类型
        - %token 声明终结符
        - %type 声明非终结符
        - %left/%right 声明符号是左结合/右结合
    ```bnf
        %union {
            str string
        }

        %token <str> 
            CREATE "CREATE"
            TABLE "TABLE"

        %type <str> Expr
    ```
- 翻译规则：与第一部分通过%%符号隔开，每条规则由一个文法产生式和后续的语义动作构成：
    - 具有相同左部的产生式可以合并到一起，每个右部用|分隔
    - $$ 表示左边非终结符的值
    - $i 表示右边第i（起始为1）个文法符号（包含终结符、非终结符）的值
    - 语义动作一般为根据`$i`计算`$$`
    - 默认语义动作为 `{ $$ = $1 }`，可省略
```bnf

SelectStmt:
    "SELECT" SelectFieldList SelectStmtLimit ';'
    {
        $$ = &SelectStmt{ Filed: $2 , Limit: $3}
    }
    |  "SELECT" SelectFieldList SelectStmtFrom SelectStmtWhere SelectStmtOrder SelectStmtLimit ';'
    {
        $$ = &SelectStmt{ Filed: $2, From: $3 ,Where: $4, Order: $5, Limit: $6 }
    }
```
- 支持例程，提供yylex()词法分析器，在goyacc中是通过实现yyLexer接口来提供词法分析

依据mysql的[Select Statement](https://dev.mysql.com/doc/refman/8.0/en/select.html)编写相应文法，将select语句分为下述几个部分
```go
type SelectStmt struct {
	Filed []*SelectField
	From  *SelectStmtFrom
	Where []BoolExpr
	Order []*OrderField
	Limit *SelectStmtLimit
}
```
- 查询字段
```go
type SelectField struct {
    Field string
    Pos   int
    Expr  *SqlFunction
    Alias string
}
```
- 查询来源
```go
type SelectStmtFrom struct {
	Table string
}
```
- 查询条件，由多个布尔表达式组成合取范式(CNF)，转换时用Negation标记是否对真值取反，WhereExpr、WhereField实现BoolExpr接口
```go
type WhereExpr struct {
	Negation bool
	Cnf      []BoolExpr
}

type BoolExpr interface {
	Prepare(fieldMapping map[string]int) error
	Filter(row *[]string) bool
	Negate()
}

type WhereField struct {
	Field   string
	Pos     int
	Value   string
	Compare CompareOp
}
```
- 排序条件
```go
type OrderField struct {
	Field string
	Asc   bool
}
```
- 限制条件
```go
type SelectStmtLimit struct {
	Offset int
	Size   int
}
```
在yacc定义上述结构
```bnf
%union {
    str string
    strArr []string
    boolean bool
    
    stmt Statment
    stmtList []Statment

    selectStmt *SelectStmt
    sqlFunc *SqlFunction
    selectFieldList []*SelectField
    selectStmtFrom *SelectStmtFrom
    whereExprList []BoolExpr
    orderFieldList []*OrderField
    selectStmtLimit *SelectStmtLimit
    compareOp CompareOp
}
```
定义selec语句关键字
```bnf
%token <str> 
    SELECT "SELECT" 
    FROM "FROM" 
    WHERE "WHERE" 
    ORDER "ORDER"
    BY "BY" 
    LIMIT "LIMIT" 
    OFFSET "OFFSET" 
    ASC "ASC" 
    DESC "DESC" 
    AND "AND" 
    OR "OR"

%token <str> 
    COMP_NE "!="
    COMP_LE "<=" 
    COMP_GE ">=" 

%token <str> VARIABLE
%type <str> Expr
%type <strArr> VaribleList
%type <stmt> Stmt
%type <stmtList> StmtList
%type <selectStmt> SelectStmt
%type <sqlFunc> AggregateFunction
%type <selectFieldList> SelectFieldList   
%type <selectStmtFrom> SelectStmtFrom
%type <whereExprList> SelectStmtWhere WhereExprList
%type <orderFieldList> SelectStmtOrder OrderFieldList
%type <selectStmtLimit> SelectStmtLimit 
%type <compareOp> CompareOp
%type <boolean> Ascend

%left OR
%left AND
%left '+' '-'
%left '*' '/'
```
定义开始符号
```bnf
%start start
```
编写语法规则，允许多条语句，语义动作 保存Stmt到StmtList
```bnf
start:
    StmtList

StmtList:
    Stmt
    {
        $$ = append($$, $1)
    }
    | StmtList Stmt
    {
        $$ = append($$, $2)
    }

Stmt:
    SelectStmt
    {
        $$ = Statment($1)
    }
```

select语句必须具有结构仅有"select" expr，其他关键字都是可选，无"from"子句时，后续只允许"limit"子句，将select分为两个候选产生式，语义动作都是将子句的结构保存到SelectStmt结构中
- 候选1形如 select expr [limit expr]
- 候选2形如 select expr from table [ where expr] [ order by expr ] [ limit expr ]
```bnf
SelectStmt:
    "SELECT" SelectFieldList SelectStmtLimit ';'
    {
        $$ = &SelectStmt{ Filed: $2 , Limit: $3}
    }
    |  "SELECT" SelectFieldList SelectStmtFrom SelectStmtWhere SelectStmtOrder SelectStmtLimit ';'
    {
        $$ = &SelectStmt{ Filed: $2, From: $3 ,Where: $4, Order: $5, Limit: $6 }
    }
```
select字段可能是表的字段，或是一个函数，yacc使用LALR(1)每次只读入一个记号，新字段读入时要区分是否已读取过字段，这样讲select查询字段分为了4个候选，当字段为第一个时在语义动作中创建新的查询字段切片，将第一个字段写入，当不是第一个字段将新字段写入已存在的切片
```bnf
SelectFieldList:
    Expr
    {
        $$ = []*SelectField{ &SelectField{ Field: $1 } }
    }
    | SelectFieldList ',' Expr
    {
        $$ = append( $1, &SelectField{ Field: $3 } )
    }
    | AggregateFunction
    {
        $$ = []*SelectField{ &SelectField{ Expr: $1 }}
    }
    | SelectFieldList ',' AggregateFunction
    {
        $$ = append( $1, &SelectField{ Expr: $3 } )
    }
```
select from后面需要时一个表，允许实际表、子查询、join后的表，这里仅定义实际表
```bnf
SelectStmtFrom:
    "FROM" Expr
    {
        $$ = &SelectStmtFrom{
            Table: $2,
        }
    }
```
select where关键字可选，将其分为两个候选
```bnf
SelectStmtWhere:
    {
        $$ = nil
    }
    | "WHERE" WhereExprList
    {
        $$ = $2
    }
```
将where子句由多个布尔表达式构成，为方便计算真值，需要将其转换为合取范式(CNF)
- 当为第一个条件，建立切片，将条件构造为BoolExpr
- 当新条件与之前已存在BoolExpr关系为 or 时，标记对BoolExpr真值取反，将条件添加到BoolExpr，构造一个新BoolExpr包含旧BoolExpr，标记真值取反
- 当新条件与之前BoolExpr关系为 and 时，将条件添加到BoolExpr
- 当新BoolExpr与与之前BoolExpr关系为 or 时，标记新旧BoolExpr取反，合并新旧BoolExpr，标记合并后BoolExpr真值取反
- 当新BoolExpr与与之前BoolExpr关系为 and 时，将新子句添加到旧子句
```bnf
WhereExprList:
    Expr CompareOp Expr
    {
        $$ = []BoolExpr{ &WhereField{Field:$1, Value:$3, Compare:$2} }
    }
    | WhereExprList OR Expr CompareOp Expr  %prec OR
    {
        $4.Negate()
        filed := &WhereField{ Field:$3, Value:$5, Compare:$4 }
        if len($$) == 1{
            $$[0].Negate()
            $$ = append($$, filed)
            $$ = []BoolExpr{ &WhereExpr{ Negation: true, Cnf: $$ }  }
        }else{
            $$ = []BoolExpr{ &WhereExpr{ Negation: true, Cnf: []BoolExpr{ &WhereExpr{ Negation: true, Cnf: $$ } , filed} } }
        }
    }
    | WhereExprList AND Expr CompareOp Expr %prec AND
    {
        $$ = append($$, &WhereField{ Field:$3, Value:$5, Compare:$4} )
    }
    | WhereExprList OR '(' WhereExprList ')' %prec OR
    {
        if len($$) == 1{
            $$[0].Negate()
            $$ = append($$, &WhereExpr{ Negation: true, Cnf: $4 })
            $$ = []BoolExpr{ &WhereExpr{ Negation: true, Cnf: $$ }  }
        }else{
           $$ =[]BoolExpr{ &WhereExpr{ Negation: true, Cnf: []BoolExpr{ &WhereExpr{ Negation: true, Cnf: $$ } , &WhereExpr{ Negation: true, Cnf: $4 } } } }
        }
    }
    | WhereExprList AND '(' WhereExprList ')' %prec AND
    {
        $$ = append($$, $4... )
    }
```
selec order关键字可选，分为两个候选，子句由多个order字段构成，分两个候选
```bnf
SelectStmtOrder:
    {
        $$ = nil
    }
    | "ORDER" "BY" OrderFieldList 
    {
        $$ = $3
    }

OrderFieldList:
    Expr Ascend
    {
        $$ = []*OrderField{&OrderField{ Field: $1, Asc: $2 }}
    }
    | OrderFieldList ',' Expr Ascend
    {
        $$ = append($1, &OrderField{ Field: $3, Asc: $4 })
    }
```
select limit子句支持三种语法对应yacc三种候选，语义动作中将字符串转换数字
```bnf
SelectStmtLimit:
    {
        $$ = nil
    }
    | "LIMIT" VARIABLE
    {
        size,err :=  strconv.Atoi($2)
        if err != nil{
            yylex.Error(err.Error())
            goto ret1
        }
        $$ = &SelectStmtLimit{Size: size }
    }
    | "LIMIT" VARIABLE ',' VARIABLE
    {   

        offset,err :=  strconv.Atoi($2)
        if err != nil{
            yylex.Error(err.Error())
            goto ret1
        }
        size,err :=  strconv.Atoi($4)
        if err != nil{
            yylex.Error(err.Error())
            goto ret1
        }
        $$ = &SelectStmtLimit{Offset: offset, Size: size }
    }
    | "LIMIT" VARIABLE "OFFSET" VARIABLE
    {
        offset,err :=  strconv.Atoi($4)
        if err != nil{
            yylex.Error(err.Error())
            goto ret1
        }
        size,err :=  strconv.Atoi($2)
        if err != nil{
            yylex.Error(err.Error())
            goto ret1
        }
        $$ = &SelectStmtLimit{Offset: offset, Size: size }
    }
```
要利用yyParser需要实现yyLexer或yyLexerEx接口
```go
type yyLexer interface {
	Lex(lval *yySymType) int
	Error(e string)
}

type yyLexerEx interface {
	yyLexer
	Reduced(rule, state int, lval *yySymType) (stop bool) 
}
```
定义一个简单的词法分析器,实现yyLexer，解析sql语句将token语法解析器
```go
type Lexer struct {
	sql    string
    stmts  []Statement
	offset int
	errs   []string
}
```
实现Lex方法，逐个读取sql中的字符，分割语句为token，将token写入lval，返回token对应符号表位置。
```go
func (l *Lexer) Lex(lval *yySymType) int {
	start := l.offset
	end := len(l.sql)
	if l.offset >= end {
		return 0
	}

	prevQuote := false
	prevBacktick := false
	prevSingleQuotes := false
	prevDoubleQuotes := false
	for i := l.offset; i < len(l.sql); i++ {
		switch l.sql[i] {
		case '\\':
			continue
		case '\'':
			prevSingleQuotes = !prevSingleQuotes
			prevQuote = prevBacktick || prevSingleQuotes || prevDoubleQuotes
		case '"':
			prevDoubleQuotes = !prevDoubleQuotes
			prevQuote = prevBacktick || prevSingleQuotes || prevDoubleQuotes
		case '`':
			prevBacktick = !prevBacktick
			prevQuote = prevBacktick || prevSingleQuotes || prevDoubleQuotes
		}

		if !prevQuote {
			switch l.sql[i] {
			case ' ', '\n', '\t':
				end = i
			case '<', '>', '!':
				end = i
				if start == end {
					if i+1 < len(l.sql) && l.sql[i+1] == '=' {
						end += 2
					} else {
						end++
					}
				}
			case '=', ',', ';', '(', ')':
				end = i
				if start == end {
					end++
				}
			}
		}

		if i == end {
			break
		}
	}

	l.offset = end
	for l.offset < len(l.sql) {
		char := l.sql[l.offset]
		if char == ' ' || char == '\n' || char == '\t' {
			l.offset++
		} else {
			break
		}
	}

	token := l.sql[start:end]
	lval.str = token

	num, ok := SqlTokenMapping[token]
	if ok {
		return num
	} else {
		return VARIABLE
	}
}
```
实现Reduced()方法，记录归约结果
```
func (l *Lexer) Reduced(rule, state int, lval *yySymType) bool {
	if state == 2 {
		l.stmts = lval.stmtList
	}
	return false
}
```
实现Error()方法，记录语法解析中的错误
```
func (l *Lexer) Error(s string) {
	l.errs = append(l.errs, s)
}
```
使用goyacc用sql.y生成go语言的sql解析器
```go
goyacc -o yyParser.go -v yacc.output sql.y
```
调用yyParse()函数，传入一个Lexer实例，完成语法解析。
```go
lex := &Lexer{sql: sql}
n := yyParse(lex)
```
```
[]kvdb/pkg/sql.Statment len: 1, cap: 1, 
[*kvdb/pkg/sql.SelectStmt {
    Filed: []*kvdb/pkg/sql.SelectField len: 1, cap: 1, [*(*"kvdb/pkg/sql.SelectField")(0xc0000145d0)], 
    From: *(*"kvdb/pkg/sql.SelectStmtFrom")(0xc00005e5a0), 
    Where: []kvdb/pkg/sql.BoolExpr len: 2, cap: 2, [...,...], 
    Order: []*kvdb/pkg/sql.OrderField len: 0, cap: 0, nil, 
    Limit: *(*"kvdb/pkg/sql.SelectStmtLimit")(0xc000020320)}]
```

[完整代码](https://github.com/nananatsu/simple-raft/tree/master/pkg/sql)

参考：
- https://cn.pingcap.com/blog/tidb-source-code-reading-5
- <<编译原理>>