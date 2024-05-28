# 这里记录下parser需要的一些功能

## parser 抽象类
### 对于不同的软件接口，读取的方式可以不同，但是最大程度的可以约束一个读取完之后的基本格式，比如数据类型，数据的shape
### 有一个基本格式之后，对于输出的形式，由于内部存储格式相同，输出也可以提前固定，比如我们可以指定几种输出的存储形式，写好方法，后续对接其他软件只需要写好获取这些物理量的方法，输出格式不用管，就可以直接调抽象类。
### 需要实现的物理量：
1. 原子结构信息，dpdata直接拿
2. 场 - p(r) V_h(r) V_xc(r) (x,y,z) - f(x,y,z)
    - p(r) - LCAO (C_i)
3. O(r,r') - H(r,r'), P(r,r), S(r,r') (i,j,R) -> []
    - Gaussian
    - VASP PW, <f|H|f>
4. Wave Function
5. kpoint eigenvalue

### 对接软件
1. ABACUS
2. Wannier
3. Gaussian
4. PYSCF
5. 