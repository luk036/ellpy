# New

$$x^T P^{-1} x + \mu(l\cdot m^T + m \cdot l^T)$$

$$\begin{bmatrix} P^{-1} & 0 \\ 
 0 & -1 \end{bmatrix}  + \mu(\begin{bmatrix} g \\ c \end{bmatrix} \cdot \begin{bmatrix} g \\ d \end{bmatrix}^T + \begin{bmatrix} g \\ d \end{bmatrix} \cdot \begin{bmatrix} g \\ c \end{bmatrix}^T)$$

$$\begin{bmatrix} P^{-1} & 0 \\ 
 0 & -1 \end{bmatrix}  + \mu(\begin{bmatrix} g g^T & d g \\ c g & cd \end{bmatrix} + \begin{bmatrix} g g^T & c g \\ d g & cd \end{bmatrix})$$

$$\begin{bmatrix} P^{-1} & 0 \\ 
 0 & -1 \end{bmatrix}  + \mu \begin{bmatrix} 2 g g^T & (c+d) g \\ (c+d) g & 2 cd \end{bmatrix} $$

 $$x^T (P^{-1} + 2\mu \cdot g g^T) x + 2\mu(c + d) g^T x + 2\mu(c\cdot d)$$

 $$x^T (P^{-1} + a \cdot g g^T) x + a(c + d) g^T x + a(c\cdot d) \leq 0$$

 $$\beta ((x - x_c)^T P'^{-1} (x - x_c) - 1) \leq 0 $$

 $$\beta (x^T P'^{-1} x - 2 x_c^T P'^{-1} x + x_c^T x_c - 1) \leq 0$$

 $$ x_c = {a(c + d) \over 2 \beta} P' g$$ 

 $$ x_c^T x_c - 1 = {a \over \beta} (c\cdot d)$$

 $$P'^{-1} = \alpha(P^{-1} + a \cdot g g^T) $$

 (Sherman-Morrison formula)
 $$(P^{-1} +  a \cdot g g^T)^{-1} = P - {a \over \sigma} P g g^T P $$

 where 

 $$\sigma = 1 + a g^T P g = 1 + a b$$

 $$P' = \beta (P - {a \over \sigma} P g g^T P)$$

 $$\red{P' = \beta (P - {a \over \sigma} h h^T ) }$$


$$ x_c = {a(c + d) \over 2 } (P - {a \over \sigma} P g g^T P) g$$ 

$$ x_c = {a(c + d) \over 2 } (h - {a b \over \sigma} h)$$ 

$$ \red{x_c = (a(c + d) / 2) / (1 + ab) h }$$ 

$$ x_c^T x_c = (a(c + d) /(1 + ab)/2)^2  h^T h$$ 

$$\alpha = (({a (c + d) } /(1 + ab)/2)^2  h^T h - 1)/ (a (c\cdot d))$$ 

$$ \beta = (a (c\cdot d)) /(({a (c + d) } /(1 + ab)/2)^2  h^T h - 1)$$ 

$$ P'^{-1} = (({a (c + d) } /(1 + ab))/2)^2  h^T h - 1)/ (a (c\cdot d)) (P^{-1} + a \cdot g g^T)$$ 

$$ P' = (a (c\cdot d)) /(({a (c + d) } / (1 + ab) )/2)^2  h^T h - 1) (P - {a / (1 + ab)} h h^T ) $$ 

$$a = \lambda$$
$$b = \tau$$