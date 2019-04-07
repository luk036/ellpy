# New

$$x^\top P^{-1} x + \mu(l\cdot m^\top + m \cdot l^\top)$$

$$\begin{bmatrix} P^{-1} & 0 \\ 
 0 & -1 \end{bmatrix}  + \mu(\begin{bmatrix} g \\ c \end{bmatrix} \cdot \begin{bmatrix} g \\ d \end{bmatrix}^\top + \begin{bmatrix} g \\ d \end{bmatrix} \cdot \begin{bmatrix} g \\ c \end{bmatrix}^\top)$$

$$\begin{bmatrix} P^{-1} & 0 \\ 
 0 & -1 \end{bmatrix}  + \mu(\begin{bmatrix} g g^\top & d g \\ c g & cd \end{bmatrix} + \begin{bmatrix} g g^\top & c g \\ d g & cd \end{bmatrix})$$

$$\begin{bmatrix} P^{-1} & 0 \\ 
 0 & -1 \end{bmatrix}  + \mu \begin{bmatrix} 2 g g^\top & (c+d) g \\ (c+d) g & 2 cd \end{bmatrix} $$

 $$x^\top (P^{-1} + 2\mu \cdot g g^\top) x - 1 + 2\mu(c + d) g^\top x + 2\mu(c\cdot d)$$

 $$x^\top (P^{-1} + a \cdot g g^\top) x + a(c + d) g^\top x + a(c\cdot d) - 1 \leq 0$$

 $$\beta ((x - x_c)^\top P'^{-1} (x - x_c) - 1) \leq 0 $$

 $$\beta (x^\top P'^{-1} x - 2 x_c^\top P'^{-1} x + x_c^\top x_c - 1) \leq 0$$

 $$ x_c = {-a(c + d) \over 2 \beta} P' g$$ 

 $$ x_c^\top x_c - 1 =  \alpha( a(c\cdot d) - 1)$$

 $$P'^{-1} = \alpha(P^{-1} + a \cdot g g^\top) $$

 (Sherman-Morrison formula)
 $$(P^{-1} +  a \cdot g g^\top)^{-1} = P - {a \over \sigma} P g g^\top P $$

 where 

 $$\sigma = 1 + a g^\top P g = 1 + a b$$

 $$P' = \beta (P - {a \over \sigma} P g g^\top P)$$

 $$\red{P' = \beta (P - {a \over \sigma} h h^\top ) }$$


$$ x_c = {-a(c + d) \over 2 } (P - {a \over \sigma} P g g^\top P) g$$ 

$$ x_c = {-a(c + d) \over 2 } (h - {a b \over \sigma} h)$$ 

$$ \red{x_c = (-a(c + d) / 2) / (1 + ab) h }$$ 

$$ x_c^\top x_c = (a(c + d) /(1 + ab)/2)^2  h^\top h$$ 

$$\alpha = (({a (c + d) } /(1 + ab)/2)^2  h^\top h - 1)/ (a (c\cdot d)-1)$$ 

$$ \beta = (a (c\cdot d)-1) /(({a (c + d) } /(1 + ab)/2)^2  h^\top h - 1)$$ 

$$ P'^{-1} = (((a (c + d) -1) /(1 + ab))/2)^2  h^\top h - 1)/ (a (c\cdot d)) (P^{-1} + a \cdot g g^\top)$$ 

$$ P' = (a (c\cdot d) - 1) /(({a (c + d) } / (1 + ab) )/2)^2  h^\top h - 1) (P - {a / (1 + ab)} h h^\top ) $$ 

$$\frac{\partial f}{\partial a} = -(1/(1+a\cdot b)\cdot h\cdot h^\top -(a\cdot b)/(1+a\cdot b)^{2}\cdot h\cdot h^\top ) = 0 $$

$$a = \lambda$$
$$b = \tau$$

```
   log(det((a*(c*d)-1)/((a*(c+d)/(1+a*b)/2)^2 * h' * h - 1)*(P - (a/(1+a*b))*h*h')))
```

