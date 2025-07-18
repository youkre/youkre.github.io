---
title: "常用函数导数与积分标准形式表"
summary: ""
---

## 标准形式表

<table class="text-center table table-striped" rules="all">
<tbody><tr><th>$\dfrac{dy}{dx}$</th><th>$y$</th><th>$\int y\, dx$</th></tr>
<tr><td colspan="3"><b>代数</b></td></tr>

<tr><td>$1$ </td><td> $x$     </td><td> $\frac{1}{2} x^2 + C$ </td></tr>
<tr><td>$0$ </td><td> $a$     </td><td> $ax + C $             </td></tr>
<tr><td>$1$ </td><td> $x ± a$ </td><td> $\frac{1}{2} x^2 ± ax + C$ </td></tr>
<tr><td>$a$ </td><td> $ax $   </td><td> $\frac{1}{2} ax^2 + C $</td></tr>
<tr><td>$2x$ </td><td> $x^2$  </td><td> $\frac{1}{3} x^3 + C $ </td></tr>
<tr><td>$nx^{n-1}$ </td><td> $x^n$ </td><td>$ \dfrac{1}{n+1} x^{n+1} + C $</td></tr>
<tr><td>$-x^{-2} $ </td><td> $x^{-1}$ </td><td> $\log_\epsilon x + C$ </td></tr>
<tr><td>$\dfrac{du}{dx} ± \dfrac{dv}{dx} ± \dfrac{dw}{dx}$
   </td><td> $u ± v ± w$     </td><td> $\int u\, dx ± \int v\, dx ± \int w\, dx$ </td></tr>
<tr><td>$u\, \dfrac{dv}{dx} + v\, \dfrac{du}{dx}$
   </td><td> $uv$  </td><td> No general form known </td></tr>
<tr><td>$\dfrac{v\, \dfrac{du}{dx} - u\, \dfrac{dv}{dx}}{v^2}$
   </td><td> $\dfrac{u}{v}$ </td><td> No general form known </td></tr>
<tr><td>$\dfrac{du}{dx}$ </td><td> $u$ </td><td> $ux - \int x\, du + C$ </td></tr>

<tr><td colspan="3"><b>指数与对数函数</b></td></tr>
<tr><td>$\epsilon^x$ </td><td> $\epsilon^x$ </td><td> $\epsilon^x + C$
</td></tr><tr><td>$x^{-1}$     </td><td> $\log_\epsilon x$ </td><td> $ x(\log_\epsilon x - 1) + C$
</td></tr><tr><td>$0.4343 × x^{-1}$ </td><td> $\log_{10} x$ </td><td> $0.4343x (\log_\epsilon x - 1) + C$ </td></tr>
<tr><td>$a^x \log_\epsilon a$ </td><td> $a^x$ </td><td> $\dfrac{a^x}{\log_\epsilon a} + C$ </td></tr>

<tr><td colspan="3"><b>三角函数</b></td></tr>
<tr><td>$\cos x$  </td><td> $\sin x$ </td><td> $-\cos x + C $ </td></tr>
<tr><td>$-\sin x$ </td><td> $\cos x$ </td><td> $\sin x + C $ </td></tr>
<tr><td>$\sec^2 x$</td><td> $\tan x$ </td><td> $-\log_\epsilon \cos x + C $ </td></tr>

<tr><td colspan="3"><b>圆形（反函数）</b></td></tr>
<tr><td>$\dfrac{1}{\sqrt{(1-x^2)}}$ </td><td> $\arcsin x$ </td><td> $x · \arcsin x + \sqrt{1 - x^2} + C$ </td></tr>
<tr><td>$-\dfrac{1}{\sqrt{(1-x^2)}}$ </td><td> $\arccos x$ </td><td> $x · \arccos x - \sqrt{1 - x^2} + C$ </td></tr>
<tr><td>$\dfrac{1}{1+x^2}$ </td><td> $\arctan x$ </td><td> $x · \arctan x - \frac{1}{2} \log_\epsilon (1 + x^2) + C$ </td></tr>

<tr><td colspan="3"><b>双曲函数</b></td></tr>
<tr><td>$\cosh x   $ </td><td> $\sinh x$ </td><td> $\cosh x + C$ </td></tr>
<tr><td>$\sinh x   $ </td><td> $\cosh x$ </td><td> $\sinh x + C$ </td></tr>
<tr><td>$\text{sech}^2 x $ </td><td> $\tanh x$ </td><td> $\log_\epsilon \cosh x + C $ </td></tr>
<tr><td colspan="3"><b>其他</b></td></tr>

<tr><td>$-\dfrac{1}{(x + a)^2}$ </td><td> $\dfrac{1}{x + a}$ </td><td> $ \log_\epsilon (x+a) + C $ </td></tr>
<tr><td>$-\dfrac{x}{(a^2 + x^2)^{\frac{3}{2}}}$
  </td><td> $\dfrac{1}{\sqrt{a^2 + x^2}}$
  </td><td> $\log_\epsilon (x + \sqrt{a^2 + x^2}) + C $ </td></tr>
<tr><td>$\mp \dfrac{b}{(a ± bx)^2}$
  </td><td> $\dfrac{1}{a ± bx}$
  </td><td> $± \dfrac{1}{b} \log_\epsilon (a ± bx) + C $ </td></tr>
<tr><td>$-\dfrac{3a^2x}{(a^2 + x^2)^{\frac{5}{2}}}$
  </td><td> $\dfrac{a^2}{(a^2 + x^2)^{\frac{3}{2}}}$
  </td><td> $\dfrac{x}{\sqrt{a^2 + x^2}} + C $ </td></tr>
<tr><td>$ a · \cos ax$ </td><td> $\sin ax$ </td><td> $-\dfrac{1}{a} \cos ax + C $ </td></tr>
<tr><td>$-a · \sin ax$ </td><td> $\cos ax$ </td><td> $ \dfrac{1}{a} \sin ax + C $ </td></tr>
<tr><td>$ a · \sec^2ax$ </td><td> $\tan ax$ </td><td> $-\dfrac{1}{a} \log_\epsilon \cos ax + C $ </td></tr>
<tr><td>$ \sin 2x$ </td><td> $\sin^2 x$ </td><td> $\dfrac{x}{2} - \dfrac{\sin 2x}{4} + C $ </td></tr>
<tr><td>$-\sin 2x$ </td><td> $\cos^2 x$ </td><td> $\dfrac{x}{2} + \dfrac{\sin 2x}{4} + C $ </td></tr>
<tr><td>$n · \sin^{n-1} x · \cos x$
  </td><td> $ \sin^n x$
  </td><td> $-\frac{\cos x}{n} \sin^{n-1} x
     + \frac{n-1}{n} \int \sin^{n-2} x\, dx + C$ </td></tr>
<tr><td>$-\dfrac{\cos x}{\sin^2 x}$
  </td><td> $\dfrac{1}{\sin x}$
  </td><td> $\log_\epsilon \tan \dfrac{x}{2} + C$ </td></tr>
<tr><td>$-\dfrac{\sin 2x}{\sin^4 x}$
  </td><td> $\dfrac{1}{\sin^2 x}$
  </td><td> $ -\text{cotan} x + C$ </td></tr>
<tr><td>$\dfrac{\sin^2 x - \cos^2 x}{\sin^2 x · \cos^2 x}$
  </td><td> $ \dfrac{1}{\sin x · \cos x}$
  </td><td> $ \log_\epsilon \tan x + C $ </td></tr>
<tr><td>
  $n · \sin mx · \cos nx + m · \sin nx · \cos mx $
  </td><td> $\sin mx · \sin nx$
  </td><td> $\frac{1}{2} \cos(m - n)x - \frac{1}{2} \cos(m + n)x + C$ </td></tr>
<tr><td>$ 2a·\sin 2ax$ </td><td> $\sin^2 ax$ </td><td> $\dfrac{x}{2} - \dfrac{\sin 2ax}{4a} + C $ </td></tr>
<tr><td>$-2a·\sin 2ax$ </td><td> $\cos^2 ax$ </td><td> $\dfrac{x}{2} + \dfrac{\sin 2ax}{4a} + C $ </td></tr>
</tbody></table>

<nav class="pagination justify-content-between">
<a href="../epilogue">尾声与寓言</a>
<a href="../">目录</a>
<span></span>
</nav>

