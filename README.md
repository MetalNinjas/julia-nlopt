julia-nlopt
===========

NLopt bindings for julia. See [here](http://ab-initio.mit.edu/wiki/index.php/Main_Page) for the documentation of nlopt.

Usage
-----

Example using a L-BFGS optimizer. The objective function returns a tuple where the first value contains the function evaluation and the second the gradient evaluation.

```julia
x = [-1.2,1,-1.2,1]

function obj(x)
  fx = 0
  g = Array(Float64,length(x))
  for i = 1:2:length(x)
    f1 = 1-x[i]
    f2 = 10* (x[i+1] - x[i]^2)
    g[i+1] = 20.0 * f2
    g[i] = -2.0 * (x[i] * g[i+1] + f1)
    fx += f1^2 + f2^2
  end
  return (fx,g)
end

nlopt_optimize(NLopt(:ld_lbfgs,
                     :ftol_abs, 1e-8,
                     :maxeval, 2000,
                     :min_objective, obj), x)
```
