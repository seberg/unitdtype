# Experimental Unit DType for NumPy

(Please expect that this readme may get outdated, open an issue if it does.)

This currently requires my branch on NumPy (As of 2022-01-26):

    https://github.com/seberg/numpy/tree/reduce-identity-array-meth

It supports more than my previous prototype and is now in C.

Note that scalars are not really supported, but they will be created.
They are currently necessary for printing, and you can create a new
array using them.
They will also work together _with_ arrays, since they will be
coerced to 0-D array (and that works).

Currently, casts within Units and to and from double (if dimensionless!)
as well as to boolean (unit is ignored, so offset is also) is supported.
The following ufuncs are supported:
* negative, positive
* add, subtract
* multiple, divide
* matmul
* fmod
* hypot  (not sure if this is quite right)
* maximum, minimum, fmax, fmin
* comparisons  (note that we never compare to 0 scalars, unlike some implementations).
* Logical ufuncs  (via casting to boolean)

And the following related functions:
* sum, mean, ...

More are likely to be added and I may not updated the README.

Many functions in NumPy should just work, however, e.g. sorting is not
implemented.  One caveat is that `np.prod` is _not_ implemented as this
requires some new API to be designed to make work (similar for theoretical
generalized ufuncs if they need the number of elements to be worked
on to figure out the result dtype).

Example?
```python3
import numpy as np; import unitdtype as udt

arr = np.arange(1000.).view(udt.Float64Unit("m"))

new = arr * arr
new_in_cm = new.astype(udt.Float64Unit("cm**2"))

new == new_in_cm  # luckily no rounding :).
```
