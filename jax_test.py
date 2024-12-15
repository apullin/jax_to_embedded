## Simple example to jax jit a math function, here the quadratic formula for roots

import jax
import jax.numpy as jnp

# Before run, set the env var:
# XLA_FLAGS=--xla_dump_to=./xla_dump --xla_dump_hlo_as_text --xla_dump_hlo_as_dot=true

# Function that we will JIT (quadratic roots)
def quad_roots(a : float ,b : float ,c : float  ):
    return ( (-b + jnp.sqrt( b**2 - 4*a*c ))/(2*a), (-b - jnp.sqrt( b**2 - 4*a*c ))/(2*a) )

# JIT-compile the function
jax_quad_roots = jax.jit(quad_roots)

# Run the compiled function to force compilation
result = jax_quad_roots( 2.0, 3.0, 4.0 )