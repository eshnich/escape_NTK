import jax
from jax import numpy as jnp, vmap
from jax import lax
from jax.experimental.host_callback import id_tap
from tqdm.auto import tqdm, trange


def tree_stack(trees):
    _, treedef = jax.tree_flatten(trees[0])
    leaf_list = [jax.tree_flatten(tree)[0] for tree in trees]
    leaf_stacked = [jnp.stack(leaves) for leaves in zip(*leaf_list)]
    return jax.tree_unflatten(treedef, leaf_stacked)


# def zeros_like_output(f, x):
#     pytree = jax.eval_shape(f, x)
#     return jax.tree_map(
#         lambda leaf: jnp.zeros(shape=leaf.shape, dtype=leaf.dtype), pytree
#     )


def batch_split(batch, n_batch: int = None, batch_size: int = None, strict=True):
    n = len(jax.tree_leaves(batch)[0])

    if n_batch is not None and batch_size is not None:
        raise Exception("Cannot specify both n_batch and batch_size")
    elif n_batch is not None:
        batch_size = n // n_batch
    elif batch_size is not None:
        n_batch = n // batch_size
    else:
        raise Exception("Need to specify either n_batch or batch_size")

    if strict:
        assert n_batch * batch_size == n
    else:
        batch = jax.tree_map(lambda x: x[: n_batch * batch_size], batch)
    batches = jax.tree_map(
        lambda x: x.reshape((n_batch, batch_size, *x.shape[1:])), batch
    )
    return batches


def _fold_clean(f):
    def clean_f(state, batch):
        fout = f(state, batch)
        if "state" not in fout:
            fout["state"] = None
        if "save" not in fout:
            fout["save"] = None
        if "avg" not in fout:
            fout["avg"] = None
        return fout

    return clean_f


def fold(
    f, state, data=None, steps=None, backend="lax", jit=False, show_progress=False
):
    if data is not None:
        n = len(jax.tree_leaves(data)[0])
        first_batch = jax.tree_map(lambda x: x[0], data)
        _f = f
    else:
        n = steps
        _f = lambda state, _: f(state)
        first_batch = None
    _f = _fold_clean(_f)
    if jit:
        _f = jax.jit(_f)

    if backend == "lax":
        output_tree = jax.eval_shape(lambda args: _f(*args), (state, first_batch))
        avg_init = jax.tree_map(
            lambda leaf: jnp.zeros(shape=leaf.shape, dtype=leaf.dtype),
            output_tree["avg"],
        )

        if show_progress:
            pbar = tqdm(total=n)

        def step(carry, batch):
            state, avg = carry
            fout = _f(state, batch)
            batch_state = fout["state"]
            batch_save = fout["save"]
            avg = jax.tree_multimap(lambda si, fi: si + fi / n, avg, fout["avg"])
            if show_progress:
                id_tap(lambda *_: pbar.update(), None)
            return (batch_state, avg), batch_save

        (state, avg), save = lax.scan(step, (state, avg_init), xs=data, length=steps)
        if show_progress:
            pbar.close()
        return dict(state=state, save=save, avg=avg)
    elif backend == "python":
        iterator = trange(n) if show_progress else range(n)
        avg = None
        save = []
        for i in iterator:
            if data is not None:
                batch = jax.tree_map(lambda x: x[i], data)
            else:
                batch = None
            fout = _f(state, batch)
            state = fout["state"]
            save.append(fout["save"])
            if avg is None:
                avg = jax.tree_map(lambda si: si / n, fout["avg"])
            else:
                avg = jax.tree_multimap(lambda si, fi: si + fi / n, avg, fout["avg"])
        save = tree_stack(save)
        return dict(state=state, save=save, avg=avg)


def laxmap(f, data, batch_size=None, **kwargs):
    if batch_size == None:
        return fold(lambda _, x: dict(save=f(x)), None, data, **kwargs)["save"]
    else:
        batches = batch_split(data, batch_size=batch_size)
        batched_out = fold(
            lambda _, batch: dict(save=vmap(f)(batch)), None, batches, **kwargs
        )["save"]
        flat_out = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), batched_out)
        return flat_out


# def laxmean(f, data, batch_size=None, unpack=True, **kwargs):
#     _f = (lambda x: f(*x)) if unpack else f
#     if batch_size == None:
#         return fold(lambda _, x: dict(avg=_f(x)), None, data, **kwargs)["avg"]
#     else:

#         def batched_f(batch):
#             out_tree = vmap(_f)(batch)
#             reduced_tree = jax.tree_map(lambda x: x.mean(0), out_tree)
#             return dict(avg=reduced_tree)

#         batches = batch_split(data, batch_size=batch_size)
#         return fold(lambda _, batch: batched_f(batch), None, batches, **kwargs)["avg"]
