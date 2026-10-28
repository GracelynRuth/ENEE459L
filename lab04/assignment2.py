from __future__ import annotations

from typing import Any
from complexity import (
    FLOPS_PER_MAC,
    BUFFER_DTYPE,
    BN_BUFFERS_PER_CHANNEL
)
from graph import (
    Graph,
    Layer,
    computed,
    dtype_bytes,
)

def _layer_parameters(ly) -> int:
    if ly.kind == "conv":
        kernel_h, kernel_w = ly.kernel
        weights = (ly.out_shape[0] * (ly.in_shape[0] // ly.groups) * kernel_h * kernel_w)

        bias = ly.out_shape[0] if ly.bias else 0

        return weights + bias
    elif ly.kind == "linear":
        bias = ly.out_elements if ly.bias else 0
        return ly.in_elements * ly.out_elements + bias
    elif ly.kind == "bn":
        return 2 * int(ly.out_shape[0])
    else:
        return 0

def _layer_macs(ly) -> int:
    if ly.kind == "conv":
        c_in = ly.in_shape[0]
        kernel_h, kernel_w = ly.kernel

        return (ly.out_elements * (c_in // ly.groups) * kernel_h * kernel_w)

    elif ly.kind == "linear":
        return ly.in_elements * ly.out_elements

    else:
        return 0

def arithmetic_intensity (graph: Graph) -> dict[str, Any]:
    graph_layers = list(graph)
    layers_names = {ly.name: ly for ly in graph_layers}

    per_ly_out: dict [str, dic[str, Any]] = {}
    total_flops = 0
    total_by = 0.0

    for i, ly in enumerate(graph_layers):
        in_dtype = graph.precision
        if ly.reads:
            in_dtype = layers_names[ly.reads[0]].act_dtype
        elif i> 0:
            in_dtype = graph_layers[i-1].act_dtype
        
        input_by = ly.in_elements * dtype_bytes(in_dtype)

        if ly.kind == "add" and len(ly.reads) == 2:
            input_by *=2

        weight_by = _layer_parameters(ly) * dtype_bytes(ly.weight_dtype)

        buffer_by = 0
        if ly.kind == "bn":
            buffer_by = (BN_BUFFERS_PER_CHANNEL * ly.out_shape[0] * dtype_bytes(BUFFER_DTYPE))

        out_by = ly.out_elements * dtype_bytes(ly.act_dtype)

        bytes_moved = float(input_by + weight_by + buffer_by + out_by)

        flops = int(_layer_macs(ly) * FLOPS_PER_MAC)

        inten = round(flops/bytes_moved, 4) if bytes_moved > 0 else 0.0

        per_ly_out[ly.name] = {
            "flops": flops,
            "bytes": bytes_moved,
            "intensity": inten
        }       
        total_flops += flops
        total_by += bytes_moved
    overall = round(total_flops / total_by, 4) if total_by > 0 else 0.0

    return computed(
        overall,
        f"{graph.name}: FLOPs over bytes, every byte counted once",
        per_layer=per_ly_out,
        total_flops=total_flops,
        total_bytes=total_by,
        flops_per_mac=FLOPS_PER_MAC,
        cache_reuse_assumed=False,
        note="no fusion and no cache reuse; both would lower the byte column",
    )



# conv = Layer(
#     name="conv1",
#     kind="conv",
#     in_shape=(4, 8, 8),
#     out_shape=(8, 6, 6),
#     kernel=(3, 3),
# )

# relu = Layer(
#     name="relu1",
#     kind="relu",
#     in_shape=(8, 6, 6),
#     out_shape=(8, 6, 6),
#     reads=("conv1",),
# )


# print("Convolution parameters:", _layer_parameters(conv))
# print("Convolution MACs:", _layer_macs(conv))
# print("ReLU parameters:", _layer_parameters(relu))
# print("ReLU MACs:", _layer_macs(relu))

# graph = Graph(
#     name="test_graph",
#     input_shape=(4, 8, 8),
#     layers=[conv, relu],
#     precision="fp32",
# )

# result = arithmetic_intensity(graph)

# print("\nFull arithmetic intensity result:")
# print(result)
