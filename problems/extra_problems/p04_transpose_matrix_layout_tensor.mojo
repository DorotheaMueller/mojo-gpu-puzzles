from gpu import thread_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from testing import assert_equal

# ANCHOR: transpose_layout
comptime HEIGHT = 3
comptime WIDTH = 2
comptime THREADS_PER_BLOCK = (4, 4)
comptime dtype = DType.float32

# 1. Define the layouts
# Input matches standard 3x2
comptime in_layout = Layout.row_major(HEIGHT, WIDTH)
# Output matches transposed 2x3
comptime out_layout = Layout.row_major(WIDTH, HEIGHT)

fn transpose_layout_kernel[
    out_layout: Layout,
    in_layout: Layout
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    input: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
):
    # Standard mapping: y is row, x is col
    row = thread_idx.y
    col = thread_idx.x
    
    if row < UInt(input.shape[0]()) and col < UInt(input.shape[1]()):
        output[col, row] = input[row, col]

# ANCHOR_END: transpose_layout

def main():
    with DeviceContext() as ctx:
        a_buf = ctx.enqueue_create_buffer[dtype](HEIGHT * WIDTH)
        with a_buf.map_to_host() as host:
             for i in range(HEIGHT * WIDTH): host[i] = i
             
        out_buf = ctx.enqueue_create_buffer[dtype](WIDTH * HEIGHT)
        out_buf.enqueue_fill(0)

        # Create Tensors wrapping the buffers
        a_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](a_buf)
        out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](out_buf)

        comptime kernel = transpose_layout_kernel[out_layout, in_layout]
        ctx.enqueue_function[kernel, kernel](
            out_tensor, a_tensor,
            grid_dim=1, block_dim=THREADS_PER_BLOCK
        )
        ctx.synchronize()

        # Verify
        expected = [0.0, 2.0, 4.0, 1.0, 3.0, 5.0]
        with out_buf.map_to_host() as out_host:
            print("Output:", out_host)
            for i in range(WIDTH * HEIGHT):
                assert_equal(out_host[i], expected[i].cast[DType.float32]())