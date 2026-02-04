from memory import UnsafePointer
from gpu import thread_idx
from gpu.host import DeviceContext
from testing import assert_equal

# ANCHOR: transpose_raw
comptime HEIGHT = 3
comptime WIDTH = 2
comptime THREADS_PER_BLOCK = (4, 4) # 16 threads for 6 elements!
comptime dtype = DType.float32

fn transpose_kernel(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    width: UInt,
    height: UInt,
):
    # Input is size (height, width)
    # Output is size (width, height)
    
    # Current thread coordinates (x=col, y=row)
    in_col = thread_idx.x
    in_row = thread_idx.y

    if in_row < height and in_col < width:
        idx_input = width * in_row + in_col
        idx_output = height * in_col + in_row
        output[idx_output] = input[idx_input]


# ANCHOR_END: transpose_raw

def main():
    with DeviceContext() as ctx:
        # Create Input: 3 rows, 2 cols (Total 6)
        # 0 1
        # 2 3
        # 4 5
        a = ctx.enqueue_create_buffer[dtype](HEIGHT * WIDTH)
        with a.map_to_host() as a_host:
            for i in range(HEIGHT * WIDTH):
                a_host[i] = i
        
        # Create Output: 2 rows, 3 cols (Total 6)
        # Expected:
        # 0 2 4
        # 1 3 5
        out = ctx.enqueue_create_buffer[dtype](WIDTH * HEIGHT)
        out.enqueue_fill(0)
        
        ctx.enqueue_function[transpose_kernel, transpose_kernel](
            out, a, UInt(WIDTH), UInt(HEIGHT),
            grid_dim=1, block_dim=THREADS_PER_BLOCK
        )
        ctx.synchronize()
        
        # Verification
        expected = [0.0, 2.0, 4.0, 1.0, 3.0, 5.0]
        with out.map_to_host() as out_host:
            print("Output:", out_host)
            for i in range(WIDTH * HEIGHT):
                assert_equal(out_host[i], expected[i].cast[DType.float32]())