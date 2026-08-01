# [BugFix][DBO] Propagate ubatch startup failures instead of hanging at the ready barrier

## What this PR does / why we need it?

This PR fixes a DBO startup deadlock in `AscendUBatchWrapper`.

During DBO execution, the main thread starts two ubatch threads and waits for
them at a three-party `threading.Barrier`. In the CUDAGraph capture path, each
ubatch thread initializes its NPU device and BLAS handles before it enters the
barrier. If an operation such as `torch.npu.set_device()` or
`torch.npu.current_blas_handle()` raises, the thread currently records the
exception and exits without reaching the barrier. The main thread then waits
forever at `ready_barrier.wait()` and never reaches the existing
`_check_thread_exceptions()` call.

The same unbounded ready-barrier wait also exists in the non-CUDAGraph ubatch
runtime path. In addition, a main-thread exception after the barrier has been
released but before the first `cpu_wait_event` is signalled can leave both
ubatch threads blocked indefinitely.

The change should introduce a shared ubatch failure coordinator used by both
capture and runtime paths. Its required behavior is:

1. Preserve the first child-thread `exc_info` and make it available to the
   main thread.
2. Abort the ready barrier immediately on every child-thread startup failure.
3. Use a finite timeout while waiting for the ready barrier.
4. Wake all CPU event waiters during failure cleanup.
5. Join child threads with a finite timeout and report threads that remain
   alive as a fatal worker failure.
6. Recreate the ready barrier before the next invocation after an abort.
7. Clean up `_THREAD_ID_TO_CONTEXT` and `_CURRENT_CONTEXTS` when
   `AscendUBatchContext.__enter__` fails before `__exit__` can run.

The main thread must re-raise the original child exception with its traceback
context when one is available. A `BrokenBarrierError` or timeout is only the
fallback diagnostic when no child exception was recorded.

### Failure sequence fixed by this PR

```text
main thread                    ubatch thread 0              ubatch thread 1
-----------                    ---------------              ---------------
start thread 0                 set_device() raises
start thread 1                 record exception
ready_barrier.wait()           exit before barrier
                               barrier.abort()
BrokenBarrierError
join / cleanup
raise original set_device exception
```

Without the abort, the main thread and the surviving ubatch thread wait
forever for the failed thread to arrive at the barrier.

## Does this PR introduce any user-facing change?

No API or configuration change is introduced.

On a NPU initialization or ubatch startup failure, the server changes from an
infinite hang to a bounded failure that reports the original exception. This
is an intentional reliability fix.

## How was this patch tested?

The implementation must add CPU-only unit tests; no NPU hardware is required
for the failure injection cases.

- Mock `torch.npu.set_device` to raise before the child enters the ready
  barrier. Assert that the caller returns within a bounded time, raises the
  injected exception with its context, aborts the barrier, and leaves no live
  ubatch thread.
- Mock `torch.npu.current_blas_handle` to raise during CUDAGraph thread
  initialization and assert the same behavior.
- Inject a main-thread failure after the ready barrier is released and before
  the first `cpu_wait_event` is set. Assert that both child threads are
  released and joined.
- Verify that a subsequent invocation creates a usable barrier after a failed
  invocation, rather than reusing an aborted barrier.
- Verify that a failed `AscendUBatchContext.__enter__` removes its thread-id
  and current-context registrations.
- Run the focused DBO unit-test module and the existing DBO precision suite.
- Run the TP=2 DeepSeek-V2 DBO E2E validation to confirm the normal overlap
  path and throughput remain unchanged.

The test commands and E2E result should be added to this section when the
implementation PR is ready for submission.

## Review checklist

- [ ] Both `_capture_ubatches` and `_run_ubatches` use the same failure and
      cleanup contract.
- [ ] No code path waits indefinitely on a ready barrier, CPU event, or child
      thread after an ubatch startup failure.
- [ ] The original exception is preferred over a generic barrier error.
- [ ] Cleanup does not silently abandon a live non-daemon NPU thread.
- [ ] The normal two-ubatch scheduling and CUDAGraph capture paths remain
      unchanged when no error occurs.
