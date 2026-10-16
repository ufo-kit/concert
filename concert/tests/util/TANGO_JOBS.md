# Tango job experiment

This is a test-only prototype, not a public Concert API. Production devices,
clients and timeout settings are unchanged.

Run from the Concert repository:

```sh
pytest -q concert/tests/integration/test_tango_jobs.py
```

PyTango is optional; the module is skipped when it is unavailable. The suite starts
a real asyncio Tango server in a separate process using `DeviceTestContext`, with
an ephemeral file database and local network sockets. No Tango database server,
hardware or GPU is needed. The client uses an explicit executor for the test's
running loop rather than Concert's IPython-specific loop selection.

For interactive experiments, start the device in a separate terminal:

```sh
python -m tango.test_context concert.tests.util.tango_jobs.JobTestDevice --host 127.0.0.1
```

Use the printed device URI with `get_tango_device(uri)` in a Concert/IPython
session, set its timeout to 500 ms, and import `RemoteJob` from
`concert.tests.util.tango_jobs` for the example below.

`JobTestDevice` supplies immediate, delayed, failing, cleanup and executor-backed
commands. `@job_command` retains the ordinary Tango command and registers it with
the experimental `StartJob`/`GetJob`/`CancelJob` protocol. One server-side `Job`
object holds each request, active task and eventual outcome; its task reference is
released on completion. A read-only `job_changed` attribute sends an individual
status update (`id`, `state`). It contains only the latest change, not
a list of jobs or a replay of history. Results and missed updates come from `GetJob`.

In a Concert/IPython session:

```python
from concert.networking.base import get_tango_device
from concert.tests.util.tango_jobs import RemoteJob

proxy = get_tango_device("tango://127.0.0.1:8888/test/nodb/jobtestdevice#dbase=no")
proxy.set_timeout_millis(500)
job = await RemoteJob.start(proxy, "slow", 10.0)
print(job.id)
print(await proxy.echo())  # Ordinary Tango calls stay on the device proxy.
result = await job
```

Each `RemoteJob` represents exactly one operation and owns its event subscription,
cached state and result future. There is no device wrapper or client-side job
registry. Monitoring starts immediately, without waiting for `await job`, and
releases its subscription automatically on completion or a connection deadline.
Independent handles have independent subscriptions, even for the same job ID.

Save `job.id` and use `await RemoteJob.attach(proxy, job_id)` to reconnect.
Cancelling or timing out `await job` only stops that wait. `await job.cancel_remote()`
requests server cancellation; awaiting the job then waits for cleanup and raises
`RemoteJobCancelled`. Use `await job.close()` to stop local monitoring early,
without cancelling the server operation. Closing is safe to repeat and does not
affect other handles. No context manager is needed. If automatic unsubscription
fails, it is logged; `close()` retries releasing the subscription.

`job.state` reads the latest known status without making a network request.
Tango's event thread caches updates for this job, so this works even
between terminal IPython cells, when the asyncio loop is stopped. Status cannot
move backwards, so duplicate or late events need no revision counter. Result
retrieval, future completion and fallback reads still run on the
asyncio loop. If an event is lost while that loop is stopped, the cached state can
remain stale until the loop runs again and reconciles it.

The integration tests use a 500 ms RPC timeout for an operation lasting 1.1 s,
exercise actual change events with a 60 s reconciliation interval, deliberately
suppress events to test fallback reads, and inject connection failures and a lost
start reply. A new client attaches through a separate real proxy. `Init` tests
loss of job history; it is not an operating-system process-crash test.

## Intentional prototype limits

- The generic decorator accepts void, string or float inputs. Only the test
  commands restrict durations to 0–60 seconds. Results must be JSON serializable.
  A general Tango/NumPy/binary codec is deferred.
- One background job is admitted at a time, with no queue. Legacy direct calls
  retain their existing behavior and do not share this admission guard yet.
- Only commands registered with `@job_command` can be started as jobs. Tango's
  `is_<command>_allowed` hooks are not called for job execution.
- Job IDs are plain UUIDs. Only the last 16 finished jobs are retained in memory.
  Idempotent start retries are guaranteed only while the record is retained.
  After eviction, restart or `Init`, attaching returns `UNAVAILABLE`, but submitting
  the same ID again starts a new operation. Do not automatically retry forgotten
  IDs if repeating the operation's effects is unsafe. No durable execution is promised.
- Transport failures during submission or subsequent subscription setup carry
  the requested ID as `JobConnectionError`, with
  the original Tango exception as the cause. Explicit admission rejections remain
  Tango exceptions. The transport reason list needs testing across PyTango versions.
- Events update the status cache and trigger result reconciliation. Each handle
  queries its record after events, on attachment and periodically while pending.
- Connection retries use a fixed bounded delay for this experiment; production
  backoff, retention policy, request-size limits and diagnostics are deferred.
- Cancellation of `start()` itself can leave acceptance ambiguous; callers that
  need recovery must supply and retain `job_id` before submitting.
- Device initialization/shutdown awaits cooperative cancellation. An uncooperative
  operation can delay teardown. Threads are never forcibly terminated.

The purpose is to validate lifecycle semantics with the installed PyTango before
moving the reusable pieces into production modules or migrating reconstruction.
