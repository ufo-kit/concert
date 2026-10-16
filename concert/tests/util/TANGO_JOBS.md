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
session, set its timeout to 500 ms, and import `JobDeviceProxy` from
`concert.tests.util.tango_jobs` for the example below.

`JobTestDevice` supplies immediate, delayed, failing, cleanup and executor-backed
commands. `@job_command` retains the ordinary Tango command and registers it with
the experimental `StartJob`/`GetJob`/`CancelJob` protocol. A read-only `jobs`
attribute exposes metadata and sends change events. Results stay in `GetJob`.

In a Concert/IPython session:

```python
from concert.networking.base import get_tango_device
from concert.tests.util.tango_jobs import JobDeviceProxy

proxy = get_tango_device("tango://127.0.0.1:8888/test/nodb/jobtestdevice#dbase=no")
proxy.set_timeout_millis(500)
device = JobDeviceProxy(proxy)

job = await device.start("slow", 10.0)
print(job.id)
print(await device.echo())
result = await job

await device.close()  # When finished with this client, possibly in a later cell.
```

The first `start()` or `attach()` initializes the event subscription and status
watcher automatically. Concurrent first calls share initialization; cancelling
one caller does not cancel setup for others. Failed setup releases its subscription
and can be retried. Ordinary device calls do not initialize monitoring.

Save `job.id` and use `await device.attach(job_id)` in another client to reconnect.
Cancelling or timing out `await job` only stops that wait. `await job.cancel_remote()`
requests server cancellation; awaiting the job then waits for cleanup and raises
`RemoteJobCancelled`. Closing a client releases its subscription and observer,
without cancelling server operations.
`close()` is safe to repeat, including before first use. A closed adapter cannot
start or attach jobs; create another adapter to reconnect. `async with` remains
available as optional shorthand for initialization and cleanup.

`job.state` reads the latest known status without making a network request.
Tango's event thread caches status snapshots, so this works even between terminal
IPython cells, when the asyncio loop is stopped. Snapshot revisions reject stale
events. Result retrieval, future completion and fallback reads still run on the
asyncio loop. If an event is lost while that loop is stopped, the cached state can
remain stale until the loop runs again and reconciles it.

The integration tests use a 500 ms RPC timeout for an operation lasting 1.1 s,
exercise actual change events with a 60 s reconciliation interval, deliberately
suppress events to test fallback reads, and inject connection failures and a lost
start reply. A new client attaches through a separate real proxy. `Init` tests
server-instance replacement; it is not an operating-system process-crash test.

## Intentional prototype limits

- Inputs are void, strings or bounded durations (0–60 seconds). Results must be
  JSON serializable. A general Tango/NumPy/binary codec is deferred.
- One background job is admitted at a time, with no queue. Legacy direct calls
  retain their existing behavior and do not share this admission guard yet.
- Conventional `is_<command>_allowed` checks are honored. Command-specific Tango
  access control, `fisallowed` options and production permission policy are deferred.
- Only the last 16 finished jobs are retained in memory. Idempotent start retries
  are guaranteed only while the record is retained, in the same server instance.
  Do not automatically retry an evicted ID. No durable execution is promised.
- Start transport failures carry the requested ID as `JobConnectionError`, with
  the original Tango exception as the cause. Explicit admission rejections remain
  Tango exceptions. The transport reason list needs testing across PyTango versions.
- Events update the status cache and trigger result reconciliation. The client
  queries records after events, on attachment and periodically while handles are
  pending. Batched status reconciliation is deferred.
- Connection retries use a fixed bounded delay for this experiment; production
  backoff, retention policy, request-size limits and diagnostics are deferred.
- Cancellation of `start()` itself can leave acceptance ambiguous; callers that
  need recovery must supply and retain `job_id` before submitting.
- Device initialization/shutdown awaits cooperative cancellation. An uncooperative
  operation can delay teardown. Threads are never forcibly terminated.

The purpose is to validate lifecycle semantics with the installed PyTango before
moving the reusable pieces into production modules or migrating reconstruction.
