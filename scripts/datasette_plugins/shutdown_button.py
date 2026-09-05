"""A "Stop server" button for the Datasette inspection UI.

Adds a floating button to every page that POSTs to `/-/shutdown`, which raises
SIGINT against Datasette's own process. Uvicorn handles that as a graceful
shutdown — in-flight requests finish, connections close, the CLI returns and the
terminal is back at a prompt.

The database is never at risk: Datasette holds only read-only (`mode=ro`)
connections, so there is nothing to flush and nothing to roll back.

Guards, because this endpoint stops a process:
  * POST only — a stray GET or a prefetched link cannot trigger it.
  * CSRF-checked by Datasette's own asgi-csrf middleware.
  * Refused unless the request's Host header is loopback, so the button is inert
    if Datasette is ever bound to a non-local interface with `-h 0.0.0.0`.
"""

import asyncio
import os
import signal

from datasette import hookimpl
from datasette.utils.asgi import Response

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

# Delay before signalling, so this request's response reaches the browser first.
_SHUTDOWN_DELAY_SECS = 0.25


def _is_loopback(request) -> bool:
    host = (request.headers.get("host") or "").strip()
    if host.startswith("["):  # [::1]:8001
        host = host[1:].split("]", 1)[0]
    else:
        host = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return host in _LOOPBACK_HOSTS


async def shutdown(request):
    if request.method != "POST":
        return Response.json({"ok": False, "error": "POST required"}, status=405)
    if not _is_loopback(request):
        return Response.json(
            {"ok": False, "error": "Shutdown is only permitted from localhost"}, status=403
        )
    asyncio.get_running_loop().call_later(
        _SHUTDOWN_DELAY_SECS, os.kill, os.getpid(), signal.SIGINT
    )
    return Response.json({"ok": True})


@hookimpl
def register_routes():
    return [(r"^/-/shutdown$", shutdown)]


_BUTTON_JS = """
(function () {
  if (document.getElementById('ds-shutdown-btn')) { return; }
  var token = '__CSRFTOKEN__';
  var btn = document.createElement('button');
  btn.id = 'ds-shutdown-btn';
  btn.type = 'button';
  btn.textContent = 'Stop server';
  btn.title = 'Shut down Datasette and return the terminal to a prompt';
  btn.style.cssText = 'position:fixed;bottom:1rem;right:1rem;z-index:9999;' +
    'padding:.5rem .9rem;border:0;border-radius:4px;background:#b3261e;color:#fff;' +
    'font:500 13px system-ui,-apple-system,sans-serif;cursor:pointer;' +
    'box-shadow:0 1px 4px rgba(0,0,0,.3)';

  function finish() {
    var msg = document.createElement('div');
    msg.style.cssText = 'position:fixed;inset:0;z-index:10000;display:flex;' +
      'align-items:center;justify-content:center;background:#1b1b1b;color:#eee;' +
      'font:400 15px/1.6 system-ui,-apple-system,sans-serif;text-align:center;padding:2rem';
    msg.innerHTML = '<div><p style="font-size:19px;margin:0 0 .6rem"><strong>' +
      'Datasette has stopped.</strong></p>' +
      '<p style="margin:0 0 .6rem">The database was untouched, and your terminal is ' +
      'back at a prompt.</p>' +
      '<p style="margin:0;color:#aaa;font-size:13px">If this tab did not close by ' +
      'itself, close it with Ctrl+W \\u2014 browsers only let a page close itself when ' +
      'a script opened it.</p></div>';
    document.body.appendChild(msg);
    setTimeout(function () { window.close(); }, 400);
  }

  btn.addEventListener('click', function () {
    if (!window.confirm('Stop the Datasette server?\\n\\nThe database is not affected.')) {
      return;
    }
    btn.disabled = true;
    btn.textContent = 'Stopping\\u2026';
    // The server exits moments after replying, so a dropped connection is also success.
    fetch('/-/shutdown', {
      method: 'POST',
      headers: {'x-csrftoken': token}
    }).then(function (r) {
      if (r.ok) { finish(); return; }
      return r.text().then(function (t) {
        btn.disabled = false;
        btn.textContent = 'Stop server';
        window.alert('Shutdown refused (HTTP ' + r.status + '):\\n' + t);
      });
    }).catch(finish);
  });

  document.body.appendChild(btn);
})();
"""


@hookimpl
def extra_body_script(request):
    try:
        token = request.scope["csrftoken"]()
    except (KeyError, TypeError, AttributeError):
        token = ""
    return _BUTTON_JS.replace("__CSRFTOKEN__", token)
