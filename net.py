"""Net resource engine: granted-address template matching and HTTP request
execution.

The granted ``address`` is a URL *template*: its fixed parts are locked,
``*`` stands for exactly one path segment and ``**`` for the rest of the path.
A scheme ending in ``*`` matches with an optional trailing 's' (``http*``
accepts both ``http`` and ``https``; ``ftp*`` accepts ``ftp`` and ``ftps``);
a plain scheme is exact.

Policy violations (scheme/host/path/query outside the grant, header shadowing)
raise :class:`NetError` with the sticky ``#PERM`` code. I/O failures (refused
connection, too many redirects, a redirect leaving the granted origin) raise it
with ``#N/A``. The language layer converts the carried code into a Grid sticky
error value, so policy decisions are observable in cells.
"""

import re
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request

PERM = '#PERM'
IOERR = '#N/A'

# The host engine sets this to the running compiler so server member builtins
# (Start/Serve/Stop) can reach the executor's request pump.
ENGINE = None


class NetError(Exception):
    """A Net policy or I/O failure carrying the Grid error code to surface."""

    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Address template parsing and matching
# ---------------------------------------------------------------------------

_SCHEME_RE = re.compile(r'^[a-zA-Z][a-zA-Z0-9.+-]*$')
_DEFAULT_PORTS = {'http': 80, 'https': 443, 'ftp': 21, 'sftp': 22,
                  'ftps': 990, 'ws': 80, 'wss': 443}


def _effective_port(parts):
    if parts.port is not None:
        return parts.port
    return _DEFAULT_PORTS.get((parts.scheme or '').lower())


def _norm_path(path):
    """Split a URL path into non-empty segments (trailing slash is optional)."""
    return [seg for seg in (path or '').split('/') if seg != ''] if path else []


def _clean_segment(seg):
    """A wildcard fill may not be empty, a dot, or smuggle a separator."""
    if seg in ('', '.', '..'):
        return False
    return not re.search(r'[/?#:*]', seg)


def _segments_match(pattern, url):
    """Glob-match path segments: '*' = exactly one, '**' = the rest.

    * '**' consumes zero or more segments (each clean) and backtracks.
    * '*' consumes exactly one clean segment.
    * any other pattern segment must match the URL segment literally.
    """
    if not pattern:
        return not url
    head = pattern[0]
    if head == '**':
        return (_segments_match(pattern[1:], url)
                or (bool(url) and _clean_segment(url[0])
                    and _segments_match(pattern, url[1:])))
    if head == '*':
        return (bool(url) and _clean_segment(url[0])
                and _segments_match(pattern[1:], url[1:]))
    return bool(url) and url[0] == head and _segments_match(pattern[1:], url[1:])


def _query_globe(patq):
    out = '^'
    for ch in patq:
        if ch == '*':
            out += '[^&=]*'
        else:
            out += re.escape(ch)
    return out + '$'


def _query_matches(pattern_query, url_query):
    """A template query allows '*' inside a component value; otherwise exact."""
    if pattern_query is None:
        return url_query is None
    if url_query is None:
        return False
    if '*' not in pattern_query:
        return pattern_query == url_query
    return re.match(_query_globe(pattern_query), url_query) is not None


def parse_address(pattern):
    """Parse the granted address template into its matching parts.

    Raises NetError(#PERM) when the template itself is malformed.
    """
    if not isinstance(pattern, str):
        raise NetError('Granted address must be text', PERM)
    scheme_end = pattern.find('://')
    if scheme_end == -1:
        raise NetError(
            'Granted address must include a scheme, e.g. http*://example.com',
            PERM)
    scheme_part = pattern[:scheme_end]
    optional_s = scheme_part.endswith('*')
    base_scheme = scheme_part.rstrip('*')
    if not _SCHEME_RE.match(base_scheme):
        raise NetError('Invalid scheme in granted address', PERM)
    base_scheme = base_scheme.lower()
    rest = pattern[scheme_end + 3:]
    try:
        parsed = urllib.parse.urlsplit(f'{base_scheme}://{rest}')
    except ValueError as exc:
        raise NetError(f'Invalid granted address: {exc}', PERM) from exc
    if parsed.username or parsed.password:
        raise NetError('Userinfo is not allowed in a granted address', PERM)
    if parsed.hostname is None:
        raise NetError('Granted address is missing its host', PERM)
    if '*' in (parsed.hostname or '') or '*' in (parsed.netloc or ''):
        raise NetError(
            'Wildcards are not allowed in the authority (host/port)', PERM)
    path_segs = _norm_path(parsed.path)
    if any('*' in seg and seg not in ('*', '**') for seg in path_segs):
        raise NetError(
            "Wildcards must be whole segments ('*' or '**')", PERM)
    return {
        'base': base_scheme,
        'optional_s': optional_s,
        'host': (parsed.hostname or '').lower(),
        'port': _effective_port(parsed),
        'path': path_segs,
        'query': parsed.query or None,
        'pattern': pattern,
    }


def validate_address(pattern, url):
    """Validate a requested URL against the granted address template.

    Raises NetError(#PERM) on any mismatch. Performs no network I/O.
    """
    template = parse_address(pattern)
    if not isinstance(url, str):
        raise NetError('Connect expects a text URL', PERM)
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError as exc:
        raise NetError(f'Invalid URL: {exc}', PERM) from exc
    scheme = (parts.scheme or '').lower()
    if template['optional_s']:
        scheme_ok = scheme == template['base'] or scheme == template['base'] + 's'
    else:
        scheme_ok = scheme == template['base']
    if not scheme_ok:
        raise NetError(
            f"Scheme '{parts.scheme}' is not allowed by the granted address "
            f"'{template['pattern']}'", PERM)
    if parts.username or parts.password:
        raise NetError('Userinfo is not allowed in the URL', PERM)
    if (parts.hostname or '').lower() != template['host']:
        raise NetError(
            f"Host '{parts.hostname}' is not allowed by the granted address",
            PERM)
    if _effective_port(parts) != template['port']:
        raise NetError(
            f"Port '{parts.port}' is not allowed by the granted address",
            PERM)
    if not _segments_match(template['path'], _norm_path(parts.path)):
        raise NetError(
            'The URL path does not match the granted address template', PERM)
    if not _query_matches(template['query'], parts.query or None):
        raise NetError(
            'The query string is not allowed by the granted address', PERM)
    return True


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

_HEADER_TOKEN_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def parse_header_list(items):
    """Normalize ``["Name: value", ...]`` into ``[(name, value), ...]``.

    Raises NetError(#PERM) on any malformed entry.
    """
    if items is None:
        return []
    if isinstance(items, str):
        items = [items]
    result = []
    for item in items:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            name, value = item
        elif isinstance(item, str) and ':' in item:
            name, _, value = item.partition(':')
        else:
            raise NetError(
                f"Invalid header entry '{item}'; expected \"Name: value\"",
                PERM)
        name = str(name).strip()
        value = str(value).strip()
        if not _HEADER_TOKEN_RE.match(name):
            raise NetError(f"Invalid header name '{name}'", PERM)
        result.append((name, value))
    return result


def merge_headers(pinned, conn_extras, op_extras):
    """Merge header lists with precedence pinned > operation > connection.

    A pinned header name may not be shadowed by any extra (raised as
    NetError(#PERM)). Extras override by name, last one winning.
    """
    pinned = parse_header_list(pinned)
    pinned_names = {name.lower() for name, _ in pinned}
    merged = dict(pinned)
    for extra in (op_extras, conn_extras):
        for name, value in parse_header_list(extra):
            if name.lower() in pinned_names:
                raise NetError(
                    f"Cannot override the pinned header '{name}'", PERM)
            merged[name.lower()] = value
    return list(merged.items())


def compose_headers(base, pinned, op_extras):
    """Overlay operation headers onto a connection's effective header list.

    ``base`` is the already-merged effective set (pinned grant headers plus
    the connection extras, shadow-checked at connect time). An operation
    header may override a connection extra but never a pinned name (raised as
    NetError(#PERM)), preserving the pinned > operation > connection order.
    """
    pin_names = {name.lower() for name, _ in parse_header_list(pinned)}
    merged = dict(parse_header_list(base))
    for name, value in parse_header_list(op_extras):
        if name.lower() in pin_names:
            raise NetError(
                f"Cannot override the pinned header '{name}'", PERM)
        merged[name.lower()] = value
    return list(merged.items())


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class _NetRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow at most ``cap`` redirects and never leave the granted origin."""

    def __init__(self, cap, origin):
        super().__init__()
        self._cap = cap
        self._origin = origin
        self._count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self._count += 1
        if self._count > self._cap:
            raise NetError(
                f'Too many redirects (granted limit {self._cap})', IOERR)
        target = urllib.parse.urlsplit(newurl)
        if ((target.scheme or '').lower(),
                (target.hostname or '').lower(),
                _effective_port(target)) != self._origin:
            raise NetError(
                'Redirect refused: the target leaves the granted origin', IOERR)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def request(method, url, headers, body, max_redirects):
    """Perform one request and return ``(status, body_text)``.

    ``headers`` is a list of ``(name, value)`` pairs (already merged and
    shadow-checked).  ``max_redirects`` caps same-origin redirects.  Raises
    NetError(#N/A) on I/O or redirect failures.
    """
    parts = urllib.parse.urlsplit(url)
    origin = ((parts.scheme or '').lower(),
              (parts.hostname or '').lower(),
              _effective_port(parts))
    header_dict = {}
    for name, value in headers:
        header_dict.setdefault(name, value)
    data = None
    if body is not None and method not in ('GET', 'HEAD'):
        data = str(body).encode('utf-8')
    req = urllib.request.Request(
        url, data=data, headers=header_dict, method=method)
    opener = urllib.request.build_opener(
        _NetRedirectHandler(max_redirects, origin))
    try:
        with opener.open(req, timeout=30) as resp:
            status = resp.getcode()
            raw = resp.read()
    except NetError:
        raise
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            raw = exc.read()
        except Exception:
            raw = b''
    except urllib.error.URLError as exc:
        raise NetError(f'Request failed: {exc.reason}', IOERR) from exc
    except OSError as exc:
        raise NetError(f'Request failed: {exc}', IOERR) from exc
    return status, raw.decode('utf-8', 'replace')


# ---------------------------------------------------------------------------
# Server engine: LISTEN grants, route matching, HTTP/1.1 responses,
# and the pluggable bind/acceptor transport.
# ---------------------------------------------------------------------------

_STANDARD_VERBS = ('GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD')

_REASON_TEXT = {
    200: 'OK', 201: 'Created', 204: 'No Content', 301: 'Moved Permanently',
    302: 'Found', 400: 'Bad Request', 403: 'Forbidden', 404: 'Not Found',
    405: 'Method Not Allowed', 500: 'Internal Server Error',
}


def normalize_methods(raw):
    """Collapse a granted/route method set into a list of uppercase verbs."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip().upper()] if raw.strip() else []
    return [str(m).strip().upper() for m in (raw or []) if m]


def check_server_mode(methods):
    """LISTEN is exclusive: a grant granting any other verb alongside it is
    refused (#PERM). Returns ``['LISTEN']`` when granted."""
    methods = normalize_methods(methods)
    if 'LISTEN' in methods and len(methods) != 1:
        raise NetError(
            'LISTEN is exclusive: a server grant may grant only LISTEN', PERM)
    return methods


def validate_server_address(pattern):
    """The server grant's address is the bind target: an http(s) URL template
    whose authority (host/port) is concrete (wildcards already rejected by
    parse_address)."""
    template = parse_address(pattern)
    if template['base'] not in ('http', 'https'):
        raise NetError(
            'A server grant must bind an http(s) address', PERM)
    return template


def route_matches(template, path, method, allowed):
    """Whether a request (method + path) matches a route template.

    ``allowed`` restricts the verbs that route serves; the path template uses
    the same segment/glob rules as the granted address.
    """
    if method.upper() not in [m.upper() for m in (allowed or [])]:
        return False
    return _segments_match(_norm_path(template), _norm_path(path))


def route_specificity(template):
    """Most-specific-wins score: (literal segments, total segments). More
    literals beats fewer; ties go to the longer template."""
    segs = _norm_path(template)
    literals = sum(1 for s in segs if s not in ('*', '**'))
    return (literals, len(segs))


def extend_route(template, endpoint):
    """Derive a nested sub-route template by pinning the first wildcard.

    ``search/*`` plus ``doc`` becomes ``search/doc``: the wildcard is replaced
    by the endpoint's segments (which may themselves be a path). A template
    with no wildcard appends the endpoint at the end.
    """
    base = [seg for seg in (template or '').split('/') if seg]
    extra = [seg for seg in (endpoint or '').split('/') if seg]
    for i, seg in enumerate(base):
        if seg in ('*', '**'):
            return '/' + '/'.join(base[:i] + extra + base[i + 1:])
    return '/' + '/'.join(base + extra)


def make_response(status, body, headers=None, pinned=None):
    """Build the raw HTTP/1.1 response bytes for a ``Reply`` or an automatic
    404. Pinned grant headers are protected case-insensitively — RFC 7230
    field names — so a per-reply header whose lower-cased name matches a
    pinned one is refused (NetError(#PERM)), matching the client side. Header
    names are written back with the case the author gave them."""
    if body is None:
        body = ''
    data = str(body).encode('utf-8')
    pinned_items = parse_header_list(pinned)
    pin = {name.lower() for name, _ in pinned_items}
    merged = list(pinned_items)
    for name, value in parse_header_list(headers):
        if name.lower() in pin:
            raise NetError(
                f"Cannot override the pinned header '{name}'", PERM)
        merged.append((name, value))
    have = {name.lower() for name, _ in merged}
    if 'content-type' not in have:
        merged.append(('Content-Type', 'text/plain; charset=utf-8'))
    if 'content-length' not in have:
        merged.append(('Content-Length', str(len(data))))
    reason = _REASON_TEXT.get(int(status), '')
    out = [f"HTTP/1.1 {int(status)} {reason}\r\n"]
    for name, value in merged:
        out.append(f"{name}: {value}\r\n")
    out.append("Connection: close\r\n\r\n")
    return ''.join(out).encode('latin-1', 'replace') + data


def parse_http_request(rfile):
    """Read one HTTP/1.1 request from a file-like object.

    Returns a dict {method, path, raw_path, headers, body} or None at EOF.
    ``raw_path`` strips a trailing query string; ``headers`` are lower-cased.
    """
    line = rfile.readline()
    if not line:
        return None
    parts = line.decode('latin-1', 'replace').rstrip('\r\n').split(' ', 2)
    if len(parts) < 2:
        return None
    method = parts[0].upper()
    path = parts[1]
    headers = {}
    while True:
        raw = rfile.readline()
        if not raw or raw in (b'\r\n', b'\n'):
            break
        key, _, val = raw.decode('latin-1', 'replace').partition(':')
        headers[key.strip().lower()] = val.strip()
    try:
        length = int(headers.get('content-length') or 0)
    except ValueError:
        length = 0
    body = (rfile.read(max(length, 0)).decode('utf-8', 'replace')
            if length else '')
    return {'method': method, 'path': path, 'raw_path': path.split('?', 1)[0],
            'headers': headers, 'body': body}


class _SocketConn:
    """File-like peer over a raw TCP socket: the engine's reply paths call
    ``write``/``flush``/``close`` (mirroring the fake-listener contract).
    ``close`` also signals the ``done`` event so a connection-handler thread
    can stop waiting once the engine has replied."""

    def __init__(self, sock, done=None):
        self._sock = sock
        self._done = done

    def write(self, data):
        self._sock.sendall(data)

    def flush(self):
        pass

    def close(self):
        if self._sock.fileno() >= 0:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._done is not None:
            self._done.set()


class _TCPServerListener:
    """Blocking TCP acceptor; call ``serve_forever(on_request)`` on a daemon
    thread. Each accepted connection gets its own handler thread which parses
    the request (10 s socket timeout), hands it to the main thread via
    ``on_request`` (which only appends to a plain queue), then waits for the
    main thread to Reply()/close the ``_SocketConn``. The accept loop never
    blocks on a client, so many connections can be in flight. ``stop()``
    closes the listening socket and any in-flight connections."""

    def __init__(self, host, port):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, int(port)))
        self._sock.listen(8)
        self._stopped = False
        self._conns = set()
        self._lock = threading.Lock()

    def serve_forever(self, on_request):
        while not self._stopped:
            try:
                conn, _addr = self._sock.accept()
            except OSError:
                break
            t = threading.Thread(
                target=self._handle_conn, args=(conn, on_request), daemon=True)
            t.start()

    def _handle_conn(self, conn, on_request):
        conn.settimeout(10)
        done = threading.Event()
        with self._lock:
            self._conns.add(conn)
        try:
            conn_request = parse_http_request(conn.makefile('rb'))
            if conn_request is None:
                return
            conn_request['conn'] = _SocketConn(conn, done)
            on_request(conn_request)
            # Wait until the main thread replies and closes (or timeout, as a
            # safety valve if the engine stalls).
            done.wait(timeout=10)
        except (OSError, ValueError, AttributeError):
            pass
        finally:
            with self._lock:
                self._conns.discard(conn)
            try:
                conn.close()
            except OSError:
                pass

    def stop(self):
        self._stopped = True
        try:
            self._sock.close()
        except OSError:
            pass
        with self._lock:
            conns = list(self._conns)
        for conn in conns:
            try:
                conn.close()
            except OSError:
                pass


def bind_server(host, port):
    """Create a listening server transport bound to ``host:port``.

    Test hosts may replace this with a synchronous fake transport (see the
    test suite); a real listener starts accepting only from a daemon thread
    spawned by the engine's ``net!server.Start`` builtin.
    """
    return _TCPServerListener(host, port)