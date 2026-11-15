"""Testbed check of the delivery model (Eq. 8) on a real Linux TCP stack with TLS 1.3.

Run inside an unprivileged network namespace so that netem can shape the loopback link:
  unshare -rn python3 scripts/testbed.py --out results/paper/table12_testbed.csv
netem adds half the RTT on each direction of the loopback interface (MTU 1500, GSO/TSO off) and the rate
limit on the uplink only (packets to the server port), as Eq. (8) assumes. For every link profile and budget the client opens a new TCP connection,
completes a TLS 1.3 handshake, sends one HTTP POST with the payload and waits for the
server's reply; the time from connect() to the reply is recorded.
"""
import argparse, csv, os, socket, ssl, subprocess, sys, tempfile, threading, time

PROFILES = [  # name, rate (kbit/s), RTT (ms)
    ("iraq_operator_low", 4500, 83),
    ("cellular_3g", 960, 300),
    ("geo_viasat", 1080, 684),
    ("gprs_2g", 8, 1000),
]
BUDGETS = [2000, 4000, 8000, 16000, 32000, 64000]
HDR = 600


def counters():
    """(SYN retransmissions, retransmitted segments) of this network namespace."""
    ext = open("/proc/net/netstat").read().split("\n"); snmp = open("/proc/net/snmp").read().split("\n")
    def get(lines, prefix, key):
        h = [l for l in lines if l.startswith(prefix)]
        return int(dict(zip(h[0].split()[1:], h[1].split()[1:]))[key])
    return get(ext, "TcpExt:", "TCPSynRetrans"), get(snmp, "Tcp:", "RetransSegs")


def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)


def set_link(rate_kbit, rtt_ms, port):
    """Uplink (client to server port): delay RTT/2 and the rate limit; downlink: delay RTT/2 only."""
    subprocess.run("tc qdisc del dev lo root 2>/dev/null", shell=True)
    d = rtt_ms / 2
    sh("tc qdisc add dev lo root handle 1: prio bands 3 priomap " + " ".join(["1"] * 16))
    sh(f"tc qdisc add dev lo parent 1:1 handle 10: netem delay {d}ms rate {rate_kbit}kbit limit 100000")
    sh(f"tc qdisc add dev lo parent 1:2 handle 20: netem delay {d}ms limit 100000")
    sh(f"tc qdisc add dev lo parent 1:3 handle 30: netem delay {d}ms limit 100000")
    sh(f"tc filter add dev lo parent 1: protocol ip prio 1 u32 match ip dport {port} 0xffff flowid 1:1")


def make_cert(d):
    key, crt = os.path.join(d, "k.pem"), os.path.join(d, "c.pem")
    sh(f"openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -keyout {key} -out {crt} "
       f"-days 2 -subj /CN=localhost >/dev/null 2>&1")
    return crt, key


def serve(sock, ctx, stop):
    while not stop.is_set():
        try:
            c, _ = sock.accept()
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # as HTTPS stacks do
        except OSError:
            break
        with ctx.wrap_socket(c, server_side=True) as s:
            head = b""
            while b"\r\n\r\n" not in head:
                head += s.recv(65536)
            h, body = head.split(b"\r\n\r\n", 1)
            n = int([l for l in h.split(b"\r\n") if l.lower().startswith(b"content-length")][0].split(b":")[1])
            while len(body) < n:
                body += s.recv(65536)
            s.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")


def attempt(port, cctx, nbytes):
    body = os.urandom(nbytes)
    head = (f"POST /verify HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/octet-stream\r\n"
            f"Content-Length: {nbytes}\r\n").encode()
    head += b"X-Pad: " + b"a" * max(0, HDR - len(head) - 11) + b"\r\n\r\n"
    t = time.perf_counter()
    raw = socket.create_connection(("127.0.0.1", port))
    raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    with cctx.wrap_socket(raw, server_hostname="localhost") as s:
        s.sendall(head + body)
        resp = b""
        while not resp.endswith(b"ok"):
            resp += s.recv(4096)
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--out", default="table12_testbed.csv")
    ap.add_argument("--profiles", default="all")
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    a = ap.parse_args()
    sh("ip link set lo up && ip link set lo mtu 1500")
    sh("ethtool -K lo gso off tso off")   # one wire packet per TCP segment
    open("/proc/sys/net/ipv4/tcp_no_metrics_save", "w").write("1")   # no cached ssthresh between attempts
    d = tempfile.mkdtemp(); crt, key = make_cert(d)
    sctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); sctx.minimum_version = ssl.TLSVersion.TLSv1_3
    sctx.load_cert_chain(crt, key)
    cctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT); cctx.minimum_version = ssl.TLSVersion.TLSv1_3
    cctx.load_verify_locations(crt)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0)); srv.listen(16); port = srv.getsockname()[1]
    stop = threading.Event(); threading.Thread(target=serve, args=(srv, sctx, stop), daemon=True).start()
    prof = PROFILES if a.profiles == "all" else [p for p in PROFILES if p[0] in a.profiles.split(",")]
    cc = open("/proc/sys/net/ipv4/tcp_congestion_control").read().strip()
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["profile", "rate_kbit", "rtt_ms", "budget", "rep", "seconds", "syn_retrans", "retrans_segs", "congestion_control"])
        for name, rate, rtt in prof:
            set_link(rate, rtt, port)
            attempt(port, cctx, 100)            # warm-up (not recorded)
            for b in map(int, a.budgets.split(",")):
                for r in range(a.reps):
                    c0 = counters(); t = attempt(port, cctx, b); c1 = counters()
                    w.writerow([name, rate, rtt, b, r, round(t, 4), c1[0] - c0[0], c1[1] - c0[1], cc]); f.flush()
    stop.set(); srv.close()


if __name__ == "__main__":
    main()
