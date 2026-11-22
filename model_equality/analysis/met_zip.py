"""Read members of a remote zip on Google Drive with HTTP range requests (no full download).
  python met_zip.py list                      -> table of contents summary
  python met_zip.py get <out dir> <regex>     -> extract only members whose name matches regex
Nothing is ever unpickled here."""
import io, os, re, sys, collections, urllib.request, zipfile

URL = "https://drive.usercontent.google.com/download?id=" + os.environ.get("MET_ID", "1EO61yiqZmb1kT019oUIoQWpwPPykwJ9f") + "&export=download&confirm=t"


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
        self.fetched = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for _ in range(5):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read(); break
            except Exception as e:
                err = e
        else:
            raise err
        self.pos += len(data); self.fetched += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


f = io.BufferedReader(RangeFile(URL), buffer_size=1 << 20)
z = zipfile.ZipFile(f)
infos = z.infolist()
if sys.argv[1] == "list":
    print(f"{len(infos)} members, total {sum(i.file_size for i in infos)/1e9:.1f} GB uncompressed")
    groups = collections.Counter(); sizes = collections.Counter()
    for i in infos:
        key = re.sub(r"\d+", "#", i.filename)
        groups[key] += 1; sizes[key] += i.compress_size
    for k, c in groups.most_common(40):
        print(f"  {c:6d} x {k}   ({sizes[k]/1e6:.0f} MB compressed)")
    for i in infos[:15]:
        print("   e.g.", i.filename, i.compress_size)
    for i in infos:
        if not i.filename.startswith("samples/") and not re.search(r"\.pkl$", i.filename):
            print("   other:", i.filename, i.compress_size)
else:
    out, pat = sys.argv[2], re.compile(sys.argv[3])
    sel = [i for i in infos if pat.search(i.filename) and not i.is_dir()]
    print(f"{len(sel)} members match, {sum(i.compress_size for i in sel)/1e6:.0f} MB compressed")
    if len(sys.argv) > 4 and sys.argv[4] == "go":
        os.makedirs(out, exist_ok=True)
        for i in sel:
            z.extract(i, out)
        print("extracted to", out, "fetched", f.raw.fetched / 1e6, "MB")
