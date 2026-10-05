"""Generate a public, project-authored one-page PNG; no fonts or packages."""
import argparse, hashlib, struct, zlib
from pathlib import Path
def build():
    w,h=768,768
    pixels=bytearray([255])*(w*h)
    def rect(x,y,a,b):
        for yy in range(y,b):
            for xx in range(x,a): pixels[yy*w+xx]=0
    glyphs={"A":["01110","10001","10001","11111","10001","10001","10001"],"B":["11110","10001","10001","11110","10001","10001","11110"],"1":["00100","01100","00100","00100","00100","00100","01110"],"2":["01110","10001","00001","00010","00100","01000","11111"]}
    def word(s,x,y,scale=5):
        for c in s:
            for j,row in enumerate(glyphs[c]):
                for i,on in enumerate(row):
                    if on=="1":rect(x+i*scale,y+j*scale,x+(i+1)*scale,y+(j+1)*scale)
            x+=6*scale
    word("AB12",72,72,7)
    for x in (64,320,576):rect(x,224,x+1,545)
    for y in (224,384,544):rect(64,y,577,y+1)
    for s,x,y in [("A1",96,280),("B2",352,280),("A2",96,440),("B1",352,440)]:word(s,x,y)
    def chunk(t,d):return struct.pack(">I",len(d))+t+d+struct.pack(">I",zlib.crc32(t+d)&0xffffffff)
    raw=b"".join(b"\x00"+pixels[y*w:(y+1)*w] for y in range(h))
    return b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,0,0,0,0))+chunk(b"IDAT",zlib.compress(raw,9))+chunk(b"IEND",b"")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise SystemExit("Refusing overwrite")
    a.output.parent.mkdir(parents=True,exist_ok=True);raw=build();a.output.write_bytes(raw)
    print("Public synthetic one-page input SHA256: "+hashlib.sha256(raw).hexdigest())
if __name__=="__main__":main()
