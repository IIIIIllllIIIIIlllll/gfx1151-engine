"""Production capacity, 64K PP, image MTP and continuation cross-binary checks."""
from pathlib import Path
import argparse,ast,json,socket,struct,sys
from ppnight_serve_checks import Engine,ROOT
from qwentok import Tokenizer

def image_frame(name):
 p=ROOT/f'data/vision-ref/{name}.patches.npy'
 with p.open('rb') as f:
  assert f.read(6)==b'\x93NUMPY'
  version=f.read(2);n=struct.unpack('<H' if version[0]==1 else '<I',f.read(2 if version[0]==1 else 4))[0]
  h=ast.literal_eval(f.read(n).decode());body=f.read()
 assert h['descr']=='<f4' and not h['fortran_order'],h
 patches=h['shape'][0];assert len(body)==patches*1536*4
 return struct.pack('<IIQ',0x56494D31,patches,len(body))+body

def main():
 ap=argparse.ArgumentParser();ap.add_argument('label');ap.add_argument('binary');ap.add_argument('--flag',action='append',default=[])
 args=ap.parse_args();tk=Tokenizer();e=Engine(args.label,args.binary,dict(x.split('=',1) for x in args.flag))
 def turn(q,img=False):
  return tk.encode('<|im_start|>user\n'+('<|vision_start|>' if img else ''))+([248056]*2040 if img else [])+tk.encode(('<|vision_end|>\n' if img else '')+q+'<|im_end|>\n<|im_start|>assistant\n')
 try:
  g=json.loads((ROOT/'data/qsa-oracle/32768.json').read_text())
  e.request('warm-8192',g['prompt_ids'][:8192],8,1)
  long=g['prompt_ids']*2
  a=e.request('long65536-spec',long,24,1)
  b=e.request('long65536-serial',long,24,0)
  print('LONG_SPEC_SERIAL',a['generated_ids']==b['generated_ids'],flush=True)
  assert a['generated_ids']==b['generated_ids']
  frame=image_frame('table');extra=' MROPE 1 1 68 120 VIMG 1'
  for mode in (1,0):
   with socket.create_connection(('127.0.0.1',18730),timeout=10) as conn:
    p=turn('How many columns does this table have? Answer with just the number.',True)
    a=e.request(f'image-{mode}-first',p,96,mode,conn,extra,frame)
    assert int(a['done'][10])==0
    p2=p+a['generated_ids']+[248046]+turn('And what is the header of the last column? Answer briefly.')
    b=e.request(f'image-{mode}-cont',p2,96,mode,conn,extra,frame)
    assert int(b['done'][10])>=len(p),b['done']
    assert 'COL-08' in tk.decode(b['generated_ids']),tk.decode(b['generated_ids'])
    # Identical shape with different image bytes must force fresh prefill.
    c=e.request(f'image-{mode}-swap',p,96,mode,conn,extra,image_frame('text'))
    assert int(c['done'][10])==0,c['done']
  e.save();print('ACCEPT PASS',args.label,flush=True)
 finally:e.close()

if __name__=='__main__':main()
