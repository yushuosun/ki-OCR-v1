"""GT-blind effective-state policy. No dataset or scorer imports."""
import math,re

VERSION='OMNI_EFFECTIVE_FAILURE_CASCADE_R1_20261002'
RULE={
 'primary':'prospective1651+eager67 FULL1651_VIEW/B0; no historical Tele substitution',
 'fallback':'fixed existing full1651 PaddleOCR-VL1.6 cache; never per-page score selection',
 'normal_output':'preserve all bytes; NULL/NONE text is not a failure marker',
 'raw_normal_conversion_failure':'same current source/input/slot, stop termination, nonempty raw, empty/unconverted current output; existing native deterministic processor first',
 'failed_slot':'empty returned output or length/error/abort termination; same current source and exact predicted geometry; current payload must equal native processing of this failed raw, otherwise the old failure is superseded',
 'whole_page_fallback':'only current whitespace-empty page or current whole-page explicit failed state; no partial-page label may select whole-page fallback',
 'slot_fallback':'unique one-to-one positive-area overlap of compatible normalized predicted boxes, with all baseline compatible slots included in reverse ambiguity checks; no GT or content similarity',
 'patch':'unique exact nonempty Markdown interval; preserve surrounding bytes; empty/ambiguous interval abstains UNKNOWN',
 'missing_state_or_cache':'UNKNOWN, retain original and scoring denominator',
 'table_gate':'reuse existing syntax census; no new671-table validity experiment',
 'fixed_pages':1651,'gpu':0,'new_model_calls':0,
}
FAILED={'length','error','abort','failed','oom','timeout'}

def payload_type(typ):
 if typ in {'table','table_body'}:return 'table'
 if typ in {'equation','interline_equation'}:return 'equation'
 if typ in {'image','image_body'}:return 'image'
 return 'text'

def effective_state(current, raw, convert, explicit_page_failure=False):
 """The raw record must already have current provenance and exact slot binding."""
 if raw is None:return 'NORMAL_RETAIN' if isinstance(current,str) and current.strip() else 'UNKNOWN_STATE'
 reason=raw.get('finish_reason');value=raw.get('raw_text')
 if not isinstance(value,str):return 'UNKNOWN_RAW'
 try:converted=convert(value)
 except Exception:return 'UNKNOWN_CONVERSION_BINDING'
 if current!=converted:
  if isinstance(current,str) and current.strip():return 'SUPERSEDED_FAILURE_RETAIN' if reason in FAILED or not value.strip() else 'NORMAL_RETAIN'
  if reason=='stop' and value.strip():return 'DETERMINISTIC_RECOVERY_FIRST'
  return 'UNKNOWN_CURRENT_RAW_BINDING'
 if reason in FAILED:return 'EXPLICIT_FAILED_SLOT'
 if reason=='stop':return 'EMPTY_RETURNED_SLOT' if not value.strip() else 'NORMAL_RETAIN'
 return 'UNKNOWN_TERMINATION'

def recover_then_fallback(current,state,raw,convert,fallback):
 if state in {'NORMAL_RETAIN','SUPERSEDED_FAILURE_RETAIN'} or state.startswith('UNKNOWN'):return current,'RETAIN_'+state
 if state=='DETERMINISTIC_RECOVERY_FIRST':
  recovered=convert(raw['raw_text'])
  if isinstance(recovered,str) and recovered.strip():return recovered,'DETERMINISTIC_RECOVERED'
 if fallback is None:return current,'UNKNOWN_FALLBACK_CACHE_OR_BINDING'
 if not isinstance(fallback,str) or not fallback.strip():return current,'UNKNOWN_UNUSABLE_FALLBACK'
 return fallback,'PADDLE_FIXED_FALLBACK'

def box(q,size):
 if not isinstance(q,(list,tuple)) or len(q)!=4:return None
 w,h=size
 try:b=[float(q[0])/w,float(q[1])/h,float(q[2])/w,float(q[3])/h]
 except (TypeError,ValueError,ZeroDivisionError):return None
 return b if all(math.isfinite(x) for x in b) and 0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1 else None

def intersection(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))

def unique_overlap(target,baseline,backup):
 hits=[i for i,x in enumerate(backup) if x['kind']==target['kind'] and x.get('box') and target.get('box') and intersection(target['box'],x['box'])>0]
 if len(hits)!=1:return None,'UNKNOWN_BACKUP_OVERLAP_'+str(len(hits))
 i=hits[0];candidate=backup[i]
 reverse=[b for b in baseline if b['kind']==target['kind'] and b.get('box') and intersection(b['box'],candidate['box'])>0]
 if len(reverse)!=1:return None,'UNKNOWN_MERGED_BACKUP_OVERLAP_'+str(len(reverse))
 return candidate,'UNIQUE_ONE_TO_ONE_POSITIVE_OVERLAP'

def exact_patch(md,old,new):
 if not isinstance(old,str) or not old:return md,'UNKNOWN_EMPTY_MARKDOWN_ANCHOR'
 if md.count(old)!=1:return md,'UNKNOWN_NONUNIQUE_MARKDOWN_ANCHOR'
 pos=md.index(old);return md[:pos]+new+md[pos+len(old):],'PATCHED_EXACT_INTERVAL'

def native_layout(text,size):
 """Native parse syntax/rounding, without importing any model packages."""
 out=[];w,h=size
 for line in text.splitlines():
  m=re.fullmatch(r'<box:([\d\s]+)><label:(\w+)><([^>]+)>',line.strip())
  if not m:return None
  numbers=[int(x) for x in m[1].split()]
  if len(numbers)!=4:return None
  # Middle native geometry floors each1000-normalized coordinate to image pixels.
  coords=[int(numbers[0]*w/1000),int(numbers[1]*h/1000),int(numbers[2]*w/1000),int(numbers[3]*h/1000)]
  out.append({'kind':payload_type(m[2]),'type':m[2],'bbox':coords})
 return out
